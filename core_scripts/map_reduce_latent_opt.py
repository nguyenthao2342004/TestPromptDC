import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
import torch.multiprocessing as mp

# --- TỰ ĐỘNG CẤU HÌNH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(repo_root)

from unidepth.models.unidepthv1.unidepthv1 import UniDepthV1

def mapper_worker(rank, world_size, image_chunks, output_queue):
    device = torch.device(f'cuda:{rank}')
    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14").to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False # Snowflake: Đóng băng Foundation Model
    
    local_results = []
    for img_path in tqdm(image_chunks, desc=f"🚀 GPU_{rank}", leave=False):
        filename = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        img_res = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (616, 462))
        img_t = torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        metas = {"intrinsic": torch.eye(3, device=device).unsqueeze(0)}

        # Tạo Sparse Depth giả lập (Pseudo-LiDAR)
        with torch.no_grad():
            init_out = model({"image": img_t}, metas)
            mask = torch.zeros_like(init_out["depth"]).bool()
            indices = torch.randperm(init_out["depth"].numel())[:500] 
            mask.view(-1)[indices] = True
            sparse_t = init_out["depth"] * mask

        # Khởi tạo Latent (Fire - Chỉ phần này được tối ưu)
        latent_opt = nn.Parameter(torch.zeros_like(img_t)) 
        optimizer = optim.Adam([latent_opt], lr=5e-3)
        
        for _ in range(10): 
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                prompted_input = img_t + torch.tanh(latent_opt) * 0.1
                outputs = model({"image": prompted_input}, metas)
                loss = F.l1_loss(outputs["depth"][mask], sparse_t[mask])
            loss.backward()
            optimizer.step()

        local_results.append({"id": filename, "mae": loss.item(), "depth_map": outputs["depth"].detach().cpu().numpy().squeeze()})
        torch.cuda.empty_cache()
    output_queue.put(local_results)

def main():
    INPUT_DIR = '/kaggle/input/my-data-vietnam-traffic/anh_video_2s/anh_trich_xuat_2s/'
    OUTPUT_DIR = "/kaggle/working/TestPromptDC/latent_opt_results/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_files = sorted([os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))])[:500]
    world_size = torch.cuda.device_count()
    chunks = np.array_split(all_files, world_size)
    
    output_queue = mp.Queue()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=mapper_worker, args=(rank, world_size, chunks[rank].tolist(), output_queue))
        p.start()
        processes.append(p)
        
    all_results = []
    for _ in range(world_size): all_results.extend(output_queue.get())
    for p in processes: p.join()

    avg_mae = np.mean([r['mae'] for r in all_results])
    print(f"✅ Avg MAE: {avg_mae:.10f}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

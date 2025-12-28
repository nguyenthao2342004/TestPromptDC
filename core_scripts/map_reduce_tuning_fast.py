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

class VisualPromptLayer(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(1, 3, h, w))
    def forward(self, x):
        return x + torch.tanh(self.prompt) * 0.05

def mapper_worker(rank, world_size, image_chunks, output_queue):
    device = torch.device(f'cuda:{rank}')
    
    # KHÔNG dùng .half() ở đây để tránh lỗi LayerNorm
    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14").to(device)
    model.eval()
    
    local_results = []
    
    for img_path in tqdm(image_chunks, desc=f"🚀 GPU_{rank}", leave=False):
        filename = os.path.basename(img_path)
        
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (616, 462))
        # Giữ tensor ở dạng float32
        img_tensor = torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        metas = {"intrinsic": torch.eye(3, device=device).unsqueeze(0)}
        
        # Sử dụng autocast để tự động tối ưu tốc độ mà không gây lỗi kiểu dữ liệu
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                init_out = model({"image": img_tensor}, metas)
                pseudo_gt = init_out["depth"]
                mask = torch.zeros_like(pseudo_gt).bool()
                indices = torch.randperm(pseudo_gt.numel())[:400]
                mask.view(-1)[indices] = True
                sparse_points = pseudo_gt * mask

        vp_layer = VisualPromptLayer(462, 616).to(device)
        optimizer = optim.Adam(vp_layer.parameters(), lr=3e-3)
        scaler = torch.cuda.amp.GradScaler() 
        
        for _ in range(8):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                prompted_image = vp_layer(img_tensor)
                outputs = model({"image": prompted_image}, metas)
                loss = F.l1_loss(outputs["depth"][mask], sparse_points[mask])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        local_results.append({
            "id": filename, "mae": loss.item(), 
            "depth_map": outputs["depth"].detach().cpu().numpy().squeeze()
        })
        
        # Giải phóng bộ nhớ GPU sau mỗi ảnh
        torch.cuda.empty_cache()
        
    output_queue.put(local_results)

def main():
    input_dir = '/kaggle/input/my-data-vietnam-traffic/anh_video_2s/anh_trich_xuat_2s/'
    output_dir = '/kaggle/working/TestPromptDC/mapreduce_results_fast/'
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])
    all_files = all_files[:500] 
    
    world_size = torch.cuda.device_count()
    chunks = np.array_split(all_files, world_size)
    
    print(f"📦 [SPLITTING]: Dùng {world_size} GPU xử lý 500 ảnh...")
    
    output_queue = mp.Queue()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=mapper_worker, args=(rank, world_size, chunks[rank].tolist(), output_queue))
        p.start()
        processes.append(p)
        
    all_mapper_outputs = []
    for _ in range(world_size):
        all_mapper_outputs.extend(output_queue.get())
        
    for p in processes:
        p.join()

    all_maes = [x['mae'] for x in all_mapper_outputs]
    avg_mae = np.mean(all_maes)
    
    for res in tqdm(all_mapper_outputs, desc="💾 Đang lưu ảnh"):
        save_path = os.path.join(output_dir, f"dense_{res['id']}")
        depth_norm = ((res['depth_map'] - res['depth_map'].min()) / (res['depth_map'].max() - res['depth_map'].min() + 1e-6) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        cv2.imwrite(save_path, depth_color)

    print(f"\n✅ HOÀN THÀNH! Avg MAE: {avg_mae:.10f}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

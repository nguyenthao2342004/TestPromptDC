import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- TỰ ĐỘNG CẤU HÌNH HỆ THỐNG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(repo_root)

# Tạo liên kết ảo 'unidepth' ngay trong script nếu chưa có
unidepth_link = os.path.join(repo_root, 'unidepth')
unidepth_custom = os.path.join(repo_root, 'unidepth_custom')
if not os.path.exists(unidepth_link):
    os.system(f"ln -s {unidepth_custom} {unidepth_link}")

# Vá lỗi xformers (attn_bias -> attn_mask & fix scale)
import xformers.ops
def patched_attention(q, k, v, attn_bias=None, p=0.0, scale=None, **kwargs):
    q_pt, k_pt, v_pt = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
    if scale is not None: q_pt = q_pt * scale
    return F.scaled_dot_product_attention(q_pt, k_pt, v_pt, attn_mask=attn_bias, dropout_p=p).transpose(1, 2).contiguous()
xformers.ops.memory_efficient_attention = patched_attention

from unidepth.models.unidepthv1.unidepthv1 import UniDepthV1

class VisualPromptLayer(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(1, 3, h, w))
    def forward(self, x):
        return x + torch.tanh(self.prompt) * 0.1

def mapper_work(model, image_list, device, mapper_id):
    local_results = []
    pbar = tqdm(image_list, desc=f"👷 Mapper_{mapper_id:02d}", leave=False)
    for img_path in pbar:
        filename = os.path.basename(img_path)
        raw_img = Image.open(img_path).convert("RGB")
        img_pil = raw_img.resize((616, 462), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        metas = {"intrinsic": torch.eye(3, device=device).unsqueeze(0)}
        
        with torch.no_grad():
            init_out = model({"image": img_tensor}, metas)
            pseudo_gt = init_out["depth"]
            mask = torch.zeros_like(pseudo_gt).bool()
            indices = torch.randperm(pseudo_gt.numel())[:1000]
            mask.view(-1)[indices] = True
            sparse_points = pseudo_gt * mask

        vp_layer = VisualPromptLayer(462, 616).to(device).float()
        optimizer = optim.Adam(vp_layer.parameters(), lr=2e-3)
        for i in range(11):
            optimizer.zero_grad()
            prompted_image = vp_layer(img_tensor)
            outputs = model({"image": prompted_image}, metas)
            pred_depth = outputs["depth"]
            loss = F.l1_loss(pred_depth[mask], sparse_points[mask])
            loss.backward(); optimizer.step()

        local_results.append({
            "id": filename, "mae": loss.item(), 
            "depth_map": pred_depth.detach().cpu().numpy().squeeze()
        })
        torch.cuda.empty_cache()
    return local_results

def main():
    device = 'cuda'
    input_dir = '/kaggle/input/my-raw-dataset/raw_images/'
    output_dir = '/kaggle/working/TestPromptDC/mapreduce_results/'
    
    IMAGES_PER_MAPPER = 10
    TOTAL_IMAGES = 50 # Chạy thử 5 ảnh trước cho nhanh
    
    all_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])
    all_files = all_files[:TOTAL_IMAGES]
    
    num_mappers = int(np.ceil(len(all_files) / IMAGES_PER_MAPPER))
    splits = np.array_split(all_files, num_mappers)
    
    print(f"📦 SPLITTING: Tổng {len(all_files)} ảnh.")
    print(f"📦 MAPPING: Chia thành {num_mappers} Mappers.")

    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14").to(device).float()
    model.eval()

    shuffled_maes = []
    shuffled_data = []

    for i, chunk in enumerate(splits):
        print(f"\n▶️ Khởi động Mapper_{i+1:02d}...")
        mapper_output = mapper_work(model, chunk.tolist(), device, mapper_id=i+1)
        for item in mapper_output:
            shuffled_maes.append(item['mae'])
            shuffled_data.append(item)
        print(f"✅ Mapper_{i+1:02d} hoàn thành.")

    avg_mae = np.mean(shuffled_maes)
    os.makedirs(output_dir, exist_ok=True)
    for res in shuffled_data:
        plt.imsave(os.path.join(output_dir, f"dense_{res['id']}"), res['depth_map'], cmap='magma_r')

    print("\n" + "="*45)
    print(f"🏆 FINAL REPORT")
    print(f"   - Mappers executed: {num_mappers}")
    print(f"   - Total images: {len(shuffled_data)}")
    print(f"   - Global Avg MAE: {avg_mae:.10f}")
    print("="*45)

if __name__ == "__main__":
    main()

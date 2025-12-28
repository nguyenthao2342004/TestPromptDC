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

# SETUP ĐƯỜNG DẪN HỆ THỐNG
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(repo_root)

# VÁ LỖI XFORMERS
import xformers.ops
def patched_attention(q, k, v, attn_bias=None, p=0.0, scale=None, **kwargs):
    q_pt = q.transpose(1, 2).contiguous()
    k_pt = k.transpose(1, 2).contiguous()
    v_pt = v.transpose(1, 2).contiguous()
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

def prepare_final_frame(path, device):
    raw_img = Image.open(path).convert("RGB")
    new_h, new_w = 462, 616 
    img_pil = raw_img.resize((new_w, new_h), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    return img_pil, img_tensor

def main():
    device = torch.device('cuda')
    print("❄️ Đang nạp Foundation Model...")
    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14").to(device).float()
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    input_folder = '/kaggle/input/my-data-vietnam-traffic/anh_video_2s/anh_trich_xuat_2s/'
    output_folder = os.path.join(repo_root, "results_fig_b")
    os.makedirs(output_folder, exist_ok=True)

    all_images = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))])[:3]

    for filename in all_images:
        img_path = os.path.join(input_folder, filename)
        img_pil, img_tensor = prepare_final_frame(img_path, device)

        with torch.no_grad():
            metas = {"intrinsic": torch.eye(3, device=device).unsqueeze(0)}
            init_out = model({"image": img_tensor}, metas)
            pseudo_gt = init_out["depth"]
            mask = torch.zeros_like(pseudo_gt).bool()
            indices = torch.randperm(pseudo_gt.numel())[:1000]
            mask.view(-1)[indices] = True
            sparse_points = pseudo_gt * mask

        vp_layer = VisualPromptLayer(462, 616).to(device).float()
        optimizer = optim.Adam(vp_layer.parameters(), lr=2e-3)

        print(f"🔥 Tuning: {filename}...")
        for i in range(201):
            optimizer.zero_grad()
            prompted_image = vp_layer(img_tensor)
            outputs = model({"image": prompted_image}, metas)
            pred_depth = outputs["depth"]
            loss = F.l1_loss(pred_depth[mask], sparse_points[mask])
            loss.backward(); optimizer.step()

        # Save result
        plt.figure(figsize=(22, 12))
        plt.subplot(2, 2, 1); plt.imshow(img_pil); plt.axis('off')
        plt.subplot(2, 2, 2); plt.imshow(np.zeros_like(img_pil))
        y, x = np.where(sparse_points.squeeze().cpu().numpy() > 0)
        plt.scatter(x, y, c=sparse_points.squeeze().cpu().numpy()[y, x], s=4, cmap='magma_r'); plt.axis('off')
        plt.subplot(2, 2, 3); plt.imshow(vp_layer.prompt.detach().squeeze().cpu().permute(1,2,0).numpy()); plt.axis('off')
        plt.subplot(2, 2, 4); plt.imshow(pred_depth.detach().squeeze().cpu().numpy(), cmap='magma_r'); plt.axis('off')
        
        plt.savefig(os.path.join(output_folder, f"res_{filename}"))
        plt.close()
    print(f"✅ Đã lưu kết quả vào {output_folder}")

if __name__ == "__main__":
    main()

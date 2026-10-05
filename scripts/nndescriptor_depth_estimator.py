import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class DenseMultiScaleDescriptor(nn.Module):
    """
    Extracts dense multi-scale descriptors by concatenating normalized local patches
    from an image pyramid (3 scales).
    """
    def __init__(self, patch_size=3):
        super(DenseMultiScaleDescriptor, self).__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2)

    def extract_scale_descriptors(self, x):
        B, C, H, W = x.shape
        patches = self.unfold(x)
        patches = patches.view(B, C * (self.patch_size ** 2), H, W)
        return F.normalize(patches, p=2, dim=1)

    def forward(self, img_tensor):
        # Scale 1: Full resolution
        desc1 = self.extract_scale_descriptors(img_tensor)
        
        # Scale 2: Half resolution
        img_s2 = F.avg_pool2d(img_tensor, kernel_size=2, stride=2)
        desc2 = self.extract_scale_descriptors(img_s2)
        desc2 = F.interpolate(desc2, size=img_tensor.shape[2:], mode='bilinear', align_corners=True)
        
        # Scale 3: Quarter resolution
        img_s3 = F.avg_pool2d(img_s2, kernel_size=2, stride=2)
        desc3 = self.extract_scale_descriptors(img_s3)
        desc3 = F.interpolate(desc3, size=img_tensor.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate multi-scale descriptors along channel axis
        multi_scale_desc = torch.cat([desc1, desc2, desc3], dim=1)
        return F.normalize(multi_scale_desc, p=2, dim=1)


class DenseEpipolarMatcher(nn.Module):
    def __init__(self, max_disparity=128):
        super(DenseEpipolarMatcher, self).__init__()
        self.max_disparity = max_disparity

    def forward(self, desc_L, desc_R):
        B, C, H, W = desc_L.shape
        D = self.max_disparity + 1
        
        # 1. Compute Cost Volume across disparity d in [0, max_disparity]
        cost_volume = torch.zeros((B, D, H, W), device=desc_L.device)
        
        for d in range(D):
            if d == 0:
                dot_prod = (desc_L * desc_R).sum(dim=1)
            else:
                desc_R_shifted = F.pad(desc_R[:, :, :, :-d], (d, 0, 0, 0), value=0)
                dot_prod = (desc_L * desc_R_shifted).sum(dim=1)
            
            cost_volume[:, d, :, :] = 1.0 - dot_prod

        # 2. Primary best match (d*) and minimum cost (c1)
        c1, d_star = torch.min(cost_volume, dim=1)
        
        # 3. 2nd lowest local minimum (c2) for Peak Ratio confidence
        d_indices = torch.arange(D, device=desc_L.device).view(1, D, 1, 1)
        mask = torch.abs(d_indices - d_star.unsqueeze(1)) <= 2
        
        masked_cost_volume = cost_volume.clone()
        masked_cost_volume[mask] = float('inf')
        c2, _ = torch.min(masked_cost_volume, dim=1)
        c2 = torch.where(torch.isinf(c2), c1 + 1e-4, c2)

        # 4. Evaluate Confidence Metrics
        S_ratio = torch.clamp(1.0 - (c1 / (c2 + 1e-5)), 0.0, 1.0)
        S_cost = torch.exp(-c1 / 0.2)

        # Sharpness of local minimum
        d_prev = torch.clamp(d_star - 1, 0, D - 1)
        d_next = torch.clamp(d_star + 1, 0, D - 1)
        
        c_prev = torch.gather(cost_volume, 1, d_prev.unsqueeze(1)).squeeze(1)
        c_next = torch.gather(cost_volume, 1, d_next.unsqueeze(1)).squeeze(1)
        
        sharpness = c_prev - 2.0 * c1 + c_next
        S_sharp = torch.clamp(sharpness / 0.5, 0.0, 1.0)

        # Final composite confidence map
        confidence_map = S_ratio * S_cost * S_sharp

        return d_star.squeeze(0), confidence_map.squeeze(0)


def visualize_results(left_img, right_img, disparity_map, confidence_map):
    """
    Renders Left Image, Right Image, Disparity Map, and Confidence Map in a 2x2 grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Left Image
    axes[0, 0].imshow(left_img)
    axes[0, 0].set_title("Left Image")
    axes[0, 0].axis("off")

    # Right Image
    axes[0, 1].imshow(right_img)
    axes[0, 1].set_title("Right Image")
    axes[0, 1].axis("off")

    # Disparity Map
    im_disp = axes[1, 0].imshow(disparity_map, cmap="jet")
    axes[1, 0].set_title("Disparity Map (d)")
    axes[1, 0].axis("off")
    fig.colorbar(im_disp, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Pixels")

    # Confidence Map
    im_conf = axes[1, 1].imshow(confidence_map, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Confidence Map S(x, y)")
    axes[1, 1].axis("off")
    fig.colorbar(im_conf, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Confidence [0, 1]")

    plt.tight_layout()
    plt.show()


# --- Main Execution Script ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate synthetic stereo pair
    H, W = 240, 320
    np.random.seed(42)
    left_img_np = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    
    # Create synthetic disparity (Shift central square by 15px)
    right_img_np = left_img_np.copy()
    right_img_np[60:180, 80:240] = np.roll(left_img_np[60:180, 80:240], shift=-15, axis=1)

    # Convert to PyTorch Tensors
    left_tensor = torch.from_numpy(left_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    right_tensor = torch.from_numpy(right_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Model inference
    descriptor_extractor = DenseMultiScaleDescriptor(patch_size=3).to(device)
    matcher = DenseEpipolarMatcher(max_disparity=128).to(device)

    with torch.no_grad():
        desc_L = descriptor_extractor(left_tensor)
        desc_R = descriptor_extractor(right_tensor)
        disparity_map, confidence_map = matcher(desc_L, desc_R)

    # Convert tensors to NumPy for Matplotlib visualization
    disp_np = disparity_map.cpu().numpy()
    conf_np = confidence_map.cpu().numpy()

    # Display plots
    visualize_results(left_img_np, right_img_np, disp_np, conf_np)
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
    def __init__(self, max_disparity=128, edge_thr=0.05, edge_dilate=1, smooth_iter=8, context_scale=1, smoother='edge'):
        super(DenseEpipolarMatcher, self).__init__()
        self.max_disparity = max_disparity
        self.smoother      = smoother      # 'edge' - context_smoother_edge (8 neighbors), 'box' - context_smoother (3x3 average)
        self.context_scale = context_scale # edges are computed on the image downscaled by this factor
        self.edge_thr      = edge_thr      # gradient scale (image in [0,1]) : smooth probability exp(-grad/edge_thr)
        self.edge_dilate   = edge_dilate   # edge band half width in pixels - closes diagonal gaps so cost can not leak across
        self.smooth_iter   = smooth_iter   # 3x3 smoothing iterations - effective radius in pixels
        self.edge_img      = None          # last context, for visualization

        sobel_x             = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
        self.register_buffer('sobel', torch.stack([sobel_x, sobel_x.t()]).unsqueeze(1))   # (2,1,3,3)

    def context_extractor(self, left_img):
        """
        Edge context of the left image.
        left_img : (B,C,H,W) in [0,1]
        returns edge_img (B,1,H,W) float in (0,1] : probability of a smooth neighborhood = exp(-grad_img / edge_thr)
                 1 - flat pixel, -> 0 - pixel on / near a strong edge
        """
        H, W     = left_img.shape[2:]
        s        = self.context_scale
        gray     = left_img.mean(dim=1, keepdim=True)
        # downscale by s (area average - also suppresses pixel noise), gradient at low resolution, upscale back by s
        gray_s   = F.avg_pool2d(gray, kernel_size=s, stride=s, ceil_mode=True) if s > 1 else gray
        grad     = F.conv2d(F.pad(gray_s, (1, 1, 1, 1), mode='replicate'), self.sobel)
        grad_img = torch.sqrt((grad ** 2).sum(dim=1, keepdim=True))
        grad_img = F.interpolate(grad_img, size=(H, W), mode='bilinear', align_corners=False) if s > 1 else grad_img

        edge_img = torch.exp(-grad_img / self.edge_thr)
        # if self.edge_dilate > 0:    # widen edges : min filter of the smooth probability
        #     k        = 2 * self.edge_dilate + 1
        #     edge_img = -F.max_pool2d(-edge_img, kernel_size=k, stride=1, padding=self.edge_dilate)

        # debug : visualize edge context
        self.grad_img = grad_img
        self.edge_img = edge_img
        return edge_img

    def context_smoother(self, cost_volume, edge_img):
        """
        Edge aware low pass filter of the cost volume : repeated 3x3 averaging weighted by the smooth probability.
        Each pixel moves toward the weighted average of its neighbors in proportion to its own smooth probability,
        so edge pixels (probability ~0) do not contribute and keep their own cost - information does not cross edges.
        cost_volume : (B,D,H,W), edge_img : (B,1,H,W) from context_extractor
        returns smoothed cost volume (B,D,H,W)
        """
        w      = edge_img
        den    = F.avg_pool2d(w, kernel_size=3, stride=1, padding=1)          # same for every disparity
        has_nb = den > 1e-6                                                   # some smooth support around the pixel
        cost   = cost_volume
        for _ in range(self.smooth_iter):
            num    = F.avg_pool2d(cost * w, kernel_size=3, stride=1, padding=1)
            avg    = torch.where(has_nb, num / den.clamp_min(1e-6), cost)
            cost   = w * avg + (1.0 - w) * cost
        return cost

    def context_smoother_edge(self, cost_volume, edge_img):
        """
        Edge aware integration of the cost volume over the 8-neighborhood, repeated smooth_iter times.
        Each pixel p adds the cost of every neighbor q with weight w_pq = min(edge_img[p], edge_img[q]) :
            cost(p) <- (cost(p) + sum_q a_q * w_pq * cost(q)) / (1 + sum_q a_q * w_pq)
        a_q = 1 for the 4 direct neighbors and 1/sqrt(2) for the diagonal ones (distance).
        w_pq is symmetric : a strong edge on either side blocks the exchange, so cost does not flow into
        or out of edge pixels and does not cross edges. Image borders are replicated.
        cost_volume : (B,D,H,W), edge_img : (B,1,H,W) from context_extractor
        returns smoothed cost volume (B,D,H,W)
        """
        H, W      = cost_volume.shape[2:]
        offsets   = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
        shift     = lambda x, dy, dx: x[:, :, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]   # neighbor (y+dy, x+dx) of a padded map

        # neighbor weights - fixed over the iterations
        w_pad     = F.pad(edge_img, (1, 1, 1, 1), mode='replicate')
        #weights   = [(dy, dx, (1.0 if dy == 0 or dx == 0 else 0.5 ** 0.5) * torch.minimum(edge_img, shift(w_pad, dy, dx)))  for dy, dx in offsets]
        weights   = [(dy, dx, shift(w_pad, dy, dx)) for dy, dx in offsets]

        den       = 1.0 + sum(w for _, _, w in weights)

        cost      = cost_volume
        for _ in range(self.smooth_iter):
            c_pad = F.pad(cost, (1, 1, 1, 1), mode='replicate')
            num   = cost.clone()
            for dy, dx, w in weights:
                num += w * shift(c_pad, dy, dx)
            cost  = num / den
        return cost

    def forward(self, desc_L, desc_R, left_img=None):
        """
        left_img : optional (B,C,H,W) in [0,1]. When given, the cost volume is smoothed inside
                   smooth regions (context_extractor + context_smoother) before disparity and confidence.
        """
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

        # 1b. Edge aware smoothing of the cost volume - used by all the processing below
        self.edge_img = self.context_extractor(left_img)
        if self.smoother == 'edge':
            cost_volume = self.context_smoother_edge(cost_volume, self.edge_img)
        else:
            cost_volume = self.context_smoother(cost_volume, self.edge_img)

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
        S_cost = torch.exp(-c1 / 0.1)

        # Sharpness of local minimum
        d_prev = torch.clamp(d_star - 1, 0, D - 1)
        d_next = torch.clamp(d_star + 1, 0, D - 1)
        
        c_prev = torch.gather(cost_volume, 1, d_prev.unsqueeze(1)).squeeze(1)
        c_next = torch.gather(cost_volume, 1, d_next.unsqueeze(1)).squeeze(1)
        
        #sharpness = c_prev - 2.0 * c1 + c_next  # is positive if sharp
        S_sharp   = 2*c1 / (c_prev + c_next + 1e-5) #1 - torch.clamp(torch.exp(sharpness / 0.1), 0.0, 1.0)

        # Final composite confidence map
        confidence_map = S_ratio * S_cost * S_sharp

        # debug : confidence components for visualization
        self.confidence_parts = {'S_ratio': S_ratio.squeeze(0), 'S_cost': S_cost.squeeze(0), 'S_sharp': S_sharp.squeeze(0)}

        return d_star.squeeze(0), confidence_map.squeeze(0)


def visualize_results(left_img, right_img, disparity_map, confidence_map):
    """
    Renders Left Image, Right Image, Disparity Map, and Confidence Map in a 2x2 grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)

    # Left Image
    axes[0, 0].imshow(left_img)
    axes[0, 0].set_title("Left Image")
    #axes[0, 0].axis("off")

    # Right Image
    axes[0, 1].imshow(right_img)
    axes[0, 1].set_title("Right Image")
    #axes[0, 1].axis("off")

    # Disparity Map
    im_disp = axes[1, 0].imshow(disparity_map, cmap="jet")
    axes[1, 0].set_title("Disparity Map (d)")
    #axes[1, 0].axis("off")
    fig.colorbar(im_disp, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Pixels")

    # Confidence Map
    im_conf = axes[1, 1].imshow(confidence_map, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Confidence Map S(x, y)")
    #axes[1, 1].axis("off")
    fig.colorbar(im_conf, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Confidence [0, 1]")

    plt.tight_layout()
   
def show_subset(img_list, ttl_list, vmin=None, vmax=None, save_path='', fig_name='', col_num=3, adjust_index=[]):
    "show some images"

    img_num  = len(img_list)
    row_num  = int(img_num/col_num) 
    col_num  = int(np.ceil(img_num/row_num))
    fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True)
    axes      = axes.reshape((row_num,col_num))

    for k in range(img_num):
        ri, ci = int(k / col_num), k % col_num
        # if vmin is None or vmax is None:
        if k in adjust_index:
            vmin, vmax = np.percentile(img_list[k][img_list[k] > 2], [5, 95])
        #     vmin, vmax = np.percentile(img_list[k], [10, 90])

        pcm = axes[ri, ci].imshow(img_list[k], vmin=vmin, vmax=vmax)
        axes[ri, ci].set_title(ttl_list[k])     
        #fig.colorbar(pcm, ax=axes[ri, ci])  

    
    plt.show(block=False)

class DataSource:
    """
    Generates synthetic stereo pairs for testing the depth estimation pipeline.
    """
    def __init__(self, H=240, W=320):
        self.H = H
        self.W = W

    def generate_stereo_pair(self, img_type = 1):

        if img_type == 1:  # Test one image against image - shifted square
            np.random.seed(42)
            left_img = (np.random.rand(self.H, self.W, 3) * 255).astype(np.uint8)
            
            # Create synthetic disparity (Shift central square by 15px)
            right_img = left_img.copy()
            right_img[60:180, 80:240] = np.roll(left_img[60:180, 80:240], shift=-15, axis=1)

        elif img_type == 4:  # home

            image2 = cv2.imread(r"C:\Work\Data\DepthRS\Corr\l3_Infrared.png", cv2.IMREAD_GRAYSCALE)
            image1 = cv2.imread(r"C:\Work\Data\DepthRS\Corr\r3_Infrared.png", cv2.IMREAD_GRAYSCALE)
            left_img, right_img = np.uint8(image2), np.uint8(image1)

        elif img_type == 11: # at office
            #imgC      = cv2.pyrDown(cv2.imread(r"C:\Work\Data\DepthRS\Corr\image_d16_000.png", cv2.IMREAD_UNCHANGED))
            imgC        = cv2.imread(r"C:\Work\Data\DepthRS\Corr\image_d16_000.png", cv2.IMREAD_UNCHANGED)
            image2      = cv2.pyrDown(imgC[:,:,0])
            image1      = cv2.pyrDown(imgC[:,:,1])   
            left_img, right_img = np.uint8(image2), np.uint8(image1)         

        elif img_type == 19:  # Test one image against image - slanted surface
            shift       = np.array([0, 15]) * 1
            image1      = np.random.rand(240, 320, 3) * 5 + 50
            image2      = np.roll(image1, shift, axis=(0, 1))  

            image1[80:160,120:150,:] = 100
            image2[80:160,140:160,:] = 100 # left

            left_img, right_img = np.uint8(image2), np.uint8(image1)

        elif img_type == 21:  # Test one image against image - squares at depth
            shift       = np.array([0, 15]) * 1
            image1      = np.random.rand(240, 320, 3) * 5 + 50
            image2      = np.roll(image1, shift, axis=(0, 1))  
            
            image1[120:130,120:130] = 100
            image2[120:130,130:140] = 100
            image1[80:130,150:160] = 100
            image2[80:130,170:180] = 100 
            left_img, right_img = np.uint8(image2), np.uint8(image1)

        elif img_type == 623: # tests from 405 setup 3 boxes at different heights
            image2 = cv2.pyrDown(cv2.imread(r"C:\Work\Data\FoundationStereo\picking\d405\imageL_d16_013.png", flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE))
            image1 = cv2.pyrDown(cv2.imread(r"C:\Work\Data\FoundationStereo\picking\d405\imageR_d16_013.png", flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE))
            left_img, right_img = np.uint8(image2), np.uint8(image1)

        # if dimensions of image_left and image_right are 2D make them by replication 3 channels
        if len(left_img.shape) == 2:
            left_img = np.stack([left_img] * 3, axis=-1)
        if len(right_img.shape) == 2:
            right_img = np.stack([right_img] * 3, axis=-1)

        return left_img, right_img

# --- Main Execution Script ---
if __name__ == "__main__":
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source                  = DataSource(H=240, W=320)

    # Generate synthetic stereo pair
    left_img_np, right_img_np = source.generate_stereo_pair(img_type=4)

    # Convert to PyTorch Tensors
    left_tensor             = torch.from_numpy(left_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    right_tensor            = torch.from_numpy(right_img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Model inference
    descriptor_extractor    = DenseMultiScaleDescriptor(patch_size=3).to(device)
    matcher                 = DenseEpipolarMatcher(max_disparity=128).to(device)

    with torch.no_grad():
        desc_L                      = descriptor_extractor(left_tensor)
        desc_R                      = descriptor_extractor(right_tensor)
        disparity_map, confidence_map = matcher(desc_L, desc_R, left_img=left_tensor)

    # Convert tensors to NumPy for Matplotlib visualization
    disp_np = disparity_map.cpu().numpy()
    conf_np = confidence_map.cpu().numpy()
    grad_np = matcher.grad_img[0, 0].cpu().numpy()
        
    # Confidence components : panels in order S_ratio, S_cost, S_sharp, S = S_ratio * S_cost * S_sharp
    parts_np = {k: v.cpu().numpy() for k, v in matcher.confidence_parts.items()}

    # Edge context used by context_smoother
    plt.figure("Gradient")
    plt.imshow(grad_np, cmap="gray", vmin=0, vmax=1)
    plt.title("Gradient")

    plt.figure("Context")
    plt.imshow(matcher.edge_img[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    plt.title("Edge context : 1 - smooth, 0 - edge")


    # Display plots
    img_list = [left_img_np, right_img_np, disp_np, conf_np]
    ttl_list = ["Left Image", "Right Image", "Disparity Map (d)", "Confidence Map S(x, y)"]
    show_subset(img_list, ttl_list, col_num=2)

    img_list = [parts_np['S_ratio'], parts_np['S_cost'], parts_np['S_sharp'], conf_np]
    ttl_list = ["S_ratio", "S_cost", "S_sharp", "Confidence Map S(x, y)"]
    show_subset(img_list, ttl_list, vmin=0, vmax=1, col_num=2)

    plt.show()
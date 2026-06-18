"""
Fine-tune FastFoundationStereo on the Pickle dataset.

The Pickle dataset provides:
  - realsense/{idx}/mono0.png  : left IR image  (uint8, 480x640)
  - realsense/{idx}/mono1.png  : right IR image (uint8, 480x640)
  - zivid/{idx}/depthmap_mm.png: ground-truth depth in mm (Zivid scanner, 1024x1224)

Strategy:
  - Freeze the ViT-L backbone (model.feature) to prevent overfitting on small datasets.
  - Train everything else with RAFT-style sequence loss over GRU iterations.
  - IR uint8 images are replicated to 3 channels.
  - Pickle depth is resized to RealSense image resolution before disparity conversion.
  - Depth is converted to disparity: disp = BF / depth_mm.

Usage:
  cd /path/to/Fast-FoundationStereo
  python scripts/finetune_pickle.py
"""

import os, sys, logging
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from core.utils.utils import InputPadder
import Utils as U
from scripts.data_manager_pickle import DataSource


# ── constants ────────────────────────────────────────────────────────────────

PICKLE_DIR    = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/pickle_datasets/pickle_basic_objects'

# adiroha machine
PICKLE_DIR    = (
    r"/mnt/validation/VIDB/IQ_AUTO/IQLab0/2026_06"
    r"/yg_pickle/2026-06-18--10-01-03/Pickle_Scene_Capture_336222073841"
    r"/data path.xlsx"
)


# MODEL_PATH = f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth'
# OUT_PATH   = f'{code_dir}/../weights/20-30-48/model_finetuned_inbolt-20260415.pth'
MODEL_PATH = f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth'
OUT_PATH   = f'{code_dir}/../weights/23-36-37/model_finetuned_pickle.pth'


# BF         = 49.8624*385.73  # D435 - focal_px * baseline_mm (calibrated from camera)  # D435 - focal_px * baseline_mm (calibrated from camera)
#BF         = 50.102706998586 * 385.509887695312 # new data 2
#BF         = 50.102706998586 * 642.4910888671875 # new data 3 2026-05-18
EPOCHS      = 50
LR          = 2e-5
ITERS       = 8          # GRU iterations (same as inference)
GAMMA       = 0.9        # sequence loss weight decay
TRAIN_RATIO = 0.75
SPLIT_SEED  = 0

# -- Helpers -------------------------------

def measure_variability(img, levele_num = 2):
    "estimate min and max values / std using 7x7 image kernel"

    """
    Finds the minimum and maximum values within the specified kernel size for each pixel in the image.

    Args:
        image: The input image as a NumPy array.
        kernel_size: The size of the square kernel (e.g., 7 for a 7x7 kernel).

    Returns:
        A tuple containing:
            - min_values: A NumPy array of the minimum values within each kernel.
            - max_values: A NumPy array of the maximum values within each kernel.
    """
    img_size    = img.shape
    for k in range(levele_num):
        img         = cv2.pyrDown(img)
        
    img         = np.uint8(img)
    kernel_size = 7

    # Create a kernel of ones for min/max filtering
    kernel      = np.ones((kernel_size, kernel_size), np.uint8) 

    # Find minimum values within the kernel
    min_values = cv2.erode(img, kernel)

    # Find maximum values within the kernel
    max_values = cv2.dilate(img, kernel)

    # diference
    max_diff   = cv2.absdiff(max_values , min_values)

    # debug
    # Display the results using Matplotlib
    #self.show_image_plt(img, min_values, max_values, max_diff)
    for k in range(levele_num):
        max_diff    = cv2.pyrUp(max_diff)

    max_diff    = cv2.resize(max_diff, img_size[::-1])

    return max_diff.astype(np.float32)

def find_flat_regions(disp_gt, valid):
    """Identify planar regions in the ground-truth disparity map using RANSAC."""
    # convert disp_gt to numpy for variability measurement
    disp_gt_np         = disp_gt # (H, W)
    valid_variability  = valid

    # Fit a plane to the valid disparities using RANSAC
    disp_variability  = measure_variability(disp_gt_np, levele_num=2)  # (H, W) variability measure (e.g., std or max-min)
    valid_variability = valid_variability & (disp_variability < 50.0)  # only consider low-variability pixels             

    return valid_variability

def extract_patches(left, right, depth, valid, patch_size=512, min_valid_ratio=0.30, max_tries=30):
    """Randomly crop a `patch_size`x`patch_size` patch from the image.

    Resamples a new random location if the fraction of valid pixels inside the
    crop is below `min_valid_ratio`. After `max_tries` attempts, returns the
    best (highest-valid-ratio) patch found.

    Args:
        left, right: (H, W, ...) numpy arrays (image channels last or 2D).
        depth:       (H, W) numpy array.
        valid:       (H, W) bool numpy array.
        patch_size:  side length of the square patch.
        min_valid_ratio: minimum fraction of valid pixels required to accept.
        max_tries:   maximum number of random crops to try.

    Returns:
        Tuple (left_p, right_p, depth_p, valid_p) cropped to patch_size x patch_size.
    """
    H, W = valid.shape[:2]
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image ({H}x{W}) smaller than patch_size ({patch_size}).")

    total = patch_size * patch_size
    #best_ratio = -1.0
    best_yx = (0, 0)

    for _ in range(max_tries):
        y = np.random.randint(0, H - patch_size + 1)
        x = np.random.randint(0, W - patch_size + 1)
        valid_num = float(valid[y:y + patch_size, x:x + patch_size].sum()) 
        ratio = valid_num / total
        best_yx = (y, x)
        if ratio >= min_valid_ratio:
            break

    y, x = best_yx
    sl = (slice(y, y + patch_size), slice(x, x + patch_size))
    return left[sl], right[sl], depth[sl], valid[sl]

# ── dataset ──────────────────────────────────────────────────────────────────

class PickleDataset(Dataset):
    def __init__(self, root, train_mode=True):
        self.source = DataSource(train_mode=train_mode)
        n = self.source.init_directory(excel_path = root)
        logging.info(f"DataSource found {n} samples in {root}")

    def __len__(self):
        return self.source.__len__()

    def __getitem__(self, idx):
        #data  = self.source.get_item(idx)  # 2026-04
        #data  = self.source.get_item_projected(idx) 
        data  = self.source.get_item_and_scene_projected(idx)  # 2026-05-18 with plane fitting
        left  = data['ir_left_img']
        right = data['ir_right_img']
        depth = data['depth_img']   #
        bf    = data['bf']                 # float32, mm*px

        # Resize depth to match RealSense stereo image resolution
        h, w  = left.shape[:2]
        if depth.shape != (h, w):
            #depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            raise ValueError(f"Depth shape {depth.shape} does not match left image shape {(h, w)}.")

        # IR uint8 → float [0, 255], replicate to 3-channel pseudo-RGB
        left  = np.clip(left.astype(np.float32),  0, 255)
        right = np.clip(right.astype(np.float32), 0, 255)
        left  = np.stack([left,  left,  left],  axis=-1)  # H x W x 3
        right = np.stack([right, right, right], axis=-1)
        valid = depth > 0

        #left, right, depth, valid = extract_patches(left, right, depth, valid)        
        # depth (mm) → disparity (pixels):  disp = focal * baseline / depth
        disp  = np.zeros_like(depth, dtype=np.float32)
        
        disp[valid] = bf / depth[valid]

        #valid = find_flat_regions(disp, valid)
        #valid = find_flat_regions(depth, valid)
        left, right, disp, valid = extract_patches(left, right, disp, valid) 

        left_t  = torch.from_numpy(left).permute(2, 0, 1).float()   # (3, H, W)
        right_t = torch.from_numpy(right).permute(2, 0, 1).float()  # (3, H, W)
        disp_t  = torch.from_numpy(disp).unsqueeze(0).float()       # (1, H, W)
        valid_t = torch.from_numpy(valid).unsqueeze(0)               # (1, H, W) bool

        return left_t, right_t, disp_t, valid_t


# ── loss ─────────────────────────────────────────────────────────────────────


def sequence_loss(disp_preds, disp_gt, valid, gamma=GAMMA):
    """RAFT-style weighted sum of smooth-L1 losses over GRU iterations."""
    n    = len(disp_preds)
    loss = 0.0
    for i, pred in enumerate(disp_preds):
        w  = gamma ** (n - 1 - i)
        gt = disp_gt
        v  = valid
        if pred.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode='nearest')
            v  = F.interpolate(valid.float(), size=pred.shape[-2:], mode='nearest').bool()
        #loss = loss + w * F.smooth_l1_loss(pred[v], gt[v])
        loss = loss + w * F.huber_loss(pred[v], gt[v], reduction='mean', delta=3.0)
    return loss


def evaluate_split_loss(model, dataloader):
    """Evaluate average sequence loss over a dataloader (no gradient updates)."""
    if len(dataloader) == 0:
        return float('nan')

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for left, right, disp_gt, valid in dataloader:
            left, right = left.cuda(), right.cuda()
            disp_gt, valid = disp_gt.cuda(), valid.cuda()

            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left_p, right_p = padder.pad(left, right)

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                _init_disp, disp_preds = model.forward(
                    left_p, right_p, iters=ITERS, test_mode=False
                )
                disp_preds = [padder.unpad(p) for p in disp_preds]
                loss = sequence_loss(disp_preds, disp_gt, valid)

            total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    U.set_logging_format()
    U.set_seed(0)

    # load full model object (weights + architecture)
    logging.info(f"Loading model from {MODEL_PATH}")
    model = torch.load(MODEL_PATH, map_location='cuda', weights_only=False)

    # freeze the ViT-L backbone — with only 24 samples it would overfit
    for param in model.feature.parameters():
        param.requires_grad = False
    logging.info("ViT backbone frozen.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable: {trainable:,} / {total:,} parameters")

    model = model.cuda().train()
    logging.info("Model on single GPU.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler('cuda')

    dataset = PickleDataset(PICKLE_DIR, train_mode=True)
    n_total = len(dataset)

    if n_total < 2:
        raise RuntimeError(f"Need at least 2 samples for a 75/25 train/test split, got {n_total}.")

    n_train = int(round(TRAIN_RATIO * n_total))
    n_train = min(max(1, n_train), n_total - 1)
    n_test = n_total - n_train

    split_generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=split_generator)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    logging.info(
        f"Random split with seed={SPLIT_SEED}: total={n_total}, train={len(train_set)} ({100.0*len(train_set)/n_total:.1f}%), "
        f"test={len(test_set)} ({100.0*len(test_set)/n_total:.1f}%)"
    )

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for left, right, disp_gt, valid in train_loader:
            #valid = find_flat_regions(disp_gt, valid)
            left, right = left.cuda(), right.cuda()
            disp_gt, valid = disp_gt.cuda(), valid.cuda()

            # pad so H and W are divisible by 32
            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left_p, right_p = padder.pad(left, right)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
                _init_disp, disp_preds = model.forward(
                    left_p, right_p, iters=ITERS, test_mode=False
                )
                disp_preds = [padder.unpad(p) for p in disp_preds]
                loss = sequence_loss(disp_preds, disp_gt, valid)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_eval_error = evaluate_split_loss(model, train_loader)
        torch.cuda.empty_cache()
        test_eval_error = evaluate_split_loss(model, test_loader)
        torch.cuda.empty_cache()

        logging.info(
            f"Epoch {epoch+1:3d}/{EPOCHS}  train_loss={train_loss:.4f}  "
            f"train_eval_error={train_eval_error:.4f}  test_eval_error={test_eval_error:.4f}"
        )

        if test_eval_error < best_loss:
            best_loss = test_eval_error
            torch.save(model, OUT_PATH.replace('.pth', f'_epoch_{epoch+1:03d}.pth'))
            logging.info(f"  → saved best model (test_eval_error={best_loss:.4f})")

    final_train_error = evaluate_split_loss(model, train_loader)
    final_test_error = evaluate_split_loss(model, test_loader)
    logging.info(f"Final train error: {final_train_error:.4f}")
    logging.info(f"Final test error:  {final_test_error:.4f}")
    logging.info(f"Training complete. Best test error: {best_loss:.4f}")
    logging.info(f"Model saved to {OUT_PATH}")


if __name__ == '__main__':
    main()

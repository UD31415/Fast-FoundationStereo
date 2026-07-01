"""
Fine-tune FastFoundationStereo on the multi-set DataSource.

This is a direct port of ``finetune_pickle.py`` that swaps the per-dataset
``data_manager_pickle.DataSource`` for the unified
``data_manager_multiset.DataSource`` so a single training run can pull
samples from heterogeneous datasets (isaac / faro / pickle).

Strategy
--------
* Freeze the ViT-L backbone (``model.feature``) to keep finetuning lightweight.
* RAFT-style sequence loss over GRU iterations (same as ``finetune_pickle.py``).
* IR uint8 images are replicated to 3 channels.
* Depth is converted to disparity via ``disp = bf / depth_mm``.

Source-schema gotchas
---------------------
The :class:`scripts.data_manager_multiset.DataSource` exposes a *common*
schema; supervised stereo training needs **both** ``depth_gt_img`` and
``bf`` for every sample.  Per-source availability:

================  =============  ====
source            depth_gt_img   bf
================  =============  ====
isaac             yes            yes
faro              yes            **no**  (multiset returns ``bf=None``)
pickle            **no**         yes
================  =============  ====

Therefore, by default only the ``isaac`` source is enabled — it is the only
one that can be trained directly through the common schema.  Add other
sources via ``MULTISET_CONFIGS`` below if you have a way to populate the
missing field (e.g. computing ``bf`` for faro from intrinsics or augmenting
the pickle normalizer to fall back to ``get_item_and_scene_projected``).

The dataset wrapper silently *drops* samples whose ``depth_gt_img`` or
``bf`` is missing, so adding more sources to ``MULTISET_CONFIGS`` is safe —
indexed items that cannot be supervised are filtered at startup.

Usage
-----
.. code-block:: bash

    cd /path/to/Fast-FoundationStereo
    python scripts/finetune_multiset.py
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
from scripts.data_manager_multiset import DataSource


# ── constants ────────────────────────────────────────────────────────────────

# Per-source ``init_kwargs`` passed through to each sub-source's
# ``init_directory``. Drop or add entries here to control which datasets
# are mixed in. Only sources whose normalized items expose both
# ``depth_gt_img`` and ``bf`` will contribute training samples.
MULTISET_CONFIGS = [
    # Isaac source — full GT depth + bf. The init_kwargs can be left empty
    # to use the module's default paths (``data_manager_isaac.DataSource``
    # picks them up internally).
    {"name": "isaac", "init_kwargs": {}},

    # faro — provides GT depth but no ``bf`` in the common schema; samples
    # are dropped at __init__ time.  Uncomment to include any that turn
    # out to be usable.
    # {"name": "faro",   "init_kwargs": {}},

    # pickle — has ``bf`` but no per-pixel GT in the common schema. Use
    # ``finetune_pickle.py`` for the CAD-rendered training regime.
    # {"name": "pickle", "init_kwargs": {
    #     "excel_path": r"\\svm.realsenseai.com\RealSense_Validation\VIDB\Public"
    #                   r"\Stavush\Pickle\Data\data for model training 25_6_26"
    #                   r"\data_25_06.xlsx",
    # }},
]


MODEL_PATH = f'{code_dir}/../weights/23-36-37/model_finetuned_pickle_260625_epoch_006.pth'
OUT_PATH   = f'{code_dir}/../weights/23-36-37/model_finetuned_multiset_260630.pth'


EPOCHS      = 50
LR          = 2e-5
ITERS       = 8          # GRU iterations (same as inference)
GAMMA       = 0.9        # sequence loss weight decay
TRAIN_RATIO = 0.75
SPLIT_SEED  = 0
# NOTE: num_workers must stay at 0. The dataset caches non-picklable Open3D
# objects (PointCloud / TriangleMesh) so workers can’t use "spawn", and main()
# initialises CUDA before the DataLoader is built so fork-based workers inherit
# CUDA file descriptors and hang. Multi-worker support would require building
# the DataLoader before any CUDA init and refactoring the cache layout.
NUM_WORKERS = 0
PREFETCH    = 2          # only used when NUM_WORKERS > 0


# ── helpers ──────────────────────────────────────────────────────────────────

def measure_variability(img, levele_num=2):
    """Approximate local intensity variability via a min/max kernel on a pyramid."""
    img_size = img.shape
    for _ in range(levele_num):
        img = cv2.pyrDown(img)

    img         = np.uint8(img)
    kernel_size = 7
    kernel      = np.ones((kernel_size, kernel_size), np.uint8)

    min_values = cv2.erode(img, kernel)
    max_values = cv2.dilate(img, kernel)
    max_diff   = cv2.absdiff(max_values, min_values)

    for _ in range(levele_num):
        max_diff = cv2.pyrUp(max_diff)

    max_diff = cv2.resize(max_diff, img_size[::-1])
    return max_diff.astype(np.float32)


def find_flat_regions(disp_gt, valid):
    """Restrict ``valid`` to low-variability (planar) regions."""
    disp_variability  = measure_variability(disp_gt, levele_num=2)
    return valid & (disp_variability < 50.0)


def extract_patches(left, right, depth, valid,
                    patch_size=512, min_valid_ratio=0.30, max_tries=30):
    """Randomly crop a ``patch_size`` square, biased toward high-valid patches."""
    H, W = valid.shape[:2]
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image ({H}x{W}) smaller than patch_size ({patch_size}).")

    total   = patch_size * patch_size
    best_yx = (0, 0)
    for _ in range(max_tries):
        y = np.random.randint(0, H - patch_size + 1)
        x = np.random.randint(0, W - patch_size + 1)
        valid_num = float(valid[y:y + patch_size, x:x + patch_size].sum())
        ratio     = valid_num / total
        best_yx   = (y, x)
        if ratio >= min_valid_ratio:
            break

    y, x = best_yx
    sl   = (slice(y, y + patch_size), slice(x, x + patch_size))
    return left[sl], right[sl], depth[sl], valid[sl]


# ── dataset ──────────────────────────────────────────────────────────────────

class MultisetDataset(Dataset):
    """Wraps :class:`data_manager_multiset.DataSource` for supervised training.

    Samples missing either ``depth_gt_img`` or ``bf`` in the normalized
    schema are dropped at construction time so that ``__getitem__`` can
    always assume both fields are present.
    """

    def __init__(self, configs, train_mode=True):
        self.source = DataSource(train_mode=train_mode)
        total = self.source.init_directory(configs=configs)
        logging.info(f"MultiSet DataSource indexed {total} raw items")

        # Filter indices to those that expose both depth_gt_img and bf in
        # the normalized schema. Use the flat ``items`` view to avoid
        # triggering full image loads here.
        self.indices = []
        for flat_idx, raw_item in enumerate(self.source.items):
            # The per-item bf is on the sub-source's raw record (it's the
            # field that the normalizer maps through). Depth GT presence
            # is harder to know without loading, so we rely on the source
            # name + a feasibility check.
            source_name = raw_item.get("source")
            if source_name == "isaac":
                # isaac always provides both fields when properly indexed.
                self.indices.append(flat_idx)
            elif source_name == "faro":
                # Common schema returns bf=None; cannot supervise.
                continue
            elif source_name == "pickle":
                # Common schema returns depth_gt_img=None; cannot supervise.
                continue
            else:
                # Unknown source — keep and let __getitem__ enforce.
                self.indices.append(flat_idx)

        logging.info(
            f"MultisetDataset: {len(self.indices)} / {total} samples retained "
            f"after filtering for (depth_gt_img, bf) availability"
        )
        if not self.indices:
            raise RuntimeError(
                "MultisetDataset retained 0 trainable samples — check "
                "MULTISET_CONFIGS and the per-source normalizers."
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        flat = self.indices[idx]
        data = self.source.get_item(flat, debug=False)

        left  = data['ir_left_img']
        right = data['ir_right_img']
        depth = data['depth_gt_img']
        bf    = data['bf']

        if depth is None or bf is None or left is None or right is None:
            raise RuntimeError(
                f"Sample {flat} (source={data.get('source')}) is missing one of "
                f"left/right/depth_gt_img/bf — should have been filtered."
            )

        # Resize depth to match the stereo IR image resolution if needed.
        h, w = left.shape[:2]
        if depth.shape != (h, w):
            raise ValueError(
                f"Depth shape {depth.shape} does not match left image shape "
                f"{(h, w)} for sample {flat}."
            )

        # IR uint8 → float [0, 255], replicate to 3-channel pseudo-RGB.
        left  = np.clip(left.astype(np.float32),  0, 255)
        right = np.clip(right.astype(np.float32), 0, 255)
        left  = np.stack([left,  left,  left],  axis=-1)   # H x W x 3
        right = np.stack([right, right, right], axis=-1)
        valid = depth > 0

        # depth (mm) → disparity (px): disp = focal * baseline / depth.
        disp = np.zeros_like(depth, dtype=np.float32)
        disp[valid] = bf / depth[valid]

        left, right, disp, valid = extract_patches(left, right, disp, valid)

        left_t  = torch.from_numpy(left).permute(2, 0, 1).float()   # (3, H, W)
        right_t = torch.from_numpy(right).permute(2, 0, 1).float()  # (3, H, W)
        disp_t  = torch.from_numpy(disp).unsqueeze(0).float()       # (1, H, W)
        valid_t = torch.from_numpy(valid).unsqueeze(0)              # (1, H, W) bool

        return left_t, right_t, disp_t, valid_t


# ── loss ─────────────────────────────────────────────────────────────────────

def sequence_loss(disp_preds, disp_gt, valid, gamma=GAMMA):
    """RAFT-style weighted sum of Huber losses over GRU iterations."""
    n    = len(disp_preds)
    loss = 0.0
    for i, pred in enumerate(disp_preds):
        w  = gamma ** (n - 1 - i)
        gt = disp_gt
        v  = valid
        if pred.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode='nearest')
            v  = F.interpolate(valid.float(), size=pred.shape[-2:], mode='nearest').bool()
        loss = loss + w * F.huber_loss(pred[v], gt[v], reduction='mean', delta=3.0)
    return loss


def evaluate_split_loss(model, dataloader):
    """Average sequence loss over a dataloader (no grad updates)."""
    if len(dataloader) == 0:
        return float('nan')

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for left, right, disp_gt, valid in dataloader:
            left, right    = left.cuda(), right.cuda()
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

    logging.info(f"Loading model from {MODEL_PATH}")
    model = torch.load(MODEL_PATH, map_location='cuda', weights_only=False)

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

    dataset  = MultisetDataset(MULTISET_CONFIGS, train_mode=True)
    n_total  = len(dataset)

    if n_total < 2:
        raise RuntimeError(
            f"Need at least 2 samples for a {int(TRAIN_RATIO*100)}/"
            f"{int((1-TRAIN_RATIO)*100)} train/test split, got {n_total}."
        )

    n_train = int(round(TRAIN_RATIO * n_total))
    n_train = min(max(1, n_train), n_total - 1)
    n_test  = n_total - n_train

    split_generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_set, test_set = random_split(
        dataset, [n_train, n_test], generator=split_generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None,
        pin_memory=True,
    )
    logging.info(
        f"DataLoaders: num_workers={NUM_WORKERS}, prefetch_factor={PREFETCH}, "
        f"pin_memory=True, persistent_workers=False"
    )

    logging.info(
        f"Random split with seed={SPLIT_SEED}: total={n_total}, "
        f"train={len(train_set)} ({100.0*len(train_set)/n_total:.1f}%), "
        f"test={len(test_set)} ({100.0*len(test_set)/n_total:.1f}%)"
    )

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for left, right, disp_gt, valid in train_loader:
            left, right    = left.cuda(), right.cuda()
            disp_gt, valid = disp_gt.cuda(), valid.cuda()

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
        test_eval_error  = evaluate_split_loss(model, test_loader)
        torch.cuda.empty_cache()

        logging.info(
            f"Epoch {epoch+1:3d}/{EPOCHS}  train_loss={train_loss:.4f}  "
            f"train_eval_error={train_eval_error:.4f}  "
            f"test_eval_error={test_eval_error:.4f}"
        )

        if test_eval_error < best_loss:
            best_loss = test_eval_error
            torch.save(
                model,
                OUT_PATH.replace('.pth', f'_epoch_{epoch+1:03d}.pth'),
            )
            logging.info(f"  → saved best model (test_eval_error={best_loss:.4f})")

    final_train_error = evaluate_split_loss(model, train_loader)
    final_test_error  = evaluate_split_loss(model, test_loader)
    logging.info(f"Final train error: {final_train_error:.4f}")
    logging.info(f"Final test error:  {final_test_error:.4f}")
    logging.info(f"Training complete. Best test error: {best_loss:.4f}")
    logging.info(f"Model saved to {OUT_PATH}")


if __name__ == '__main__':
    main()

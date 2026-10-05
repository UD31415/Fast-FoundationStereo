"""
Distill Fast-FoundationStereo (teacher) into the Shazam MobileNet volume fusion network (student).

The student (shazam_mobile_net.MobileNetVolumeFusion) merges the multiscale feature volumes of
ShazamDepthEstimator.multiscale_feature_volumes into one disparity probability volume + confidence.

Stage A - build_teacher_cache :
  For every Pickle frame run FFS once on the full IR pair and save
  left, right, teacher disparity, CAD GT disparity (bf / depth_cad_projected), bf  ->  CACHE_DIR/{idx:05d}.npz
  Frames already cached are skipped. The teacher is released from the GPU afterwards.

Stage B - finetune_mobile_net :
  Random crops of the cached frames, feature volumes computed on the crop (CPU, DataLoader workers),
  loss = KL(Laplace(teacher) || P) + w_l1 * smoothL1(soft-argmax, teacher)
       + w_gt * huber(soft-argmax, CAD GT) + w_conf * BCE(conf, |argmax - teacher| < 1)

Usage:
  cd /path/to/Fast-FoundationStereo
  python scripts/finetune_mobile_net.py
"""

import os, sys, glob, logging
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from shazam_mobile_net import (MobileNetVolumeFusion, install_shazam_stubs, volumes_to_tensor,
                               save_checkpoint, count_parameters)
install_shazam_stubs()
from shazam_depth_estimator import ShazamDepthEstimator  # noqa: E402


# ── constants ────────────────────────────────────────────────────────────────

# new data 2026-06-26
PICKLE_DIR   = (
    r"\\svm.realsenseai.com\RealSense_Validation\VIDB\Public\Stavush\Pickle\Data\data for model training 25_6_26"
    r"\data_25_06.xlsx"
)

TEACHER_PATH = f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth'
OUT_PATH     = f'{code_dir}/../weights/mobile_net/shazam_mobile_net.pth'
CACHE_DIR    = f'{code_dir}/../cache/mobile_net_teacher'

MAX_FRAMES   = None         # None - all frames of the manifest
EPOCHS       = 40
LR           = 1e-3
BATCH_SIZE   = 2
CROP         = (192, 320)   # (h, w) : w > max disparity, both divisible by 4 (3 pyramid levels)
CROPS_PER_FRAME = 4         # random crops per frame and epoch
TRAIN_RATIO  = 0.8
SPLIT_SEED   = 0
ITERS        = 8            # FFS GRU iterations
NUM_WORKERS  = 4            # stage B reads only npz files - workers are safe here
MAX_DISP     = 128

W_KL, W_L1, W_GT, W_CONF = 1.0, 0.1, 0.3, 0.1
LAPLACE_B    = 1.0          # width of the teacher target distribution (px)
CONF_TOL     = 1.0          # |argmax - teacher| below this is a confident pixel


# ── stage A : teacher cache ──────────────────────────────────────────────────

@torch.no_grad()
def teacher_disparity(model, left, right):
    "FFS disparity (H,W) for a single IR pair - same preprocessing as benchmark_pickle_shazam.infer_depth_mm"
    from core.utils.utils import InputPadder
    import Utils as U
    l               = np.clip(left.astype(np.float32), 0, 255)
    r               = np.clip(right.astype(np.float32), 0, 255)
    l_t             = torch.as_tensor(np.stack([l] * 3, axis=-1))[None].permute(0, 3, 1, 2).cuda()
    r_t             = torch.as_tensor(np.stack([r] * 3, axis=-1))[None].permute(0, 3, 1, 2).cuda()
    padder          = InputPadder(l_t.shape, divis_by=32, force_square=False)
    l_t, r_t        = padder.pad(l_t, r_t)
    with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
        disp        = model.forward(l_t, r_t, iters=ITERS, test_mode=True)
    disp            = padder.unpad(disp.float())
    return disp.cpu().numpy().reshape(left.shape[:2]).clip(0, None).astype(np.float32)


def depth_to_disparity(depth_mm, bf):
    disp            = np.zeros_like(depth_mm, dtype=np.float32)
    valid           = depth_mm > 0
    disp[valid]     = bf / depth_mm[valid]
    return disp


def build_teacher_cache(pickle_dir=PICKLE_DIR, teacher_path=TEACHER_PATH, cache_dir=CACHE_DIR, max_frames=MAX_FRAMES):
    "run FFS once per Pickle frame and store the result. returns the list of cached npz files"
    from scripts.data_manager_pickle import DataSource
    os.makedirs(cache_dir, exist_ok=True)

    source          = DataSource(train_mode=True)
    n               = source.init_directory(excel_path=pickle_dir)
    n               = n if max_frames is None else min(n, max_frames)
    logging.info(f"DataSource found {n} samples in {pickle_dir}")

    path_of         = lambda i: os.path.join(cache_dir, f'{i:05d}.npz')
    missing         = [i for i in range(n) if not os.path.exists(path_of(i))]
    logging.info(f"Teacher cache : {n - len(missing)} cached, {len(missing)} to compute")

    if missing:
        logging.info(f"Loading teacher from {teacher_path}")
        teacher     = torch.load(teacher_path, map_location='cpu', weights_only=False).cuda().eval()
        for j, idx in enumerate(missing):
            data    = source.get_item_and_scene_projected(idx)
            left, right = data['ir_left_img'], data['ir_right_img']
            if left.ndim == 3:
                left, right = left[..., 0], right[..., 0]
            bf      = float(data['bf'])
            gt_mm   = data['depth_cad_projected'].astype(np.float32)
            if gt_mm.shape != left.shape[:2]:
                logging.warning(f"Item {idx}: GT {gt_mm.shape} != image {left.shape[:2]}; skipping")
                continue
            np.savez_compressed(path_of(idx),
                                left=np.clip(left, 0, 255).astype(np.uint8), right=np.clip(right, 0, 255).astype(np.uint8),
                                disp_teacher=teacher_disparity(teacher, left, right),
                                disp_gt=depth_to_disparity(gt_mm, bf), bf=np.float32(bf))
            if (j + 1) % 20 == 0 or (j + 1) == len(missing):
                logging.info(f"  teacher {j + 1}/{len(missing)}")
        del teacher
        torch.cuda.empty_cache()

    return [path_of(i) for i in range(n) if os.path.exists(path_of(i))]


# ── stage B : dataset ────────────────────────────────────────────────────────

class DistillDataset(Dataset):
    "crops of cached frames + Shazam feature volumes computed on the crop"
    def __init__(self, npz_files, crop=CROP, crops_per_frame=CROPS_PER_FRAME, train=True, max_disparity=MAX_DISP):
        self.files          = list(npz_files)
        self.crop           = crop
        self.crops_per_frame = crops_per_frame if train else 1
        self.train          = train
        self.max_disparity  = max_disparity
        self.estimator      = None      # created per worker

    def __len__(self):
        return len(self.files) * self.crops_per_frame

    def _crop_origin(self, H, W, h, w, idx):
        if self.train:
            return np.random.randint(0, H - h + 1), np.random.randint(0, W - w + 1)
        return (H - h) // 2, (W - w) // 2      # deterministic test crop

    def __getitem__(self, idx):
        if self.estimator is None:
            self.estimator  = ShazamDepthEstimator()
        d               = np.load(self.files[idx // self.crops_per_frame])
        left, right     = d['left'].astype(np.float32), d['right'].astype(np.float32)
        t_disp, gt_disp = d['disp_teacher'], d['disp_gt']

        H, W            = left.shape
        h, w            = min(self.crop[0], H) // 4 * 4, min(self.crop[1], W) // 4 * 4
        y, x            = self._crop_origin(H, W, h, w, idx)
        sl              = (slice(y, y + h), slice(x, x + w))
        left, right, t_disp, gt_disp = left[sl], right[sl], t_disp[sl], gt_disp[sl]

        if self.train and np.random.rand() < 0.5:      # photometric jitter, same for both views
            a, b        = np.random.uniform(0.8, 1.2), np.random.uniform(-15, 15)
            left, right = np.clip(left * a + b, 0, 255), np.clip(right * a + b, 0, 255)

        cost            = self.estimator.multiscale_feature_volumes(left, right, max_disparity=self.max_disparity)
        cost_t, img_t   = volumes_to_tensor(cost, left)

        # the match x - d must lie inside the crop, and d inside the volume
        cols            = np.arange(w, dtype=np.float32)[None, :]
        mask_t          = (t_disp > 0.5) & (t_disp < self.max_disparity - 1) & (cols >= t_disp)
        mask_gt         = (gt_disp > 0.5) & (gt_disp < self.max_disparity - 1) & (cols >= gt_disp)

        return (cost_t.half(), img_t,
                torch.from_numpy(t_disp)[None], torch.from_numpy(gt_disp)[None],
                torch.from_numpy(mask_t)[None], torch.from_numpy(mask_gt)[None])


# ── loss ─────────────────────────────────────────────────────────────────────

def laplace_target(t_disp, D, b=LAPLACE_B):
    "discretized Laplace distribution over D centered on the teacher disparity : (B,1,H,W) -> (B,D,H,W)"
    d_index         = torch.arange(D, device=t_disp.device, dtype=torch.float32).view(1, D, 1, 1)
    return torch.softmax(-torch.abs(d_index - t_disp) / b, dim=1)


def distillation_loss(out, t_disp, gt_disp, mask_t, mask_gt):
    logits, disp, conf = out['logits'], out['disp'], out['conf']
    D               = logits.shape[1]
    zero            = logits.sum() * 0.0
    m_t             = mask_t[:, 0]

    if m_t.any():
        target      = laplace_target(t_disp, D)
        kl          = (target * (torch.log(target + 1e-9) - torch.log_softmax(logits, dim=1))).sum(dim=1)
        loss_kl     = kl[m_t].mean()
        loss_l1     = F.smooth_l1_loss(disp[mask_t], t_disp[mask_t])
        d_hard      = torch.argmax(logits, dim=1, keepdim=True).float()
        conf_target = (torch.abs(d_hard - t_disp) < CONF_TOL).float()
        loss_conf   = F.binary_cross_entropy(conf[mask_t].clamp(1e-6, 1 - 1e-6), conf_target[mask_t])
    else:
        loss_kl = loss_l1 = loss_conf = zero

    loss_gt         = F.huber_loss(disp[mask_gt], gt_disp[mask_gt], delta=3.0) if mask_gt.any() else zero

    loss            = W_KL * loss_kl + W_L1 * loss_l1 + W_GT * loss_gt + W_CONF * loss_conf
    return loss, {'kl': loss_kl.item(), 'l1': loss_l1.item(), 'gt': loss_gt.item(), 'conf': loss_conf.item()}


# ── evaluation ───────────────────────────────────────────────────────────────

def disparity_errors(pred, ref, mask):
    "sums for EPE / bad-1 / bad-3 accumulation"
    err             = torch.abs(pred - ref)[mask]
    return err.sum().item(), (err > 1).sum().item(), (err > 3).sum().item(), err.numel()


@torch.no_grad()
def evaluate(model, loader):
    "EPE / bad-1 / bad-3 (%) of the student vs the teacher and vs CAD GT, and mean loss"
    model.eval()
    acc             = {'teacher': np.zeros(4), 'gt': np.zeros(4)}
    loss_sum, n     = 0.0, 0
    for cost, img, t_disp, gt_disp, mask_t, mask_gt in loader:
        cost, img   = cost.cuda().float(), img.cuda()
        t_disp, gt_disp, mask_t, mask_gt = t_disp.cuda(), gt_disp.cuda(), mask_t.cuda(), mask_gt.cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            out     = model(cost, img)
        loss, _     = distillation_loss(out, t_disp, gt_disp, mask_t, mask_gt)
        loss_sum   += loss.item(); n += 1
        acc['teacher'] += disparity_errors(out['disp'], t_disp, mask_t)
        acc['gt']      += disparity_errors(out['disp'], gt_disp, mask_gt)
    model.train()

    res             = {'loss': loss_sum / max(n, 1)}
    for k, (s, b1, b3, cnt) in acc.items():
        cnt         = max(cnt, 1)
        res.update({f'{k}_epe': s / cnt, f'{k}_bad1': 100.0 * b1 / cnt, f'{k}_bad3': 100.0 * b3 / cnt})
    return res


# ── training ─────────────────────────────────────────────────────────────────

def seed_worker(worker_id):
    "different random crops per DataLoader worker"
    np.random.seed(torch.initial_seed() % 2 ** 32)


def finetune_mobile_net(npz_files, out_path=OUT_PATH, epochs=EPOCHS, lr=LR):
    "distill the teacher disparities in npz_files into MobileNetVolumeFusion. returns the best checkpoint path"
    n_total         = len(npz_files)
    if n_total < 2:
        raise RuntimeError(f"Need at least 2 cached frames for a train/test split, got {n_total}.")
    rng             = np.random.default_rng(SPLIT_SEED)
    order           = rng.permutation(n_total)
    n_train         = min(max(1, int(round(TRAIN_RATIO * n_total))), n_total - 1)
    train_files     = [npz_files[i] for i in order[:n_train]]
    test_files      = [npz_files[i] for i in order[n_train:]]
    logging.info(f"Split seed={SPLIT_SEED}: train={len(train_files)} frames, test={len(test_files)} frames")

    loader_kw       = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
                           worker_init_fn=seed_worker)
    train_loader    = DataLoader(DistillDataset(train_files, train=True), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_kw)
    test_loader     = DataLoader(DistillDataset(test_files, train=False), batch_size=1, shuffle=False, **loader_kw)

    model           = MobileNetVolumeFusion(D=MAX_DISP).cuda().train()
    model.use_checkpoint = True
    logging.info(f"Student parameters : {count_parameters(model):,}")

    optimizer       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler       = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler          = torch.amp.GradScaler('cuda')
    best_epe        = float('inf')

    for epoch in range(epochs):
        epoch_loss, parts, n_batches = 0.0, {}, 0
        for cost, img, t_disp, gt_disp, mask_t, mask_gt in train_loader:
            cost, img   = cost.cuda(non_blocking=True).float(), img.cuda(non_blocking=True)
            t_disp, gt_disp = t_disp.cuda(non_blocking=True), gt_disp.cuda(non_blocking=True)
            mask_t, mask_gt = mask_t.cuda(non_blocking=True), mask_gt.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                out     = model(cost, img)
            loss, comp  = distillation_loss(out, t_disp, gt_disp, mask_t, mask_gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item(); n_batches += 1
            for k, v in comp.items():
                parts[k] = parts.get(k, 0.0) + v
        scheduler.step()

        n_batches   = max(n_batches, 1)
        parts_str   = ' '.join(f'{k}={v / n_batches:.3f}' for k, v in parts.items())
        ev          = evaluate(model, test_loader)
        logging.info(
            f"Epoch {epoch+1:3d}/{epochs}  train_loss={epoch_loss / n_batches:.4f} ({parts_str})  test_loss={ev['loss']:.4f}  "
            f"vs teacher: epe={ev['teacher_epe']:.2f} bad1={ev['teacher_bad1']:.1f}% bad3={ev['teacher_bad3']:.1f}%  "
            f"vs GT: epe={ev['gt_epe']:.2f} bad3={ev['gt_bad3']:.1f}%")

        if ev['teacher_epe'] < best_epe:
            best_epe = ev['teacher_epe']
            save_checkpoint(out_path, model, extra={'epoch': epoch + 1, 'eval': ev, 'teacher': TEACHER_PATH})
            logging.info(f"  -> saved best model (teacher epe={best_epe:.3f}) to {out_path}")

    logging.info(f"Training complete. Best test EPE vs teacher: {best_epe:.3f}")
    return out_path


def main():
    import Utils as U
    U.set_logging_format()
    U.set_seed(0)
    npz_files       = build_teacher_cache()
    finetune_mobile_net(npz_files)


if __name__ == '__main__':
    main()

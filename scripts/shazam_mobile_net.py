"""
MobileNet-style fusion network for the Shazam multiscale feature cost volumes.

ShazamDepthEstimator.multiscale_feature_volumes produces K = levels x features guided-filtered
cost volumes (H, W, K, D). This network merges them, guided by the left image, into a
single disparity probability volume P (D over H, W) plus a per pixel confidence map.

  context branch (2D) : left image -> MobileNetV2 inverted residual blocks -> ctx
  cost branch    (3D) : K costs -> 1x1x1 conv + ctx broadcast over D -> 3D inverted residual blocks
                        (depthwise 3x3x3) -> logit per (d, y, x), plus a learned prior -mean(cost)
  confidence head     : [max P, entropy, ctx] -> 2D inverted residual -> sigmoid

Trained by distillation from Fast-FoundationStereo - see finetune_mobile_net.py.

Usage:
    from shazam_mobile_net import MobileNetVolumeFusion, predict_tiled, load_checkpoint
"""

import os
import sys
import types
import logging
import importlib.util

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ── shazam_depth_estimator import shim ────────────────────────────────────────
# shazam_depth_estimator.py imports helper modules from external checkouts
# (C:\Work\Projects\Utils\src, C:\Work\Projects\DepthRS\src). Stub the ones that
# can not be found so the estimator imports on machines without those packages.
SHAZAM_EXTERNAL_PATHS = [r'C:\Work\Projects\Utils\src', r'C:\Work\Projects\DepthRS\src']

def install_shazam_stubs() -> None:
    for p in SHAZAM_EXTERNAL_PATHS:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)

    def _stub(name: str, **attrs):
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            return
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    class _NullRectSelector:
        def __init__(self, *a, **kw): pass
        def draw(self, *a, **kw): pass

    class _NullRealSense:
        def __init__(self, *a, **kw): pass

    def _noop(*a, **kw): pass

    _stub("logger", log=logging.getLogger("shazam"))
    _stub("common", RectSelector=_NullRectSelector)
    _stub("depth_data_source", DataSource=type("DataSource", (), {}))
    _stub("opencv_realsense_camera", RealSense=_NullRealSense, draw_str=_noop)


# ── network ───────────────────────────────────────────────────────────────────

DEFAULT_CFG = dict(K=12, D=128, ch2d=16, ch3d=8, n_blocks2d=3, n_blocks3d=3, expand=2, cost_clip=8.0, argmax_radius=4)


def local_soft_argmax(prob, radius=4):
    "expected disparity in a +-radius window around the peak - far modes do not pull the estimate : (B,D,H,W) -> (B,1,H,W)"
    D               = prob.shape[1]
    d_index         = torch.arange(D, device=prob.device, dtype=prob.dtype).view(1, D, 1, 1)
    d_peak          = torch.argmax(prob, dim=1, keepdim=True)
    window          = (torch.abs(d_index - d_peak) <= radius).to(prob.dtype)
    p_win           = prob * window
    return torch.sum(p_win * d_index, dim=1, keepdim=True) / (torch.sum(p_win, dim=1, keepdim=True) + 1e-9)


class InvertedResidual2d(nn.Module):
    "MobileNetV2 block : 1x1 expand -> 3x3 depthwise -> 1x1 project, residual"
    def __init__(self, ch, expand=2, dilation=1):
        super().__init__()
        hid             = ch * expand
        self.block      = nn.Sequential(
            nn.Conv2d(ch, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True),
            nn.Conv2d(hid, hid, 3, padding=dilation, dilation=dilation, groups=hid, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True),
            nn.Conv2d(hid, ch, 1, bias=False), nn.BatchNorm2d(ch))

    def forward(self, x):
        return x + self.block(x)


class InvertedResidual3d(nn.Module):
    "3D MobileNetV2 block over (D, H, W) : pointwise expand -> 3x3x3 depthwise -> pointwise project, residual"
    def __init__(self, ch, expand=2):
        super().__init__()
        hid             = ch * expand
        self.block      = nn.Sequential(
            nn.Conv3d(ch, hid, 1, bias=False), nn.BatchNorm3d(hid), nn.ReLU6(inplace=True),
            nn.Conv3d(hid, hid, 3, padding=1, groups=hid, bias=False), nn.BatchNorm3d(hid), nn.ReLU6(inplace=True),
            nn.Conv3d(hid, ch, 1, bias=False), nn.BatchNorm3d(ch))

    def forward(self, x):
        return x + self.block(x)


class MobileNetVolumeFusion(nn.Module):
    """
    inputs  : cost (B,K,D,H,W) normalized costs, img (B,1,H,W) in [0,255]
    outputs : dict(logits (B,D,H,W), prob (B,D,H,W), disp (B,1,H,W) soft-argmax, conf (B,1,H,W))
    """
    def __init__(self, **cfg):
        super().__init__()
        self.cfg            = dict(DEFAULT_CFG, **cfg)
        c                   = self.cfg
        K, ch2, ch3, ex     = c['K'], c['ch2d'], c['ch3d'], c['expand']
        self.use_checkpoint = False     # gradient checkpointing of the 3D blocks (training memory)

        # context branch
        self.ctx_stem       = nn.Sequential(nn.Conv2d(1, ch2, 3, padding=1, bias=False), nn.BatchNorm2d(ch2), nn.ReLU6(inplace=True))
        self.ctx_blocks     = nn.Sequential(*[InvertedResidual2d(ch2, ex, dilation=2 ** i) for i in range(c['n_blocks2d'])])
        self.ctx_to_vol     = nn.Conv2d(ch2, ch3, 1)

        # cost branch
        self.cost_stem      = nn.Sequential(nn.Conv3d(K, ch3, 1, bias=False), nn.BatchNorm3d(ch3), nn.ReLU6(inplace=True))
        self.cost_blocks    = nn.ModuleList([InvertedResidual3d(ch3, ex) for _ in range(c['n_blocks3d'])])
        self.cost_head      = nn.Conv3d(ch3, 1, 1)
        self.prior_scale    = nn.Parameter(torch.tensor(4.0))   # logits start as softmin of the mean cost

        # confidence head
        self.conf_stem      = nn.Sequential(nn.Conv2d(2 + ch2, ch2, 1, bias=False), nn.BatchNorm2d(ch2), nn.ReLU6(inplace=True))
        self.conf_block     = InvertedResidual2d(ch2, ex)
        self.conf_head      = nn.Conv2d(ch2, 1, 1)

        nn.init.zeros_(self.cost_head.weight)
        nn.init.zeros_(self.cost_head.bias)

    def forward(self, cost, img):
        D                   = cost.shape[2]
        img_n               = img / 127.5 - 1.0
        ctx                 = self.ctx_blocks(self.ctx_stem(img_n))                                   # B,ch2,H,W

        x                   = self.cost_stem(cost) + self.ctx_to_vol(ctx).unsqueeze(2)                # B,ch3,D,H,W
        for blk in self.cost_blocks:
            x               = checkpoint(blk, x, use_reentrant=False) if (self.use_checkpoint and self.training) else blk(x)
        logits              = self.cost_head(x).squeeze(1) - self.prior_scale * cost.mean(dim=1)      # B,D,H,W
        logits              = logits.float()

        prob                = torch.softmax(logits, dim=1)
        disp                = local_soft_argmax(prob, self.cfg['argmax_radius'])

        # confidence sees P detached - its loss must not reshape the probability volume
        p_det               = prob.detach()
        p_max               = p_det.max(dim=1, keepdim=True)[0]
        entropy             = -(p_det * torch.log(p_det + 1e-9)).sum(dim=1, keepdim=True) / np.log(D)
        f                   = self.conf_stem(torch.cat([p_max.to(ctx.dtype), entropy.to(ctx.dtype), ctx], dim=1))
        conf                = torch.sigmoid(self.conf_head(self.conf_block(f)).float())

        return {'logits': logits, 'prob': prob, 'disp': disp, 'conf': conf}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ── numpy <-> torch helpers ───────────────────────────────────────────────────

def volumes_to_tensor(cost_HWKD, img_HW, cost_clip=DEFAULT_CFG['cost_clip']):
    "numpy (H,W,K,D), (H,W) -> torch (K,D,H,W), (1,H,W) float32"
    cost            = np.clip(cost_HWKD.astype(np.float32), 0, cost_clip) / (cost_clip / 2)
    cost_t          = torch.from_numpy(np.ascontiguousarray(cost.transpose(2, 3, 0, 1)))
    img_t           = torch.from_numpy(np.ascontiguousarray(img_HW.astype(np.float32)))[None]
    return cost_t, img_t


@torch.no_grad()
def predict_tiled(model, cost_HWKD, img_HW, tile_rows=96, overlap=16, device='cpu'):
    """
    Full frame inference in row bands so only a band of the (H,W,K,D) volume is on the device.
    returns prob (H,W,D) float32, conf (H,W) float32
    """
    model.eval()
    H, W, K, D      = cost_HWKD.shape
    prob_out        = np.zeros((H, W, D), np.float32)
    conf_out        = np.zeros((H, W), np.float32)
    use_amp         = str(device).startswith('cuda')
    cost_clip       = model.cfg.get('cost_clip', DEFAULT_CFG['cost_clip'])

    for r0 in range(0, H, tile_rows):
        r1          = min(H, r0 + tile_rows)
        a0, a1      = max(0, r0 - overlap), min(H, r1 + overlap)
        cost_t, img_t = volumes_to_tensor(cost_HWKD[a0:a1], img_HW[a0:a1], cost_clip)
        cost_t, img_t = cost_t[None].to(device), img_t[None].to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            out     = model(cost_t, img_t)
        s0, s1      = r0 - a0, r1 - a0
        prob_out[r0:r1] = out['prob'][0, :, s0:s1].permute(1, 2, 0).float().cpu().numpy()
        conf_out[r0:r1] = out['conf'][0, 0, s0:s1].float().cpu().numpy()
    return prob_out, conf_out


# ── checkpoint I/O ────────────────────────────────────────────────────────────

def save_checkpoint(path, model, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'cfg': model.cfg, 'state_dict': model.state_dict(), 'extra': extra or {}}, path)


def load_checkpoint(path=None, device='cpu', **cfg):
    "path None -> untrained network with cfg (pipeline testing)"
    if path is None:
        logging.warning('MobileNetVolumeFusion : no weights given, using an untrained network')
        return MobileNetVolumeFusion(**cfg).to(device).eval()
    ckpt            = torch.load(path, map_location='cpu', weights_only=False)
    model           = MobileNetVolumeFusion(**ckpt['cfg'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    m               = MobileNetVolumeFusion()
    print(f'parameters : {count_parameters(m):,}')
    cost            = torch.rand(1, 12, 128, 64, 160)
    img             = torch.rand(1, 1, 64, 160) * 255
    out             = m.eval()(cost, img)
    print({k: tuple(v.shape) for k, v in out.items()})
    assert torch.allclose(out['prob'].sum(1), torch.ones(1, 64, 160), atol=1e-4)
    assert out['conf'].min() >= 0 and out['conf'].max() <= 1
    prob, conf      = predict_tiled(m, np.random.rand(100, 160, 12, 128).astype(np.float16), np.random.rand(100, 160) * 255, tile_rows=40)
    print(f'tiled : prob {prob.shape}, conf {conf.shape}')
    print('ok')

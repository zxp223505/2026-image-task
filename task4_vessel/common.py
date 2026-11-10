"""Shared utilities: data, augmentation, model, metrics."""
import os
import json
import math
import random
import time

import cv2
import numpy as np
import torch
from PIL import Image

# 项目根目录 / 数据目录均可通过环境变量覆盖
WORK = os.environ.get("VESSEL_WORK", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("VESSEL_DATA", r"C:\Users\zhaopangpang\Desktop\4\segmentation")
# 预训练权重缓存放在可写目录
os.environ.setdefault("TORCH_HOME", os.path.join(WORK, "cache"))

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision

# working resolution: 512 matches the evaluation grid; 640 gives the network more
# pixels per vessel (the native images are 565x584, so 512 actually downsamples them)
SIZE = int(os.environ.get("VESSEL_SIZE", "512"))
# mask binarisation threshold after resizing the label.
# 127 = original thin convention; 0 = official `image > 0` convention (thick GT).
GT_THR = int(os.environ.get("VESSEL_GT_THR", "127"))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

cv2.setNumThreads(2)


# ----------------------------------------------------------------------------- data
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_ids():
    return sorted(int(os.path.splitext(f)[0]) for f in os.listdir(os.path.join(DATA_ROOT, "train", "image")))


def test_ids():
    return sorted(int(os.path.splitext(f)[0]) for f in os.listdir(os.path.join(DATA_ROOT, "test", "image")))


def read_rgb(path, retries=3):
    """Read a JPEG; retry a few times because Windows occasionally fails a read
    while another process (antivirus, preview) is touching the file."""
    for k in range(retries):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        time.sleep(0.2)
    raise IOError(f"cannot read image: {path}")


def read_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def fov_mask(img_rgb, erode=2):
    """Circular field-of-view mask at 512x512."""
    g = img_rgb[:, :, 1]
    m = (g > 12).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == k).astype(np.uint8)
    # fill holes
    ff = m.copy()
    h, w = m.shape
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 1)
    m = (m | (1 - ff)).astype(np.uint8)
    if erode > 0:
        m = cv2.erode(m, np.ones((erode, erode), np.uint8))
    return m


def load_train_item(i, size=SIZE):
    img = read_rgb(os.path.join(DATA_ROOT, "train", "image", f"{i}.jpg"))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    # use PIL BILINEAR for the mask so that GT_THR=0 exactly reproduces the official
    # `segmentation_to_csv.py` convention (PIL resize, then any positive pixel = vessel)
    label = Image.open(os.path.join(DATA_ROOT, "train", "label", f"{i}.jpg")).convert("L")
    gt = np.array(label.resize((size, size), Image.BILINEAR))
    gt = (gt > GT_THR).astype(np.uint8)
    fov = fov_mask(img)
    return img, gt, fov


def load_extra_item(img_path, mask_path, size=SIZE):
    """One external (e.g. CHASE) image + binary vessel mask, resized to `size`."""
    img = read_rgb(img_path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    label = Image.open(mask_path).convert("L")
    gt = np.array(label.resize((size, size), Image.BILINEAR))
    gt = (gt > GT_THR).astype(np.uint8)
    fov = fov_mask(img)
    return img, gt, fov


def load_test_item(i, size=SIZE):
    img = read_rgb(os.path.join(DATA_ROOT, "test", "image", f"{i}.jpg"))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    fov = fov_mask(img)
    return img, fov


def clahe_channel(green, clip=2.0):
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return c.apply(green)


def build_input(img, clip=2.0):
    """uint8 RGB 512x512 -> normalized 4xHxW float tensor."""
    cl = clahe_channel(img[:, :, 1], clip)
    rgb = img.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    cl = cl.astype(np.float32) / 255.0
    cl = (cl - 0.5) / 0.25
    x = np.concatenate([rgb.transpose(2, 0, 1), cl[None]], 0).astype(np.float32)
    return torch.from_numpy(x)


# ----------------------------------------------------------------------------- augment
def _affine(img, gt, M, out_size):
    img = cv2.warpAffine(img, M, (out_size, out_size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    gt = cv2.warpAffine(gt, M, (out_size, out_size), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, gt


def photometric(img, rng):
    """img uint8 RGB HxWx3 -> uint8 RGB."""
    f = img.astype(np.float32)

    # brightness
    if rng.random() < 0.8:
        f *= rng.uniform(0.75, 1.25)
    # contrast around mean
    if rng.random() < 0.8:
        m = f.mean()
        f = (f - m) * rng.uniform(0.75, 1.3) + m
    # gamma
    if rng.random() < 0.7:
        g = rng.uniform(0.6, 1.6)
        f = 255.0 * np.clip(f / 255.0, 0, 1) ** g
    f = np.clip(f, 0, 255)

    # illumination gradient (flash / camera variation)
    if rng.random() < 0.4:
        h, w = f.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gx, gy = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
        field = 1.0 + gx * (xx / max(w - 1, 1) - 0.5) * 2.0 + gy * (yy / max(h - 1, 1) - 0.5) * 2.0
        f = f * field[..., None]

    # vignetting
    if rng.random() < 0.3:
        h, w = f.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        v = np.clip(1.0 - rng.uniform(0.10, 0.45) * r2, 0.0, 1.0)
        f = f * v[..., None]

    img = np.clip(f, 0, 255).astype(np.uint8)

    # hue / saturation jitter
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.7, 1.3), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # blur
    if rng.random() < 0.25:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    # noise
    if rng.random() < 0.3:
        noise = rng.normal(0, rng.uniform(2, 8), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # jpeg compression artefacts (test images are jpg)
    if rng.random() < 0.3:
        q = int(rng.randint(45, 96))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def elastic(img, gt, rng):
    h, w = gt.shape
    if rng.random() > 0.3:
        return img, gt
    alpha = rng.uniform(8, 20)
    sigma = rng.uniform(6, 10)
    dx = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mapx = x + dx
    mapy = y + dy
    img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    gt = cv2.remap(gt, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, gt


class VesselDataset(Dataset):
    """Random crops from the training images with heavy augmentation.

    `items` lets the caller pass extra (image, mask, fov) triplets, e.g. external
    datasets; by default the DRIVE training ids are loaded.
    """

    def __init__(self, ids, crop=384, samples_per_epoch=256, seed=0, items=None):
        self.items = items if items is not None else [load_train_item(i) for i in ids]
        self.crop = crop
        self.n = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        # precompute vessel coords for centre sampling
        self.vpts = []
        for img, gt, fov in self.items:
            ys, xs = np.nonzero(gt)
            if len(ys) > 0:
                idx = np.random.RandomState(0).choice(len(ys), min(len(ys), 20000), replace=False)
                self.vpts.append(np.stack([ys[idx], xs[idx]], 1))
            else:
                self.vpts.append(np.zeros((1, 2), np.int64))

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.RandomState((self.seed * 1000003 + self.epoch * 7919 + idx) % (2 ** 31))
        k = rng.randint(len(self.items))
        img, gt, fov = self.items[k]
        C, H, W = self.crop, img.shape[0], img.shape[1]

        # choose crop centre (bias to vessel pixels 65% of the time)
        if rng.random() < 0.65:
            p = self.vpts[k][rng.randint(len(self.vpts[k]))]
            cy, cx = int(p[0]), int(p[1])
            cy += int(rng.randint(-40, 41))
            cx += int(rng.randint(-40, 41))
        else:
            cy, cx = rng.randint(0, H), rng.randint(0, W)
        y0 = int(np.clip(cy - C // 2, 0, H - C))
        x0 = int(np.clip(cx - C // 2, 0, W - C))
        img = img[y0:y0 + C, x0:x0 + C]
        gt = gt[y0:y0 + C, x0:x0 + C]

        # flips
        if rng.random() < 0.5:
            img, gt = img[:, ::-1], gt[:, ::-1]
        if rng.random() < 0.5:
            img, gt = img[::-1], gt[::-1]

        # rotation + scale
        ang = rng.uniform(-180, 180)
        scale = rng.uniform(0.85, 1.2)
        M = cv2.getRotationMatrix2D((C / 2 - 0.5, C / 2 - 0.5), ang, scale)
        M[0, 2] += rng.uniform(-0.06, 0.06) * C
        M[1, 2] += rng.uniform(-0.06, 0.06) * C
        img, gt = _affine(img, gt, M, C)

        img, gt = elastic(img, gt, rng)
        img = photometric(np.ascontiguousarray(img), rng)
        gt = np.ascontiguousarray(gt).astype(np.float32)

        green = img[:, :, 1]
        clip = rng.uniform(1.0, 4.0)
        clahe = clahe_channel(green, clip)

        rgb = img.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        cl = clahe.astype(np.float32) / 255.0
        cl = (cl - 0.5) / 0.25
        x = np.concatenate([rgb.transpose(2, 0, 1), cl[None]], 0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(gt[None])


# ----------------------------------------------------------------------------- model
class ConvGN(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.gn = nn.GroupNorm(min(8, cout), cout)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)


class SEBlock(nn.Module):
    """Channel squeeze-excitation on decoder features."""

    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(1, ch // r)
        self.fc1 = nn.Conv2d(ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout, se=False):
        super().__init__()
        self.c1 = ConvGN(cin + cskip, cout)
        self.c2 = ConvGN(cout, cout)
        self.se = SEBlock(cout) if se else nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], 1)
        return self.se(self.c2(self.c1(x)))


class ResNetUNet(nn.Module):
    """ResNet encoder + UNet decoder.

    se:   squeeze-excitation in the decoder (off by default so that old checkpoints load)
    aux:  deep-supervision heads at stride 4 / 16 (only used during training)
    """

    ENC_CH = {
        "resnet18": (64, 128, 256, 512),
        "resnet34": (64, 128, 256, 512),
        "resnet50": (256, 512, 1024, 2048),
        "resnet101": (256, 512, 1024, 2048),
    }

    def __init__(self, arch="resnet34", in_ch=4, pretrained=True, dec=(256, 128, 64, 32),
                 se=False, aux=False):
        super().__init__()
        if arch not in self.ENC_CH:
            raise ValueError(f"unsupported arch {arch}, choose from {list(self.ENC_CH)}")
        c1, c2, c3, c4 = self.ENC_CH[arch]
        weights = "IMAGENET1K_V1" if pretrained else None
        m = getattr(torchvision.models, arch)(weights=weights)
        w = m.conv1.weight.data
        if in_ch != 3:
            extra = w.mean(1, keepdim=True).repeat(1, in_ch - 3, 1, 1)
            w = torch.cat([w, extra], 1)
        self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.conv1.weight.data = w
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4

        d1, d2, d3, d4 = dec
        self.up4 = UpBlock(c4, c3, d1, se)
        self.up3 = UpBlock(d1, c2, d2, se)
        self.up2 = UpBlock(d2, c1, d3, se)
        self.up1 = UpBlock(d3, 64, d4, se)
        self.head = nn.Sequential(
            ConvGN(d4 + 3, 32, 3),
            nn.Conv2d(32, 1, 1),
        )
        self.aux_enabled = aux
        if aux:
            self.aux_head1 = nn.Conv2d(d1, 1, 1)   # stride 16
            self.aux_head2 = nn.Conv2d(d3, 1, 1)   # stride 4

    def forward(self, x, return_aux=False):
        rgb = x[:, :3]
        e0 = self.relu(self.bn1(self.conv1(x)))       # 64 x 256
        e1 = self.layer1(self.maxpool(e0))             # 64 x 128
        e2 = self.layer2(e1)                           # 128 x 64
        e3 = self.layer3(e2)                           # 256 x 32
        e4 = self.layer4(e3)                           # 512 x 16
        d1 = self.up4(e4, e3)
        d2 = self.up3(d1, e2)
        d3 = self.up2(d2, e1)
        d4 = self.up1(d3, e0)
        d = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.head(torch.cat([d, rgb], 1))
        if return_aux and self.aux_enabled:
            size = out.shape[-2:]
            a1 = F.interpolate(self.aux_head1(d1), size=size, mode="bilinear", align_corners=False)
            a2 = F.interpolate(self.aux_head2(d3), size=size, mode="bilinear", align_corners=False)
            return out, a1, a2
        return out

    def train(self, mode=True):
        super().train(mode)
        # keep encoder BN statistics frozen (small batches, tiny dataset)
        for mod in [self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self


# ----------------------------------------------------------------------------- loss / metric
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return 1 - (num / den).mean()


def edge_weighted_bce(logits, target, edge_w=1.0):
    """BCE with extra weight on the 1-px vessel boundary band.

    Thin vessels are almost all boundary; up-weighting edge pixels makes the loss
    focus on their outline instead of the large easy background.
    """
    edge = F.max_pool2d(target, 3, 1, 1) - (-F.max_pool2d(-target, 3, 1, 1))
    w = 1.0 + edge_w * edge
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).mean()


def dice_np(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    return 2.0 * inter / (pred.sum() + gt.sum() + 1e-6)


# ----------------------------------------------------------------------------- clDice
# Soft-skeleton topology loss (Shit et al., CVPR 2021). Keeps thin vessels connected.
def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iters=6):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(logits, target, iters=6, eps=1e-6):
    """1 - clDice(sigmoid(logits), target); per-sample, then averaged."""
    p = torch.sigmoid(logits)
    skel_p = soft_skel(p, iters)
    skel_t = soft_skel(target, iters)
    dims = (1, 2, 3)
    tprec = ((skel_p * target).sum(dims) + eps) / (skel_p.sum(dims) + eps)
    tsens = ((skel_t * p).sum(dims) + eps) / (skel_t.sum(dims) + eps)
    cl = 2.0 * tprec * tsens / (tprec + tsens + eps)
    return 1.0 - cl.mean()


@torch.no_grad()
def predict_full(model, x, tta=4, rots=1):
    """x: 1x4xHxW normalized. Returns probability map HxW (numpy).

    tta: number of flip variants (1/2/4); rots: number of 90-degree rotations (1..4).
    The input is zero-padded to a multiple of 32 so odd sizes (e.g. multi-scale TTA
    at 0.9x -> 461 px) do not break the decoder's skip connections.
    """
    model.eval()
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    logits = torch.zeros(1, 1, x.shape[-2], x.shape[-1], device=x.device)
    n = 0
    flips = [(False, False), (True, False), (False, True), (True, True)][:tta]
    for r in range(rots):
        xr = torch.rot90(x, r, dims=[2, 3]) if r else x
        for fh, fw in flips:
            xi = xr
            if fh:
                xi = torch.flip(xi, dims=[2])
            if fw:
                xi = torch.flip(xi, dims=[3])
            out = model(xi)
            if fh:
                out = torch.flip(out, dims=[2])
            if fw:
                out = torch.flip(out, dims=[3])
            if r:
                out = torch.rot90(out, -r, dims=[2, 3])
            logits += torch.sigmoid(out)
            n += 1
    p = (logits / n)[0, 0].float().cpu().numpy()
    return p[:h, :w] if (ph or pw) else p

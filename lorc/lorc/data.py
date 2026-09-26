#!/usr/bin/env python3
"""Manifests, labels and preprocessing for LoRC.

LABELS: TWO KEYS THAT LOOK ALIKE AND MEAN DIFFERENT THINGS
---------------------------------------------------------
The manifests in this project carry one of two label keys, and they are NOT the same scheme:

  `label`      3-class -- 0=real, 1=pad, 2=deepfake.  EVAL_SPACE/manifests/es_*.json,
               PAAS_v4_full/runs/manifests_devheld/spc3_train.json.
               Verified against get_label_all on every row of es_dev_eval_c99 and a 20k sample of
               spc3_train: 0 disagreements.

  `cls_label`  BINARY -- 0=real, 1=fake.  PAAS_v4_full/runs/manifests_devheld/images_train.json.
               Reading it as 3-class silently relabels every deepfake as `pad`: a 20k sample of
               that file has 5,015 rows where cls_label=1 and the true 3-class label is 2, and
               nothing about the resulting run looks wrong -- it trains, the loss falls, and the
               deepfake class is simply never learned.

So `cls_label` is REFUSED as a 3-class source (load_manifest raises), with the fix named in the
error. Both files list the same 1,257,404 images; spc3_train.json is the one with real labels.

WHOLE-IMAGE PREPROCESSING, AND WHY NOT PADDING
----------------------------------------------
The paper: "Images are randomly cropped to 224x224 for training and center-cropped for inference,
with padding applied for insufficient dimensions." That is right for its benchmarks -- whole
natural/generated photographs of roughly uniform size. It is wrong for this corpus, twice over.

FIRST, A FIXED CROP IS A SHORTCUT HERE. Image size is class-correlated (900-image sample of
es_dev_sel, p5/p50/p95):

    real      W 480/720/1189   H 640/989/1920
    pad       W 320/1080/1920  H 240/1080/1920
    deepfake  W 224/299/1618   H 224/299/1080

A 224 window covers ~4% of a median 1080x1080 pad frame but ~56% of a median 299x299 deepfake
frame, so crop scale alone is informative about the class, independent of any forgery artefact.
(Essentially no image is under 224 -- 0-1% per class -- so the padding branch is almost never
taken and is not the concern.)

SECOND, GEOMETRY IS CLASS-CORRELATED TOO, so there is NO fully neutral fixed-size square protocol
on this data. Aspect ratio W/H, 1200-image sample:

    real      0.56/0.75/0.78   square within 5%:  0%    mean border if padded: 33.1%
    pad       0.50/0.56/1.78   square within 5%:  6%    mean border if padded: 37.3%
    deepfake  0.73/1.00/1.74   square within 5%: 63%    mean border if padded: 11.7%

That rules the options as follows:

  "resize"      Resize((size, size)). THE DEFAULT. Uses every pixel, no cropping. The residual
                confound is that the AMOUNT of aspect distortion is class-correlated -- real and
                pad get stretched, deepfake (63% already square) barely does.
  pad-to-square NOT IMPLEMENTED, deliberately. It is the obvious way to avoid distortion, and it
                is worse than distortion HERE: a constant border contributes near-zero patch
                residuals straight into Eq 6's second moment R^T R, i.e. it biases the exact
                quantity L_SS operates on -- by 12% of the frame for deepfake against 33-37% for
                real and pad. Trading a distortion confound for a direct corruption of the
                method's own statistic is not a trade worth making.
  "paper"       RandomCrop(224)/CenterCrop(224). Kept as the faithfulness ablation
                (configs/lorc_crop224.json), carrying the scale confound above.
  "resize_crop" Resize short side to resize_short, then crop. Scale-normalised but discards pixels.

Whichever is used is written into the checkpoint and into runs/lorc.json, so a score column can
never be compared across protocols by accident.

RESOLUTION is 384, not the paper's 224, and that is measured rather than preferred:
EVAL_SPACE/runs/dino3_resolution_probe.json probed DINOv3 on a 24k slice of es_dev_sel and found
384 clearly better than 224 (bin-AUC 0.99242 vs 0.98705, fake-recall@real98 0.9519 vs 0.9331) at
2.9x the cost, with 512 no better than 384. Under whole-image resize the argument is stronger
still: squashing a 1080x1080 frame to 224x224 discards most of the high-frequency texture whose
residual SPECTRUM this method reads.
"""
import io
import json
import os
import random

import torch
import torch.distributed as dist
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

from .get_label import get_label_all, REAL, PAD, DEEPFAKE, MAKEUP, UNKNOWN
from . import paths as P
from .model import MEAN, STD

# 3-class throughout: real / pad / deepfake. MAKEUP folds into pad and UNKNOWN rows are dropped,
# the same scheme the GSD, SeLop, PE-SPC and DINOv3-SPC members use, so a LoRC score column is
# comparable to theirs without a relabelling step.
CLASS_NAMES = ("real", "pad", "deepfake")


def label_from_path(path):
    """-> 3-class label, or None to drop the row. MAKEUP folds into pad, as everywhere else here."""
    lab = get_label_all(path)
    if lab == UNKNOWN:
        return None
    if lab == REAL:
        return 0
    if lab == DEEPFAKE:
        return 2
    return 1                                       # PAD or MAKEUP


def load_manifest(path, label_source="manifest", verify_sample=2000, log=print):
    """-> list[(image_path, label3)].

    label_source:
      "manifest"  use the `label` key; cross-check a sample against the path rule and refuse on
                  any disagreement (they agree on every file checked, so a disagreement means the
                  manifest is not what it claims).
      "path"      ignore the key and derive from the path.
    """
    recs = json.load(open(path))
    if not recs:
        raise SystemExit(f"[data] {path} is empty")
    keys = set(recs[0])

    if label_source == "manifest":
        if "label" not in keys:
            if "cls_label" in keys:
                raise SystemExit(
                    f"[data] REFUSING {path}: it carries `cls_label`, which is BINARY "
                    f"(0=real, 1=fake), not the 3-class `label` this model trains on. Reading it "
                    f"as 3-class relabels every deepfake as `pad`. Use the 3-class manifest of the "
                    f"same images -- PAAS_v4_full/runs/manifests_devheld/spc3_train.json -- or "
                    f"pass --label-source path to derive labels from the image paths instead.")
            raise SystemExit(f"[data] {path} has no `label` key (keys: {sorted(keys)})")
        out = [(r["image"], int(r["label"])) for r in recs]
        bad, checked = [], 0
        step = max(1, len(out) // max(verify_sample, 1))
        for p, lab in out[::step]:
            ref = label_from_path(p)
            checked += 1
            if ref is not None and ref != lab:
                bad.append((p, lab, ref))
        if bad:
            raise SystemExit(
                f"[data] REFUSING {path}: {len(bad)} of {checked} sampled rows disagree with the "
                f"path rule, e.g. {bad[0][0]} -> manifest {bad[0][1]}, path {bad[0][2]}. One of "
                f"the two is wrong and training on either would be guesswork.")
        log(f"[data] {os.path.basename(path)}: {len(out):,} rows, `label` verified against the "
            f"path rule on {checked:,} sampled rows (0 disagreements)")
        return out

    if label_source != "path":
        raise ValueError(f"label_source must be 'manifest' or 'path', got {label_source!r}")
    out, dropped = [], 0
    for r in recs:
        lab = label_from_path(r["image"])
        if lab is None:
            dropped += 1
            continue
        out.append((r["image"], lab))
    log(f"[data] {os.path.basename(path)}: {len(out):,} rows from paths, {dropped:,} UNKNOWN dropped")
    return out


def build_transforms(train, crop_mode="resize", size=384, resize_short=256, hflip=True,
                     jpeg_qf=0, degrade=None):
    """The three protocols documented at the top of this file.

    jpeg_qf mirrors the paper's "PNG images are recompressed using JPEG (quality factor 96) for
    consistency". It is OFF by default here: this corpus is already JPEG, so applying it would be
    a SECOND compression generation rather than the levelling step the paper intends. Left
    available because "does a re-compression pass change anything" is a fair robustness question.
    """
    norm = transforms.Normalize(MEAN, STD)
    bic = transforms.InterpolationMode.BICUBIC
    pre = [JpegRecompress(jpeg_qf)] if jpeg_qf else []
    # Degradation goes BEFORE the geometry: it models what happened to the image on its way to
    # disk, and resizing a degraded image is not the same as degrading a resized one.
    if train and degrade and float(degrade.get("p", 0)) > 0:
        pre = pre + [DegradeUniform(
            p=degrade.get("p", 0.5), jpeg_range=degrade.get("jpeg_range", (65, 95)),
            scale_range=degrade.get("scale_range", (0.6, 1.0)),
            blur_max=degrade.get("blur_max", 0.8),
            kernels=degrade.get("kernels", DegradeUniform.KERNELS),
            seed=degrade.get("seed"))]

    if crop_mode == "resize":
        # Whole image, no cropping, no padding -- see the module docstring for why padding is
        # refused rather than offered.
        geom = [transforms.Resize((size, size), interpolation=bic)]
    elif crop_mode == "paper":
        if train:
            geom = [transforms.RandomCrop(size, pad_if_needed=True, padding_mode="reflect")]
        else:
            # CenterCrop zero-pads an undersized image, which scores a face against an invented
            # black border. ResizeIfSmaller upscales only in that case (0-1% of rows here), so the
            # common path is an untouched centre crop.
            geom = [ResizeIfSmaller(size), transforms.CenterCrop(size)]
    elif crop_mode == "resize_crop":
        geom = [transforms.Resize(resize_short, interpolation=bic)]
        geom += ([transforms.RandomCrop(size, pad_if_needed=True, padding_mode="reflect")]
                 if train else [transforms.CenterCrop(size)])
    else:
        raise ValueError(f"crop_mode must be resize|paper|resize_crop, got {crop_mode!r}. "
                         f"Pad-to-square is deliberately absent -- see this module's docstring.")

    if train and hflip:
        geom.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(pre + geom + [transforms.ToTensor(), norm])



class DegradeUniform:
    """Acquisition-degradation augmentation applied IDENTICALLY to every class.

    WHY UNIFORM IS THE WHOLE POINT. This corpus's classes were not captured the same way -- PAD is
    a camera capture of a screen or print, deepfakes come from a dozen generators at a dozen output
    resolutions -- so JPEG generation, resampling kernel and sharpness are all partly
    class-correlated BEFORE any augmentation. A detector can score well by reading acquisition
    instead of forgery, and that shortcut does not survive a change of source. Applying the same
    degradation distribution to all three classes cannot remove an existing correlation, but it
    stops the model relying on the part of it that a re-encode would destroy.

    WHY IT IS AN ARM AND NOT A DEFAULT. LoRC reads the SPECTRUM of the patch residual, and JPEG,
    downsampling and blur all attack high-frequency content -- which is where a generator's
    fingerprint lives. So for this method specifically, degradation augmentation is as likely to
    erase the signal as to harden it. `configs/lorc_degrade.json` exists to measure which.

    Order is deliberate: resample (kernel choice) -> blur -> JPEG, i.e. the order a real pipeline
    applies them, so the JPEG generation is the last thing to touch the pixels.
    """

    KERNELS = ("bilinear", "bicubic", "nearest", "lanczos")

    def __init__(self, p=0.5, jpeg_range=(65, 95), scale_range=(0.6, 1.0), blur_max=0.8,
                 kernels=KERNELS, seed=None):
        self.p = float(p)
        self.jpeg_range = tuple(jpeg_range) if jpeg_range else None
        self.scale_range = tuple(scale_range) if scale_range else None
        self.blur_max = float(blur_max)
        self.kernels = tuple(kernels)
        self.seed = seed
        self._rng = None
        self._pid = None

    @property
    def rng(self):
        """Per-worker, per-rank stream -- created lazily, AFTER the fork.

        A random.Random built in __init__ is copied into every DataLoader worker and every DDP rank
        with IDENTICAL state, so with 8 workers x 4 ranks all 32 processes would draw the same
        JPEG qualities, the same kernels and the same blur radii in the same order. The images
        differ, so it is not literally one augmentation -- but the augmentation DISTRIBUTION is
        sampled 32 times at the same points instead of independently, which is most of the
        diversity this arm exists to provide. Re-seeding per process fixes it while staying
        reproducible: the stream is a pure function of (seed, rank, worker id).
        """
        pid = os.getpid()
        if self._rng is None or self._pid != pid:
            wi = torch.utils.data.get_worker_info()
            worker = wi.id if wi is not None else 0
            rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
            base = 0 if self.seed is None else int(self.seed)
            self._rng = random.Random((base * 1_000_003 + rank * 1009 + worker) & 0xFFFFFFFF)
            self._pid = pid
        return self._rng

    def __call__(self, img):
        if self.rng.random() >= self.p:
            return img
        if self.scale_range:
            lo, hi = self.scale_range
            f = self.rng.uniform(lo, hi)
            if f < 0.999:
                w, h = img.size
                k = getattr(Image.Resampling, self.rng.choice(self.kernels).upper())
                small = img.resize((max(int(w * f), 8), max(int(h * f), 8)), k)
                # Back to the original size, with an INDEPENDENTLY chosen kernel: a fixed
                # down/up pair is itself a signature the model could learn.
                k2 = getattr(Image.Resampling, self.rng.choice(self.kernels).upper())
                img = small.resize((w, h), k2)
        if self.blur_max > 0:
            r = self.rng.uniform(0.0, self.blur_max)
            if r > 0.05:
                img = img.filter(ImageFilter.GaussianBlur(radius=r))
        if self.jpeg_range:
            q = self.rng.randint(*self.jpeg_range)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img

class ResizeIfSmaller:
    """Upscale only when a side is under `size`, so the eval crop never sees invented border."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if min(w, h) >= self.size:
            return img
        s = self.size / min(w, h)
        return img.resize((max(self.size, round(w * s)), max(self.size, round(h * s))),
                          Image.BICUBIC)


class JpegRecompress:
    def __init__(self, quality):
        self.quality = int(quality)

    def __call__(self, img):
        import io
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class LoRCDataset(Dataset):
    """Returns (tensor, label, row_index, ok).

    `ok` rides through the DataLoader instead of a parent-process counter because __getitem__ runs
    in forked workers, where a counter increment is invisible to the parent. An unreadable image
    yields a mid-grey frame with ok=0 and the caller refuses on any ok==0 -- a black-frame
    substitution corrupted a GSD anchor earlier in this project while every log line still looked
    healthy, so silent substitution is not available here.

    `row_index` is the index into the ORIGINAL manifest, which is what makes a score array
    alignable to the manifest by row order (the contract dinospc_to_shard.py verifies by length).
    """

    def __init__(self, samples, transform, size=224, rows=None):
        self.samples = samples
        self.transform = transform
        self.size = size
        self.rows = rows if rows is not None else list(range(len(samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            # LORC_IMAGE_ROOT is applied HERE, at open time, and nowhere else. The manifest path
            # stays the row's identity -- scores align to the manifest, so rewriting it in the
            # sample list would silently change what a score file claims to be about. A remap
            # function that nothing called was worse than none: it read as working.
            img = Image.open(P.remap(path)).convert("RGB")
            ok = 1
        except Exception:                                             # noqa: BLE001
            img = Image.new("RGB", (self.size, self.size), (127, 127, 127))
            ok = 0
        return self.transform(img), int(label), int(self.rows[i]), ok


def class_counts(samples):
    h = [0, 0, 0]
    for _, lab in samples:
        h[lab] += 1
    return h


def class_weights(samples, device=None):
    """Inverse-frequency weights over (real, pad, deepfake), normalised to mean 1.

    Indexed exactly as LoRCModel.classification_loss expects. OFF by default in the configs: the
    balanced sampler already equalises the classes in the stream, and doing both double-counts the
    correction.
    """
    h = class_counts(samples)
    n = sum(h)
    w = [(n / (len(h) * c) if c else 0.0) for c in h]
    m = sum(x for x in w if x) / max(sum(1 for x in w if x), 1)
    w = [x / m for x in w]
    return torch.tensor(w, dtype=torch.float32, device=device)

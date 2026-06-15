"""
change_common.py
================

Shared helpers for the open-end shock-tube high-speed-camera *change-detection*
analysis.

Data model
----------
A recording is a directory of single-frame TIFFs (e.g. ``run/run000001.tif`` ...),
named so that *alphabetical sort == temporal order*. Each frame is a 2-D
grayscale image of the open end of a shock tube. For most of the recording the
scene is quiescent (a static background); at some point gas/flow exits the tube
and *something changes* in the image -- a plume, a density gradient, ejected
particles, a faint shimmer. We do **not** look for a specific shape: we look for
**any** departure from the quiescent background.

The core idea
-------------
Two things make a pixel value at frame *t* different from the static scene:

1. A genuine, *spatially coherent* change (the flow) -- many neighbouring pixels
   all move together, even if only slightly.
2. *Independent* per-pixel sensor noise -- each pixel wanders on its own.

Both can have the same single-pixel amplitude, so one pixel of noise is
indistinguishable from one pixel of faint flow. What separates them is **spatial
coherence**, the spatial analogue of the temporal coherence used by the
companion ``shock_tube_image_analysis`` pipeline.

We exploit it like this:

    raw frame  ──(subtract per-pixel background)──>  difference image
               ──(divide by per-pixel noise sigma)──>  SNR map  (~N(0,1) if quiet)
               ──(box-average over a small window)───>  pooled SNR map
                       │  coherent flow  -> survives the average  (stays large)
                       │  random noise   -> averages towards 0    (cancels out)
                       ▼
            count pixels with |pooled SNR| > pix_k     = per-frame "activity"

The background and the per-pixel noise sigma are both estimated from the
*quiescent* frames (robustly, via median / MAD over a temporal sample), so even
a long-lasting flow does not pollute them (see ``build_background``).

This module is imported by ``detect_activity.py`` (Script 1) and
``analyze_activity.py`` (Script 2). It depends only on numpy + Pillow.
The Photron ``.cihx`` parsing is shared, unchanged, with the sister project.
"""

from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Frame discovery & I/O
# --------------------------------------------------------------------------- #

def list_frames(input_dir, pattern="*.tif"):
    """Return frame file paths sorted in temporal (== filename) order."""
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No frames matching {pattern!r} found in {input_dir!r}")
    return files


def frame_number(path):
    """Extract the integer frame number from a filename (last run of digits)."""
    m = re.findall(r"(\d+)", os.path.basename(path))
    return int(m[-1]) if m else -1


def load_frame(path, row_lo=None, row_hi=None, col_lo=None, col_hi=None):
    """
    Load one frame as a 2-D float32 image, optionally cropped to a sub-window.

    Unlike the sister project (which collapses each strip to a 1-D column
    profile), change detection keeps the full 2-D image: the flow can appear
    anywhere, with any shape. A 1-D input is promoted to a single-row image.

    row_lo/row_hi/col_lo/col_hi optionally restrict to a region of interest.
    """
    a = np.asarray(Image.open(path), dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim == 3:                      # collapse an accidental RGB(A) to luma
        a = a[..., :3].mean(axis=2)
    if row_lo is not None or row_hi is not None:
        a = a[row_lo:row_hi, :]
    if col_lo is not None or col_hi is not None:
        a = a[:, col_lo:col_hi]
    return np.ascontiguousarray(a, dtype=np.float32)


def frame_shape(path, row_lo=None, row_hi=None, col_lo=None, col_hi=None):
    """Return the (height, width) of a frame after the same optional crop."""
    return load_frame(path, row_lo, row_hi, col_lo, col_hi).shape


# --------------------------------------------------------------------------- #
# Photron .cihx camera-metadata parsing (auto-fill fps / pixel size)
# --------------------------------------------------------------------------- #
# (Identical to the sister project's shock_common.py so a recording's .cihx is
#  read the same way by both pipelines.)

_UNIT_TO_UM = {"mm": 1000.0, "millimeter": 1000.0, "millimetre": 1000.0,
               "um": 1.0, "µm": 1.0, "micrometer": 1.0, "micrometre": 1.0,
               "cm": 1.0e4, "m": 1.0e6, "nm": 1.0e-3}


@dataclass
class CameraMeta:
    """Camera characteristics read from a Photron .cihx (.cih) sidecar file."""
    source: str = ""
    device: str | None = None
    date: str | None = None
    fps: float | None = None              # record rate, frames/s
    exposure_us: float | None = None      # shutter time, microseconds
    width: int | None = None
    height: int | None = None
    bit_depth: int | None = None
    total_frames: int | None = None
    pixel_size_um: float | None = None    # None when uncalibrated (see notes)
    pixel_size_raw: float | None = None   # sizeOfPixel as stored
    pixel_unit: str | None = None
    magnification: float | None = None
    scale_mode: int | None = None
    notes: list = field(default_factory=list)


def find_cihx(input_dir):
    """Return the path to a .cihx/.cih sidecar in input_dir, or None."""
    cands = sorted(glob.glob(os.path.join(input_dir, "*.cihx")) +
                   glob.glob(os.path.join(input_dir, "*.cih")))
    return cands[0] if cands else None


def _unit_factor_um(unit_text):
    """Map a unit string like 'Milimeters (mm)' to a micrometre factor."""
    if not unit_text:
        return None, None
    m = re.search(r"\(([^)]+)\)", unit_text)        # prefer the (mm) symbol
    sym = (m.group(1) if m else unit_text).strip().lower()
    sym = sym.replace("meters", "m").replace("metres", "m")
    if sym in _UNIT_TO_UM:
        return _UNIT_TO_UM[sym], sym
    for key, fac in _UNIT_TO_UM.items():            # fall back to substring
        if key in unit_text.lower():
            return fac, key
    return None, sym


def parse_cihx(path):
    """
    Parse a Photron .cihx (binary-wrapped XML) or .cih file into a CameraMeta.

    fps is taken as authoritative. Spatial pixel size is only filled in when the
    file is actually calibrated; the PFV default (scaleMode 0 / sizeOfPixel 1.0)
    is treated as *uncalibrated* and pixel_size_um is left None, with a note.
    """
    raw = open(path, "rb").read()
    s = raw.find(b"<cih")
    e = raw.find(b"</cih>")
    if s < 0 or e < 0:
        meta = CameraMeta(source=path)
        meta.notes.append("could not locate <cih> XML block in file")
        return meta
    xml = raw[s:e + 6].decode("utf-8", "replace")
    root = ET.fromstring(xml)

    def ftext(*paths):
        for p in paths:
            el = root.find(p)
            if el is not None and el.text and el.text.strip():
                return el.text.strip()
        return None

    def fnum(*paths, cast=float):
        v = ftext(*paths)
        try:
            return cast(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    meta = CameraMeta(source=path)
    meta.device = ftext("deviceInfo/deviceName", "basicInfo/cameraName")
    d, t = ftext("fileInfo/date"), ftext("fileInfo/time")
    meta.date = f"{d} {t}".strip() if (d or t) else None
    meta.fps = fnum("recordInfo/recordRate", "deviceInfo/recordRate")
    ns = fnum("recordInfo/shutterSpeedNsec")
    meta.exposure_us = ns / 1000.0 if ns else None
    meta.width = fnum("imageDataInfo/resolution/width",
                      "imageFileInfo/resolution/width", cast=int)
    meta.height = fnum("imageDataInfo/resolution/height",
                       "imageFileInfo/resolution/height", cast=int)
    meta.bit_depth = fnum("imageDataInfo/colorInfo/bit",
                          "imageDataInfo/effectiveBit/depth", cast=int)
    meta.total_frames = fnum("frameInfo/totalFrame", cast=int)

    meta.scale_mode = fnum("plugin/calibration/scaleMode", cast=int)
    meta.pixel_size_raw = fnum("plugin/calibration/sizeOfPixel")
    meta.magnification = fnum("plugin/calibration/magnification")
    meta.pixel_unit = ftext("plugin/warp/unit", "plugin/calibration/unit")

    if meta.fps is None:
        meta.notes.append("no recordRate found")
    sp = meta.pixel_size_raw
    mag = meta.magnification or 1.0
    uncalibrated = (meta.scale_mode in (None, 0)) or (sp is None) or \
                   (sp == 1.0 and mag == 1.0)
    if uncalibrated:
        meta.notes.append(
            "recording is NOT spatially calibrated "
            f"(scaleMode={meta.scale_mode}, sizeOfPixel={sp}); "
            "pass --px-size-um for speeds in m/s, "
            "or calibrate the scale in PFV.")
    else:
        fac, sym = _unit_factor_um(meta.pixel_unit)
        if fac is None:
            meta.notes.append(
                f"unknown calibration unit {meta.pixel_unit!r}; "
                "pass --px-size-um explicitly.")
        else:
            meta.pixel_size_um = sp * fac / mag
            meta.notes.append(
                f"spatial calibration: {sp} {sym}/px / mag {mag} "
                f"= {meta.pixel_size_um:.4g} um/px")
    return meta


def resolve_calibration(input_dir, fps_arg=None, px_size_um_arg=None):
    """
    Merge CLI overrides with any .cihx sidecar in input_dir.

    Returns (fps, pixel_size_um, meta, info_lines). CLI args always win; the
    .cihx fills whatever the user did not pass.
    """
    info = []
    meta = None
    path = find_cihx(input_dir)
    if path:
        meta = parse_cihx(path)
        info.append(f"camera metadata: {os.path.basename(path)}"
                    + (f"  [{meta.device}]" if meta.device else ""))
        if meta.fps:
            info.append(f"  recordRate = {meta.fps:g} fps"
                        + (f", exposure = {meta.exposure_us:g} us"
                           if meta.exposure_us else ""))
        if meta.width and meta.height:
            info.append(f"  resolution = {meta.width}x{meta.height}"
                        + (f", {meta.bit_depth}-bit" if meta.bit_depth else ""))
        for n in meta.notes:
            info.append(f"  note: {n}")
    else:
        info.append("no .cihx/.cih sidecar found; "
                    "using CLI arguments / defaults only")

    fps = fps_arg if fps_arg else (meta.fps if meta else None)
    px = px_size_um_arg if px_size_um_arg else (
        meta.pixel_size_um if meta else None)
    if fps_arg and meta and meta.fps and fps_arg != meta.fps:
        info.append(f"  (CLI --fps {fps_arg:g} overrides cihx {meta.fps:g})")
    if px_size_um_arg:
        info.append(f"  (using CLI --px-size-um {px_size_um_arg:g})")
    return fps, px, meta, info


# --------------------------------------------------------------------------- #
# 2-D smoothing (separable box average via an integral image, no scipy)
# --------------------------------------------------------------------------- #

def box2d_mean(img, k):
    """
    Mean over a k x k window (k forced odd), edge-padded so the output has the
    same shape as the input. Implemented with a summed-area table -> O(H*W)
    regardless of k.
    """
    k = int(k)
    if k <= 1:
        return img.astype(np.float32, copy=True)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ap = np.pad(img, pad, mode="edge").astype(np.float64)
    # Summed-area table with a leading zero row/column.
    sat = ap.cumsum(0).cumsum(1)
    sat = np.pad(sat, ((1, 0), (1, 0)), mode="constant")
    h, w = img.shape
    # Each output pixel = sum over its k x k window, by inclusion-exclusion on
    # the SAT corners (bottom-right - top-right - bottom-left + top-left).
    win = (sat[k:k + h, k:k + w] - sat[0:h, k:k + w]
           - sat[k:k + h, 0:w] + sat[0:h, 0:w])
    return (win / float(k * k)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Background + per-pixel noise model (the quiescent scene)
# --------------------------------------------------------------------------- #

def sample_indices(n, k):
    """Up to `k` indices spread uniformly across range(n) (always includes
    the first and last frame)."""
    if k >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def estimate_bg_noise(stack, noise_floor_frac=0.25):
    """
    Robust per-pixel background and noise from a stack of frames (T, H, W).

    background = per-pixel median       (transients do not pull a median)
    sigma      = per-pixel 1.4826 * MAD (robust standard deviation)

    Dead/constant pixels get sigma == 0; those are floored to
    ``noise_floor_frac * median(positive sigma)`` so the SNR map never divides
    by zero (and a genuine change at a very low-noise pixel still reads as a
    large, but finite, SNR).
    """
    stack = np.asarray(stack, dtype=np.float32)
    bg = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - bg), axis=0)
    sigma = (1.4826 * mad).astype(np.float32)
    pos = sigma[sigma > 0]
    floor = float(np.median(pos) * noise_floor_frac) if pos.size else 1.0
    floor = max(floor, 1e-6)
    sigma = np.maximum(sigma, floor)
    return bg.astype(np.float32), sigma


@dataclass
class BackgroundModel:
    bg: np.ndarray                 # (H, W) per-pixel background
    sigma: np.ndarray             # (H, W) per-pixel noise (robust std)
    sigma2: np.ndarray            # sigma**2, cached for SNR pooling
    inflation: float              # empirical pooled-SNR std on quiet frames (~1)
    thresh: float                 # effective pix threshold = pix_k * inflation
    n_used: int                   # quiescent sample frames used
    n_dropped: int                # active sample frames excluded by refinement
    info: list                    # human-readable log lines


def pooled_snr_scale(stack, bg, sigma2, p, edge_margin):
    """
    Robust spread (1.4826 * MAD) of the pooled-SNR values over a stack of
    quiescent frames, measured on interior pixels only.

    If the per-pixel noise model and the independence assumption were perfect
    this would be ~1. In practice residual spatial correlation or imperfect
    detrending make it differ; using it to *rescale* the threshold makes the
    per-pixel false-positive rate well-defined regardless of those details.
    """
    interior = _edge_mask((stack.shape[1], stack.shape[2]), edge_margin)
    vals = []
    for j in range(stack.shape[0]):
        snr = activity_snr(stack[j], bg, sigma2, p.smooth, p.detrend_band)
        vals.append(snr[interior])
    v = np.concatenate(vals)
    mad = np.median(np.abs(v - np.median(v)))
    scale = float(1.4826 * mad) if mad > 0 else float(v.std())
    return max(scale, 1e-6)


def build_background(files, p):
    """
    Build the quiescent-scene model from a uniform temporal *sample* of frames.

    Two passes make it robust to flow of *unknown duration* (the user may not
    know how long the ejection lasts):

      1. Median/MAD over the sample give a provisional background + noise, and an
         empirical pooled-SNR scale (``inflation``).
      2. Each sample frame's activity is measured against that provisional
         model; the clearly-active ones (robust-threshold outliers) are dropped
         and the model is rebuilt from the remaining quiescent frames.

    So even if the flow occupies a large fraction of the recording, as long as
    enough genuinely-quiescent frames exist anywhere in the sample, they define
    the background and the flow frames are excluded from it.

    Memory is O(sample) frames, independent of recording length.
    """
    n = len(files)
    idxs = sample_indices(n, p.bg_sample)
    stack = np.stack([load_frame(files[i], p.row_lo, p.row_hi,
                                 p.col_lo, p.col_hi) for i in idxs], axis=0)

    bg, sigma = estimate_bg_noise(stack, p.noise_floor_frac)
    sigma2 = sigma * sigma
    inflation = pooled_snr_scale(stack, bg, sigma2, p, p.edge_margin)
    n_dropped = 0

    if p.refine_bg and stack.shape[0] >= 8:
        # Pass 2: score each sample frame against the provisional model, then
        # rebuild from the quiescent ones so a long-lasting flow can't leak into
        # the background. A frame is "active" if its area is a robust outlier
        # AND above --min-area (so noise jitter alone never triggers a drop).
        thresh = p.pix_k * inflation
        areas = np.array([
            measure_activity(
                activity_snr(stack[j], bg, sigma2, p.smooth, p.detrend_band),
                thresh, p.edge_margin).area
            for j in range(stack.shape[0])], dtype=np.float64)
        thr, _, _ = robust_threshold(areas, p.refine_k)
        thr = max(thr, float(p.min_area))
        quiet = areas <= thr
        # Only refine if enough quiescent frames remain to define a stable
        # background -- otherwise keep the provisional model.
        if quiet.sum() >= max(8, stack.shape[0] // 4) and (~quiet).any():
            n_dropped = int((~quiet).sum())
            bg, sigma = estimate_bg_noise(stack[quiet], p.noise_floor_frac)
            sigma2 = sigma * sigma
            inflation = pooled_snr_scale(stack[quiet], bg, sigma2, p,
                                         p.edge_margin)

    thresh = p.pix_k * inflation
    info = [
        f"background model: {len(idxs)} sampled frames"
        f" ({stack.shape[1]}x{stack.shape[2]} px)",
        f"  dropped {n_dropped} active frame(s) from the background"
        if n_dropped else "  no active frames detected in the sample",
        f"  noise sigma: median={np.median(sigma):.3g}"
        f"  (5th-95th pct {np.percentile(sigma,5):.3g}"
        f"-{np.percentile(sigma,95):.3g})",
        f"  pooled-SNR inflation={inflation:.3g}; "
        f"effective pix threshold={thresh:.3g} sigma",
    ]
    return BackgroundModel(bg, sigma, sigma2, inflation, thresh,
                           n_used=int(stack.shape[0] - n_dropped),
                           n_dropped=n_dropped, info=info)


# --------------------------------------------------------------------------- #
# Per-frame activity (the SNR map and its reduction to a scalar)
# --------------------------------------------------------------------------- #

def activity_snr(frame, bg, sigma2, smooth, detrend_band=0):
    """
    Spatially-pooled, drift-corrected signal-to-noise map of one frame against
    the background.

        diff      = frame - bg
        diff_bp   = box-average(diff, smooth) - box-average(diff, detrend_band)
        noise_sm  = sqrt(box-average(sigma^2, smooth)) / smooth
        snr       = diff_bp / noise_sm

    The wide ``detrend_band`` term is a spatial **band-pass**: it removes
    slowly-varying changes -- global illumination flicker and optical drift,
    which are coherent across the whole frame and would otherwise swamp the
    pooled SNR -- while preserving a localised change (a plume is small compared
    to ``detrend_band``, so the wide average barely sees it). It is the 2-D
    analogue of the sister project's rolling-temporal-baseline detrending. The
    narrow term's noise dominates, so the per-pixel noise still propagates
    correctly through the division.

    Under quiescence each output pixel is ~N(0, scale); a spatially coherent
    change survives and reads as many-sigma. ``detrend_band <= 0`` disables it.
    """
    diff = frame - bg
    s = int(smooth) if smooth and smooth > 1 else 1
    if s % 2 == 0:
        s += 1
    narrow = box2d_mean(diff, s) if s > 1 else diff
    if detrend_band and detrend_band > s:
        narrow = narrow - box2d_mean(diff, detrend_band)
    if s > 1:
        noise_sm = np.sqrt(box2d_mean(sigma2, s)) / s
    else:
        noise_sm = np.sqrt(sigma2)
    return narrow / noise_sm


@dataclass
class ActivityMeasurement:
    """Per-frame summary of the pooled SNR map."""
    area: int            # # pixels with |snr| > pix_k (after edge masking)
    peak_snr: float      # max |snr|
    energy: float        # sum of |snr| over active pixels
    cx: float            # |snr|-weighted centroid column of active pixels
    cy: float            # |snr|-weighted centroid row of active pixels
    polarity: int        # +1 brighter than background, -1 darker, 0 if none
    mean_snr: float      # signed mean snr over active pixels


def _edge_mask(shape, edge_margin):
    """Boolean mask that is False within `edge_margin` of any border."""
    h, w = shape
    em_r = min(int(edge_margin), max(0, h // 4))
    em_c = min(int(edge_margin), max(0, w // 4))
    m = np.ones(shape, dtype=bool)
    if em_r > 0:
        m[:em_r, :] = False
        m[-em_r:, :] = False
    if em_c > 0:
        m[:, :em_c] = False
        m[:, -em_c:] = False
    return m


def measure_activity(snr, thresh, edge_margin):
    """Reduce a pooled SNR map to an ActivityMeasurement. ``thresh`` is the
    pixel cut in SNR units (typically pix_k * inflation)."""
    mask = np.abs(snr) > thresh
    if edge_margin > 0:
        mask &= _edge_mask(snr.shape, edge_margin)
    area = int(mask.sum())
    if area == 0:
        return ActivityMeasurement(0, float(np.abs(snr).max()), 0.0,
                                   0.0, 0.0, 0, 0.0)
    w = np.abs(snr[mask])
    ys, xs = np.nonzero(mask)
    wsum = float(w.sum())
    cx = float((xs * w).sum() / wsum)
    cy = float((ys * w).sum() / wsum)
    mean_snr = float(snr[mask].mean())
    return ActivityMeasurement(
        area=area,
        peak_snr=float(w.max()),
        energy=wsum,
        cx=cx, cy=cy,
        polarity=1 if mean_snr > 0 else -1,
        mean_snr=mean_snr,
    )


def largest_blob(mask):
    """
    Size and mask of the largest 4-connected component of a boolean image,
    via iterative flood fill (no scipy). Cheap because it is only ever called
    on the few frames of a single event, where the active set is small.
    """
    if not mask.any():
        return 0, np.zeros_like(mask)
    visited = np.zeros_like(mask)
    h, w = mask.shape
    best_size = 0
    best_mask = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                        and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) > best_size:
            best_size = len(comp)
            bm = np.zeros_like(mask)
            for y, x in comp:
                bm[y, x] = True
            best_mask = bm
    return best_size, best_mask


def robust_threshold(scores, k):
    """
    Robust threshold = median + k * (1.4826 * MAD). Resistant to the handful of
    real-event frames that may sit in the distribution.
    """
    scores = np.asarray(scores, dtype=np.float64)
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    sigma = 1.4826 * mad if mad > 0 else scores.std()
    return float(med + k * sigma), float(med), float(sigma)

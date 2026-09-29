#!/usr/bin/env python3
"""
detect_activity.py  --  Script 1: find frames where the scene changes.
======================================================================

Streams through every frame of a recording (in temporal order) and flags the
short stretches where *something happens* near the open end of the shock tube --
without assuming any particular shape for it.

How
---
1. Build a quiescent-scene model -- a per-pixel **background** and per-pixel
   **noise sigma** -- from a temporal sample of frames (robust median / MAD,
   refined to exclude any active frames; see ``change_common.build_background``).
   Unlit pixels (outside the window, ``--dark-frac``) and the border are not
   measured.
2. For every frame, form the spatially-pooled **SNR map**
   ``(frame - background) / noise`` and reduce it to an *activity* score: the
   number of pixels whose pooled SNR exceeds ``--pix-k``. Random sensor noise is
   spatially incoherent and averages away; a coherent change -- even a faint one
   -- survives and lights up a contiguous region.
3. A frame is *active* when its activity (active-pixel area) reaches
   ``--min-area``. Group consecutive active frames into **events** (bridging
   gaps up to ``--max-gap``, requiring at least ``--min-len`` frames). The
   per-frame and per-event requirements together demand both spatial coherence
   (a real region, not stray noise pixels) and temporal persistence (it lasts
   more than one frame) -- the change-detection analogue of the sister
   project's spatio-temporal coherence gate.
4. Flag the frames that are not a picture of the flow: the camera changed its
   timing / exposure (read from a Phantom ``.chd``: the trigger frame), or the
   whole frame's brightness jumps (flash, sensor recovering from saturation).
5. Build the **x-t diagram** (height-averaged change along the tube, stacked
   over time) and search it for straight slanted tracks -- moving fronts such
   as a shock -- summing the signal along each track so fronts too faint for
   any single frame are still found (see ``xt_diagram``). Also estimates the
   faintest front the search would have reported.

It scales to long recordings: the background uses only a bounded sample, and the
streaming pass holds one frame at a time (plus the fixed-size background maps).

Outputs, in ``<run>_analysis/`` next to the frame folder ``<run>``
(``--out-dir`` to change):

  * ``<run>_events.csv``    one row per detected event, with its flagged-frame
                            counts and overlapping shock candidates;
  * ``<run>_shocks.csv``    one row per shock-search candidate (header only if
                            none);
  * ``<run>_timeline.png``  x-t diagram + activity for the whole recording;
  * ``<run>_activity.npz``  per-frame arrays and the x-t diagram, read by the
                            analysis overview;
  * ``<run>_moments.csv``   the strongest moments outside the events (below
                            the event thresholds).

Runs of flagged frames outside any event (the trigger frame, flashes) become
short events of their own, so they are always analysed and shown.

Feed the CSV to ``analyze_activity.py`` (Script 2) to characterise and label
each event, or run both steps at once with ``full_detect_analysis.py``.

If the input folder holds no frames itself, each of its subfolders that does is
processed in turn (batch mode), skipping those whose ``<run>_events.csv``
exists unless ``--force`` is given.

Example
-------
    python3 detect_activity.py run                  # -> run_analysis/run_events.csv
    python3 detect_activity.py run --pix-k 5 --min-area 20 --no-plot-timeline
    python3 detect_activity.py campaign --rotate 90 # every shot in campaign/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

import change_common as cc
import xt_diagram as xtd


# --------------------------------------------------------------------------- #
# Streaming per-frame activity
# --------------------------------------------------------------------------- #

def measure_all(files, model, p):
    """
    Stream every frame and return per-frame arrays:
    area, peak_snr, energy, centroid x/y, polarity, signed mean SNR, mean
    brightness of the measured pixels, and the x-t diagram rows.
    """
    n = len(files)
    area = np.zeros(n, dtype=np.int64)
    peak = np.zeros(n, dtype=np.float32)
    energy = np.zeros(n, dtype=np.float32)
    cx = np.zeros(n, dtype=np.float32)
    cy = np.zeros(n, dtype=np.float32)
    polar = np.zeros(n, dtype=np.int8)
    mean_snr = np.zeros(n, dtype=np.float32)
    fnum = np.zeros(n, dtype=np.int64)
    level = np.zeros(n, dtype=np.float32)
    fhash = np.zeros(n, dtype=np.uint64)
    xt = np.zeros((n, model.bg.shape[1] // p.xt_bin), dtype=np.float32)

    for i, path in enumerate(files):
        frame = cc.load_frame(path, p.row_lo, p.row_hi, p.col_lo, p.col_hi,
                              p.rotate)
        snr = cc.activity_snr(frame, model.bg, model.sigma2, p.smooth,
                              p.detrend_band)
        m = cc.measure_activity(snr, model.thresh, model.valid)
        level[i] = float(frame[model.valid].mean())
        # Fingerprint of the frame, to find the same recording exported twice.
        fhash[i] = int.from_bytes(hashlib.blake2b(
            frame.tobytes(), digest_size=8).digest(), "little")
        xt[i] = xtd.xt_row(snr, model.valid, p.xt_bin)
        area[i] = m.area
        peak[i] = m.peak_snr
        energy[i] = m.energy
        cx[i] = m.cx
        cy[i] = m.cy
        polar[i] = m.polarity
        mean_snr[i] = m.mean_snr
        fnum[i] = cc.frame_number(path)
        if (i + 1) % 20000 == 0:
            print(f"[detect]  processed {i + 1}/{n}", file=sys.stderr)

    return dict(area=area, peak=peak, energy=energy, cx=cx, cy=cy,
                polar=polar, mean_snr=mean_snr, fnum=fnum, level=level,
                xt=xt, frame_hash=fhash)


# --------------------------------------------------------------------------- #
# Event grouping (temporal persistence of an active region)
# --------------------------------------------------------------------------- #

def group_events(a, p):
    """
    Group frames whose active-pixel area reaches --min-area into events,
    bridging gaps up to --max-gap and requiring at least --min-len frames
    (and at most --max-len, if set). Returns a list of event dicts.
    """
    area, fnum = a["area"], a["fnum"]
    flagged = np.where(area >= p.min_area)[0]      # indices of "active" frames
    events = []
    if flagged.size == 0:
        return events

    # Chain flagged frames into clusters, allowing up to --max-gap inactive
    # frames between two active ones. Adjacent active frames differ by 1, so a
    # gap of g inactive frames shows up as an index jump of g+1 -> max-gap + 1.
    clusters = []
    cur = [flagged[0]]
    for idx in flagged[1:]:
        if idx - cur[-1] <= p.max_gap + 1:
            cur.append(idx)
        else:
            clusters.append(cur)
            cur = [idx]
    clusters.append(cur)

    for cl in clusters:
        cl = np.array(cl)
        if cl.size < p.min_len:                    # too short -> reject (noise)
            continue
        span = int(cl[-1] - cl[0] + 1)
        if p.max_len and span > p.max_len:
            continue

        events.append(make_event(a, cl))
    return events


def make_event(a, cl):
    """Event dict for the frame indices ``cl`` (first to last, gaps bridged)."""
    area, fnum = a["area"], a["fnum"]
    cl = np.asarray(cl)
    span = int(cl[-1] - cl[0] + 1)
    seg = slice(int(cl[0]), int(cl[-1]) + 1)            # include bridged gaps
    areas = area[seg]
    peak_local = int(cl[0] + int(np.argmax(areas)))
    # Event polarity = sign-vote over its active frames; ties fall back to the
    # onset frame's polarity.
    pol = int(np.sign(np.sum(a["polar"][cl]))) or int(a["polar"][cl[0]])
    return {
        "start_index": int(cl[0]),
        "end_index": int(cl[-1]),
        "peak_index": peak_local,
        "start_frame": int(fnum[cl[0]]),
        "end_frame": int(fnum[cl[-1]]),
        "peak_frame": int(fnum[peak_local]),
        "n_active": int(cl.size),
        "span": span,
        "peak_area_px": int(area[peak_local]),
        "mean_area_px": round(float(areas.mean()), 1),
        "peak_snr": round(float(a["peak"][seg].max()), 3),
        "peak_energy": round(float(a["energy"][seg].max()), 1),
        "polarity": "brighter" if pol > 0 else "darker",
        "onset_cx": round(float(a["cx"][cl[0]]), 2),
        "onset_cy": round(float(a["cy"][cl[0]]), 2),
        "peak_cx": round(float(a["cx"][peak_local]), 2),
        "peak_cy": round(float(a["cy"][peak_local]), 2),
    }


def add_flagged_events(events, a, flagged):
    """
    Turn each run of flagged frames (camera / brightness jump) that no event
    covers into a short event of its own, whatever its length or area, so the
    trigger frame and flashes are always analysed and shown. Returns the events
    in time order.
    """
    runs, cur = [], []
    for i in flagged:
        if cur and i - cur[-1] > 2:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    out = list(events)
    for run in runs:
        if any(ev["start_index"] <= run[-1] and ev["end_index"] >= run[0]
               for ev in events):
            continue
        ev = make_event(a, np.arange(run[0], run[-1] + 1))
        ev["n_active"] = int((a["area"][run[0]:run[-1] + 1] > 0).sum())
        out.append(ev)
    return sorted(out, key=lambda ev: ev["start_index"])


def strongest_moments(a, events, n, flagged, sep=10):
    """
    The ``n`` most active frames outside every event (and at least ``sep``
    frames apart): what the detector came closest to reporting. Ranked by
    active-pixel area, then by peak pooled SNR.
    """
    if n <= 0:
        return []
    T = len(a["area"])
    free = np.ones(T, bool)
    for ev in events:
        free[max(0, ev["start_index"] - 2):ev["end_index"] + 3] = False
    free[list(flagged)] = False
    order = np.lexsort((-a["peak"], -a["area"]))
    out = []
    for i in order:
        if len(out) >= n:
            break
        if not free[i]:
            continue
        out.append(int(i))
        free[max(0, i - sep):i + sep + 1] = False
    return [{"rank": k, "index": i, "frame": int(a["fnum"][i]),
             "area_px": int(a["area"][i]),
             "peak_snr": round(float(a["peak"][i]), 3),
             "cx": round(float(a["cx"][i]), 2), "cy": round(float(a["cy"][i]), 2),
             "polarity": ("brighter" if a["polar"][i] > 0 else
                          "darker" if a["polar"][i] < 0 else "")}
            for k, i in enumerate(out)]


MOMENT_COLS = ["rank", "frame", "area_px", "peak_snr", "cx", "cy", "polarity",
               "index"]


# --------------------------------------------------------------------------- #
# Flagged frames and the shock search
# --------------------------------------------------------------------------- #

def flag_frames(a, meta, n):
    """
    Frames that cannot be trusted as a picture of the flow: the camera changed
    its timing / exposure (read from a Phantom .chd, typically the trigger
    frame), or the whole frame's brightness jumps (flash, light glitch, sensor
    recovering from saturation). Returns (camera, flash) index lists.
    """
    camera = [i for i in (meta.irregular_frames if meta else []) if i < n]
    flash = cc.level_jumps(a["level"])
    return camera, flash


def count_in(ev, idxs, pad=1):
    """How many of ``idxs`` fall within the event (+/- ``pad`` frames)."""
    lo, hi = ev["start_index"] - pad, ev["end_index"] + pad
    return sum(lo <= i <= hi for i in idxs)


def shock_search(a, model, p, flagged):
    """
    Look for straight slanted tracks in the x-t diagram (see xt_diagram).
    Returns (candidates, best_score, sensitivity).
    """
    z, col_scale = xtd.normalise(a["xt"], flagged)
    nb = z.shape[1]
    vmax = p.shock_vmax if p.shock_vmax else nb * p.xt_bin / 4.0
    score, speed, klen, nspeeds = xtd.search(z, p.shock_vmin / p.xt_bin,
                                             vmax / p.xt_bin)
    cands = xtd.candidates(score, speed, klen, z, p.shock_k, p.xt_bin)
    sens = xtd.sensitivity(model, p, col_scale, p.xt_bin, p.shock_k)
    fnum = a["fnum"]
    px_um = getattr(p, "px_size_um", None)
    for cid, c in enumerate(cands):
        t1 = min(len(fnum) - 1, c["t0"] + c["n_frames"] - 1)
        c.update(candidate_id=cid, start_index=c["t0"], end_index=t1,
                 start_frame=int(fnum[c["t0"]]), end_frame=int(fnum[t1]),
                 x_end_px=c["x0_px"] + c["v_px_per_frame"] * (t1 - c["t0"]))
        if p.fps:
            c["v_px_per_s"] = c["v_px_per_frame"] * p.fps
            if px_um:
                c["v_m_per_s"] = c["v_px_per_s"] * px_um * 1e-6
    print(f"[detect] shock search: {nspeeds} speeds x 2 directions, "
          f"{p.shock_vmin:g}-{vmax:g} px/frame; best track score "
          f"{float(score.max()):.2f} (report threshold {p.shock_k:g})",
          file=sys.stderr)
    if sens:
        k, (amp, pct) = 8, sens.get(8, next(iter(sens.values())))
        print(f"[detect]   sensitivity: a full-height front of >= {amp:.2f} "
              f"pixel-noise sigma (~{pct:.2f}% of the brightness) seen over "
              f"{k} frames would be reported", file=sys.stderr)
    return cands, float(score.max()), sens


def frame_times(meta, n, fps):
    """Time of each file from the trigger (us): the camera's own timestamps
    when a Phantom header gives them, else index / fps from the trigger."""
    if meta is not None and meta.frame_times_us is not None \
            and len(meta.frame_times_us) >= n:
        return np.asarray(meta.frame_times_us[:n], dtype=np.float64)
    trig = meta.trigger_index if meta is not None and \
        meta.trigger_index is not None else None
    if fps and trig is not None:
        return (np.arange(n) - trig) / fps * 1e6
    return np.full(n, np.nan)


def meta_dict(meta, n_files, fps):
    """Camera metadata for the dashboard (JSON-safe)."""
    d = {"n_files": n_files, "fps": fps}
    if meta is None:
        return d
    d.update(source=os.path.basename(meta.source), device=meta.device,
             serial=meta.serial, date=meta.date,
             exposure_us=meta.exposure_us, width=meta.width,
             height=meta.height, bit_depth=meta.bit_depth,
             header_images=meta.total_frames,
             trigger_index=meta.trigger_index,
             irregular_frames=list(meta.irregular_frames),
             pixel_size_um=meta.pixel_size_um, notes=list(meta.notes))
    if meta.exposures_us is not None:
        d["exposure_measured_us"] = float(np.median(meta.exposures_us))
    return d


PARAM_KEYS = ["rotate", "row_lo", "row_hi", "col_lo", "col_hi", "edge_margin",
              "dark_frac", "bg_sample", "smooth", "detrend_band", "pix_k",
              "min_area", "min_len", "max_gap", "shock_k", "shock_vmin",
              "shock_vmax", "xt_bin", "moments"]


def params_dict(p):
    return {k: getattr(p, k, None) for k in PARAM_KEYS}


def model_stats(model):
    """Brightness and noise of the measured field, for comparing shots."""
    v = model.valid
    bg = float(np.median(model.bg[v])) if v.any() else float("nan")
    sig = float(np.median(model.sigma[v])) if v.any() else float("nan")
    return {"brightness": bg, "noise": sig,
            "noise_pct": 100.0 * sig / bg if bg > 0 else None,
            "measured_frac": float(v.mean()), "inflation": model.inflation,
            "pix_threshold": model.thresh}


SHOCK_COLS = ["candidate_id", "score", "start_frame", "end_frame", "n_frames",
              "x0_px", "x_end_px", "v_px_per_frame", "v_px_per_s", "v_m_per_s",
              "polarity", "start_index", "end_index"]


def write_shocks(path, cands):
    with open(path, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=SHOCK_COLS)
        wcsv.writeheader()
        for c in cands:
            wcsv.writerow({k: (round(c[k], 3) if isinstance(c.get(k), float)
                               else c.get(k, "")) for k in SHOCK_COLS})


# --------------------------------------------------------------------------- #
# Timeline plot for the whole recording: x-t diagram + activity
# --------------------------------------------------------------------------- #

def plot_timeline(a, events, p, out_png, flagged=(), cands=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fnum = a["fnum"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 7), sharey=True,
        gridspec_kw=dict(width_ratios=[3, 1]))
    z, _ = xtd.normalise(a["xt"], flagged)
    xtd.draw_xt(ax1, z, fnum, p.xt_bin, cands, flagged)
    ax1.set_title("x-t diagram (height-averaged change; yellow guides = shock "
                  "candidate, grey = flagged frame)", fontsize=9)

    ax2.plot(a["area"], fnum, lw=0.8, color="black")
    ax2.axvline(p.min_area, color="orange", lw=1.0, ls="--",
                label=f"min-area = {p.min_area}")
    for ev in events:
        ax2.axhspan(ev["start_frame"] - 0.5, ev["end_frame"] + 0.5,
                    color="red", alpha=0.18, lw=0)
    ax2.set_xscale("symlog", linthresh=max(1, p.min_area))
    ax2.set_xlim(0, max(10 * p.min_area, 1.5 * float(a["area"].max())))
    ax2.set_xlabel("active-pixel area")
    ax2.set_title("activity (red = events)", fontsize=9)
    ax2.legend(loc="lower right", fontsize=7)
    fig.suptitle(os.path.basename(os.path.normpath(p.input_dir)))
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser(description="Detect frames where the open-tube scene "
                               "changes."):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("input_dir",
                    help="Directory of single-frame TIFFs, or a directory of "
                         "such directories (batch mode: each one is a shot).")
    ap.add_argument("--pattern", default="*.tif", help="Glob for frames.")
    ap.add_argument("--out-dir", default=None,
                    help="Folder receiving every output of the shot "
                         "(default: <input_dir>_analysis; single shot only).")
    ap.add_argument("--force", action="store_true",
                    help="In batch mode, reprocess shots whose outputs "
                         "already exist.")

    g = ap.add_argument_group("region of interest")
    g.add_argument("--row-lo", type=int, default=None,
                   help="First row of the analysed region (default: all).")
    g.add_argument("--row-hi", type=int, default=None,
                   help="Last row (exclusive) of the analysed region.")
    g.add_argument("--col-lo", type=int, default=None,
                   help="First column of the analysed region (default: all).")
    g.add_argument("--col-hi", type=int, default=None,
                   help="Last column (exclusive) of the analysed region.")
    g.add_argument("--edge-margin", type=int, default=6,
                   help="Pixels ignored at each border (unreliable optics edge).")
    g.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270),
                   help="Rotate frames clockwise by this many degrees after "
                        "cropping (e.g. 90 to lay a vertical tube "
                        "horizontally). Crop bounds stay in original-image "
                        "coordinates.")

    g = ap.add_argument_group("background / noise model")
    g.add_argument("--bg-sample", type=int, default=150,
                   help="Frames sampled uniformly to build the background.")
    g.add_argument("--noise-floor-frac", type=float, default=0.25,
                   help="Floor on per-pixel sigma, as a fraction of the median "
                        "sigma (guards low-noise pixels).")
    g.add_argument("--no-refine-bg", dest="refine_bg", action="store_false",
                   help="Skip dropping active frames from the background.")
    g.add_argument("--refine-k", type=float, default=4.0,
                   help="Robustness of the active-frame cut during refinement.")
    g.add_argument("--dark-frac", type=float, default=0.1,
                   help="Ignore unlit pixels: background below this fraction "
                        "of the median background (plus an --edge-margin band "
                        "around them). 0 keeps every pixel.")

    g = ap.add_argument_group("change detection")
    g.add_argument("--smooth", type=int, default=5,
                   help="Side of the spatial box average pooling the SNR map. "
                        "Larger = more sensitive to faint, broad changes.")
    g.add_argument("--detrend-band", type=int, default=101,
                   help="Wide spatial box (px) subtracted from the difference "
                        "image to remove illumination flicker / drift "
                        "(band-pass). Make it >> the expected change size; "
                        "0 disables it.")
    g.add_argument("--pix-k", type=float, default=6.0,
                   help="Per-pixel pooled-SNR threshold (sigma). A pixel counts "
                        "as active above this. Lower (e.g. 5) for more "
                        "sensitivity to faint changes; raise to reject noise.")
    g.add_argument("--min-area", type=int, default=None,
                   help="Active-pixel count for a frame to be 'active'. "
                        "Default: auto from frame size.")
    g.add_argument("--min-area-frac", type=float, default=5e-5,
                   help="Auto --min-area = max(8, frac * frame area).")

    g = ap.add_argument_group("event grouping")
    g.add_argument("--min-len", type=int, default=3,
                   help="Min consecutive active frames per event "
                        "(temporal persistence).")
    g.add_argument("--max-len", type=int, default=None,
                   help="Max frame span of an event (default: unlimited).")
    g.add_argument("--max-gap", type=int, default=2,
                   help="Max inactive frames bridged within an event.")

    g = ap.add_argument_group("shock search in the x-t diagram")
    g.add_argument("--no-shock-search", dest="shock_search",
                   action="store_false",
                   help="Skip the search for moving fronts.")
    g.add_argument("--shock-k", type=float, default=7.0,
                   help="Report tracks whose score (sigma) reaches this. Pure "
                        "noise stays below ~6.3 on the test recordings.")
    g.add_argument("--shock-vmin", type=float, default=4.0,
                   help="Slowest front searched, px/frame. Slower tracks pick "
                        "up stationary changes.")
    g.add_argument("--shock-vmax", type=float, default=None,
                   help="Fastest front searched, px/frame (default: a quarter "
                        "of the frame width, i.e. seen in >= 4 frames).")
    g.add_argument("--xt-bin", type=int, default=4,
                   help="Columns merged in the x-t diagram (px).")
    g.add_argument("--moments", type=int, default=8,
                   help="List this many strongest moments outside the events "
                        "(below the event thresholds) in NAME_moments.csv and "
                        "on the overview. 0 disables.")

    g = ap.add_argument_group("metadata / diagnostics")
    g.add_argument("--fps", type=float, default=None,
                   help="Camera frame rate; auto-read from a .cihx or .chd "
                        "if present.")
    g.add_argument("--save-activity", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Write NAME_activity.npz: per-frame arrays and the x-t "
                        "diagram, used by the analysis overview.")
    g.add_argument("--plot-timeline", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Write NAME_timeline.png: the x-t diagram and "
                        "activity for the whole movie.")
    return ap


def resolve_outputs(p):
    """Fill in this shot's output paths (see cc.ShotPaths)."""
    p.paths = cc.shot_paths(p.input_dir, p.out_dir)
    p.out_dir = p.paths.dir
    p.out = p.paths.events


def main(argv=None):
    ap = build_parser()
    p = ap.parse_args(argv)
    shots, batch = cc.find_shots(p.input_dir, p.pattern)
    cc.check_batch_args(ap, p, batch, ["--out-dir"])

    def process(shot):
        q = argparse.Namespace(**vars(p))
        q.input_dir = shot
        resolve_outputs(q)
        if batch and not q.force and os.path.exists(q.out):
            return f"{q.out} exists; use --force to redo"
        run(q)

    return cc.run_shots("detect", shots, batch, process)


def run(p):
    """Detect events in one shot folder; ``p`` has its outputs resolved."""
    t0 = time.time()
    os.makedirs(p.out_dir, exist_ok=True)

    files = cc.list_frames(p.input_dir, p.pattern)
    n = len(files)
    fps, _, meta, info = cc.resolve_calibration(p.input_dir, p.fps, None, n)
    for line in info:
        print(f"[detect] {line}", file=sys.stderr)
    p.fps = fps
    print(f"[detect] {n} frames in {p.input_dir!r}", file=sys.stderr)

    # Resolve an auto --min-area from the (cropped) frame size if not given.
    h, w = cc.frame_shape(files[0], p.row_lo, p.row_hi, p.col_lo, p.col_hi,
                          p.rotate)
    if p.min_area is None:
        p.min_area = max(8, int(round(p.min_area_frac * h * w)))
    print(f"[detect] frame {h}x{w} px; min-area = {p.min_area} px, "
          f"pix-k = {p.pix_k}, smooth = {p.smooth}", file=sys.stderr)

    model = cc.build_background(files, p)
    for line in model.info:
        print(f"[detect] {line}", file=sys.stderr)

    a = measure_all(files, model, p)

    # Diagnostic only: how the per-frame activity is distributed. Gating uses
    # the fixed --min-area, not this robust threshold.
    _thr, med, sigma = cc.robust_threshold(a["area"], 1.0)
    print(f"[detect] activity area: median={med:.1f} robust-sigma={sigma:.1f} "
          f"max={int(a['area'].max())}", file=sys.stderr)

    camera, flash = flag_frames(a, meta, n)
    flagged = sorted(set(camera) | set(flash))
    if flagged:
        print(f"[detect] flagged frames (index): camera timing/exposure "
              f"{camera[:10]}, brightness jump {flash[:10]}"
              + (" ..." if len(flash) > 10 else ""), file=sys.stderr)

    cands, best_score, sens = [], 0.0, {}
    if p.shock_search:
        cands, best_score, sens = shock_search(a, model, p, flagged)
        for c in cands:
            spd = (f", {c['v_m_per_s']:.0f} m/s" if "v_m_per_s" in c else "")
            print(f"[detect]   candidate {c['candidate_id']}: score "
                  f"{c['score']:.1f}, frames {c['start_frame']}-"
                  f"{c['end_frame']}, x {c['x0_px']:.0f}->{c['x_end_px']:.0f}"
                  f" px, {c['v_px_per_frame']:+.1f} px/frame{spd}",
                  file=sys.stderr)

    events = add_flagged_events(group_events(a, p), a, flagged)
    for ev in events:
        ev["camera_frames"] = count_in(ev, camera)
        ev["flash_frames"] = count_in(ev, flash)
        ev["shock_candidates"] = " ".join(
            str(c["candidate_id"]) for c in cands
            if c["start_index"] <= ev["end_index"] + 1
            and c["end_index"] >= ev["start_index"] - 1)
    print(f"[detect] {len(events)} event(s) detected "
          f"in {time.time()-t0:.1f}s", file=sys.stderr)
    moments = strongest_moments(a, events, p.moments, flagged)

    cols = ["event_id", "start_frame", "end_frame", "peak_frame",
            "n_active", "span", "peak_area_px", "mean_area_px",
            "peak_snr", "peak_energy", "polarity",
            "onset_cx", "onset_cy", "peak_cx", "peak_cy",
            "camera_frames", "flash_frames", "shock_candidates",
            "start_index", "end_index", "peak_index"]
    with open(p.out, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=cols)
        wcsv.writeheader()
        for eid, ev in enumerate(events):
            ev = dict(ev, event_id=eid)
            wcsv.writerow({c: ev.get(c, "") for c in cols})
    print(f"[detect] wrote {p.out}", file=sys.stderr)

    if p.moments > 0:
        with open(p.paths.moments, "w", newline="") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=MOMENT_COLS)
            wcsv.writeheader()
            wcsv.writerows(moments)
        print(f"[detect] wrote {p.paths.moments} (strongest moments below "
              "the event thresholds)", file=sys.stderr)
    if p.shock_search:
        write_shocks(p.paths.shocks, cands)
        print(f"[detect] wrote {p.paths.shocks}", file=sys.stderr)
    if p.save_activity:
        np.savez_compressed(
            p.paths.activity, min_area=p.min_area, pix_k=p.pix_k,
            smooth=p.smooth, xt_bin=p.xt_bin, frame_shape=(h, w),
            camera_frames=np.array(camera, dtype=np.int64),
            flash_frames=np.array(flash, dtype=np.int64),
            shock_search=p.shock_search, shock_k=p.shock_k,
            best_score=best_score,
            sensitivity=np.array([(k, v[0], v[1]) for k, v in sens.items()]),
            trigger_index=(meta.trigger_index if meta and
                           meta.trigger_index is not None else -1),
            time_us=frame_times(meta, n, p.fps),
            meta_json=json.dumps(meta_dict(meta, n, p.fps)),
            params_json=json.dumps(params_dict(p)),
            stats_json=json.dumps(model_stats(model)),
            **a)
        print(f"[detect] wrote {p.paths.activity}", file=sys.stderr)
    if p.plot_timeline:
        plot_timeline(a, events, p, p.paths.timeline, flagged, cands)
        print(f"[detect] wrote {p.paths.timeline}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

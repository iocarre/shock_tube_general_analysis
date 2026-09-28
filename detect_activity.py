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

It scales to long recordings: the background uses only a bounded sample, and the
streaming pass holds one frame at a time (plus the fixed-size background maps).

Output: a CSV (default ``activity_events.csv``) with one row per detected event.
Feed it to ``analyze_activity.py`` (Script 2) to characterise each one.

Example
-------
    python3 detect_activity.py run --out activity_events.csv
    python3 detect_activity.py run --pix-k 5 --min-area 20 --save-activity act.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

import change_common as cc


# --------------------------------------------------------------------------- #
# Streaming per-frame activity
# --------------------------------------------------------------------------- #

def measure_all(files, model, p):
    """
    Stream every frame and return per-frame arrays:
    area, peak_snr, energy, centroid x/y, polarity, signed mean SNR.
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

    for i, path in enumerate(files):
        frame = cc.load_frame(path, p.row_lo, p.row_hi, p.col_lo, p.col_hi,
                              p.rotate)
        snr = cc.activity_snr(frame, model.bg, model.sigma2, p.smooth,
                              p.detrend_band)
        m = cc.measure_activity(snr, model.thresh, p.edge_margin)
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
                polar=polar, mean_snr=mean_snr, fnum=fnum)


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

        seg = slice(int(cl[0]), int(cl[-1]) + 1)        # include bridged gaps
        areas = area[seg]
        peak_local = int(cl[0] + int(np.argmax(areas)))
        # Event polarity = sign-vote over its active frames; ties fall back to
        # the onset frame's polarity.
        pol = int(np.sign(np.sum(a["polar"][cl]))) or int(a["polar"][cl[0]])

        events.append({
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
        })
    return events


# --------------------------------------------------------------------------- #
# Optional activity-timeline plot for the whole recording
# --------------------------------------------------------------------------- #

def plot_timeline(a, events, p, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(a["fnum"], a["area"], lw=0.8, color="black")
    ax.axhline(p.min_area, color="orange", lw=1.0, ls="--",
               label=f"min-area = {p.min_area}")
    for ev in events:
        ax.axvspan(ev["start_frame"], ev["end_frame"],
                   color="red", alpha=0.18)
    ax.set_xlabel("frame number")
    ax.set_ylabel("active-pixel area")
    ax.set_title("activity timeline (red bands = detected events)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    ap = argparse.ArgumentParser(
        description="Detect frames where the open-tube scene changes.")
    ap.add_argument("input_dir", help="Directory of single-frame TIFFs.")
    ap.add_argument("--pattern", default="*.tif", help="Glob for frames.")
    ap.add_argument("--out", default="activity_events.csv",
                    help="Output CSV path.")

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

    g = ap.add_argument_group("metadata / diagnostics")
    g.add_argument("--fps", type=float, default=None,
                   help="Camera frame rate; auto-read from a .cihx if present.")
    g.add_argument("--save-activity", default=None,
                   help="Optional .npz dump of the per-frame activity arrays.")
    g.add_argument("--plot-timeline", default=None,
                   help="Optional PNG of activity-vs-frame for the whole movie.")
    return ap


def main(argv=None):
    p = build_parser().parse_args(argv)
    t0 = time.time()

    fps, _, _meta, info = cc.resolve_calibration(p.input_dir, p.fps, None)
    for line in info:
        print(f"[detect] {line}", file=sys.stderr)
    p.fps = fps

    files = cc.list_frames(p.input_dir, p.pattern)
    n = len(files)
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

    events = group_events(a, p)
    print(f"[detect] {len(events)} event(s) detected "
          f"in {time.time()-t0:.1f}s", file=sys.stderr)

    cols = ["event_id", "start_frame", "end_frame", "peak_frame",
            "n_active", "span", "peak_area_px", "mean_area_px",
            "peak_snr", "peak_energy", "polarity",
            "onset_cx", "onset_cy", "peak_cx", "peak_cy",
            "start_index", "end_index", "peak_index"]
    with open(p.out, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=cols)
        wcsv.writeheader()
        for eid, ev in enumerate(events):
            ev = dict(ev, event_id=eid)
            wcsv.writerow({c: ev.get(c, "") for c in cols})
    print(f"[detect] wrote {p.out}", file=sys.stderr)

    if p.save_activity:
        np.savez_compressed(p.save_activity, min_area=p.min_area,
                            pix_k=p.pix_k, smooth=p.smooth, **a)
        print(f"[detect] wrote {p.save_activity}", file=sys.stderr)
    if p.plot_timeline:
        plot_timeline(a, events, p, p.plot_timeline)
        print(f"[detect] wrote {p.plot_timeline}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

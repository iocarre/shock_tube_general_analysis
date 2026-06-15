#!/usr/bin/env python3
"""
analyze_activity.py  --  Script 2: characterise each detected change event.
===========================================================================

Reads the events found by ``detect_activity.py`` and, for each one, reloads only
that event's frames (plus a margin), then measures *what the change is doing*:

  * how strong it is        -- active-pixel area, pooled SNR, integrated SNR,
  * where it is             -- centroid (x, y) and bounding box per frame,
  * how it moves            -- centroid trajectory and its velocity (px/frame,
                               and m/s if --fps and --px-size-um are known),
  * how far it reaches      -- leading-edge distance from the onset centroid,
  * how it grows / fades    -- active area vs time, onset / peak / end timing,
  * whether it is brighter or darker than the quiescent background (polarity).

Unlike the sister project there is **no assumed shape or straight-line motion**;
the trajectory fit is a descriptive summary of where the activity's centroid
goes, not an acceptance gate.

For each event it writes a per-frame CSV and (optionally) two diagnostic PNGs:
a montage of pooled-SNR difference images across the event with the active
region outlined, and the centroid-trajectory / area-vs-time curves. A summary
CSV aggregates one row per event.

Example
-------
    python3 analyze_activity.py run activity_events.csv --out-dir analysis \\
        --px-size-um 50 --plots
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

import change_common as cc


def load_events(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def local_background(window, core_mask, p):
    """
    Per-pixel background + noise (and pooled-SNR threshold) from the *margin*
    frames of an event window -- the ones outside the detected core -- so the
    change itself does not bias the baseline. Falls back to the whole window
    when too few margin frames exist.
    """
    margin = window[~core_mask] if (~core_mask).sum() >= 5 else window
    bg, sigma = cc.estimate_bg_noise(margin, p.noise_floor_frac)
    sigma2 = sigma * sigma
    inflation = cc.pooled_snr_scale(margin, bg, sigma2, p, p.edge_margin)
    return bg, sigma2, p.pix_k * inflation


def analyze_event(ev, files, fnum_to_idx, p):
    """Measure one event. Returns (summary, per_frame_rows, plot_pack)."""
    i0 = fnum_to_idx[int(ev["start_frame"])]
    i1 = fnum_to_idx[int(ev["end_frame"])]
    a = max(0, i0 - p.margin)
    b = min(len(files) - 1, i1 + p.margin)
    idxs = list(range(a, b + 1))

    window = np.stack([cc.load_frame(files[i], p.row_lo, p.row_hi,
                                     p.col_lo, p.col_hi) for i in idxs], axis=0)
    # True for the detected event frames, False for the surrounding margin
    # frames (which supply the local quiescent baseline).
    core_mask = np.array([(i0 <= i <= i1) for i in idxs])
    bg, sigma2, thresh = local_background(window, core_mask, p)

    snr_stack = np.empty_like(window)
    rows = []
    onset_centroid = None
    for r, i in enumerate(idxs):
        snr = cc.activity_snr(window[r], bg, sigma2, p.smooth, p.detrend_band)
        snr_stack[r] = snr
        mask = np.abs(snr) > thresh
        if p.edge_margin > 0:
            mask &= cc._edge_mask(snr.shape, p.edge_margin)

        in_core = bool(i0 <= i <= i1)
        if not mask.any():
            rows.append(_empty_row(files[i], i, in_core))
            continue

        w = np.abs(snr[mask])
        ys, xs = np.nonzero(mask)
        wsum = float(w.sum())
        cx = float((xs * w).sum() / wsum)
        cy = float((ys * w).sum() / wsum)
        blob_size, _ = cc.largest_blob(mask)
        signed = float(snr[mask].mean())

        # Leading edge = farthest active pixel from where the event first
        # appeared (its onset centroid) -> how far the activity has reached.
        if onset_centroid is None and in_core:
            onset_centroid = (cx, cy)
        if onset_centroid is not None:
            lead = float(np.max(np.hypot(xs - onset_centroid[0],
                                         ys - onset_centroid[1])))
        else:
            lead = 0.0

        rows.append({
            "frame": int(cc.frame_number(files[i])),
            "order_index": i,
            "in_core": in_core,
            "area_px": int(mask.sum()),
            "blob_px": int(blob_size),
            "centroid_x": round(cx, 3),
            "centroid_y": round(cy, 3),
            "bbox_w": int(xs.max() - xs.min() + 1),
            "bbox_h": int(ys.max() - ys.min() + 1),
            "leading_edge_px": round(lead, 3),
            "peak_snr": round(float(w.max()), 3),
            "energy": round(wsum, 2),
            "mean_snr": round(signed, 3),
            "polarity": "brighter" if signed > 0 else "darker",
        })

    summary = summarize(ev, rows, p)
    return summary, rows, (window, snr_stack, idxs, rows, p, thresh)


def _empty_row(path, i, in_core):
    return {
        "frame": int(cc.frame_number(path)), "order_index": i,
        "in_core": in_core, "area_px": 0, "blob_px": 0,
        "centroid_x": "", "centroid_y": "", "bbox_w": 0, "bbox_h": 0,
        "leading_edge_px": 0.0, "peak_snr": 0.0, "energy": 0.0,
        "mean_snr": 0.0, "polarity": "",
    }


def _empty_summary(ev, rows, p):
    """
    Degenerate summary for an event with no active frames under the local
    baseline: keep the declared frame range, zero every measurement.
    """
    core_idx = [r["order_index"] for r in rows if r["in_core"]]
    span = (max(core_idx) - min(core_idx) + 1) if core_idx else 0
    summary = {
        "event_id": ev.get("event_id", ""),
        "start_frame": int(ev.get("start_frame", 0) or 0),
        "peak_frame": int(ev.get("peak_frame", ev.get("start_frame", 0)) or 0),
        "end_frame": int(ev.get("end_frame", 0) or 0),
        "n_frames_active": 0,
        "duration_frames": int(span),
        "polarity": ev.get("polarity", ""),
        "peak_area_px": 0,
        "mean_area_px": 0.0,
        "peak_snr": 0.0,
        "max_leading_edge_px": 0.0,
        "onset_cx": "", "onset_cy": "",
        "peak_cx": "", "peak_cy": "",
        "centroid_travel_px": 0.0,
        "centroid_vx_px_per_frame": 0.0,
        "centroid_vy_px_per_frame": 0.0,
        "centroid_speed_px_per_frame": 0.0,
    }
    if p.fps:
        summary["centroid_speed_px_per_s"] = 0.0
        summary["duration_us"] = round(span / p.fps * 1e6, 3)
        if p.px_size_um:
            summary["centroid_speed_m_per_s"] = 0.0
            summary["max_leading_edge_mm"] = 0.0
    return summary


def summarize(ev, rows, p):
    """Aggregate per-frame rows into one event summary, with a trajectory fit."""
    # Prefer the active frames inside the detected core; fall back to any active
    # frame in the window if the core itself shows none.
    core = [r for r in rows if r["in_core"] and r["area_px"] > 0]
    if not core:
        core = [r for r in rows if r["area_px"] > 0]
    if not core:
        # No frame in the window crosses the local threshold -- the event was
        # marginal under detection's global background but vanishes against the
        # stricter local one. Emit a zeroed summary rather than crashing.
        return _empty_summary(ev, rows, p)

    t = np.array([r["order_index"] for r in core], dtype=np.float64)
    cx = np.array([float(r["centroid_x"]) for r in core], dtype=np.float64)
    cy = np.array([float(r["centroid_y"]) for r in core], dtype=np.float64)
    areas = np.array([r["area_px"] for r in core], dtype=np.float64)
    tt = t - t.min()                          # frames since onset (0-based)

    # Descriptive linear fit of the centroid path (vx, vy) -> speed. This is a
    # summary of where the activity went, NOT an acceptance gate.
    if len(core) >= 2 and tt.max() > 0:
        vx = float(np.polyfit(tt, cx, 1)[0])
        vy = float(np.polyfit(tt, cy, 1)[0])
    else:
        vx = vy = 0.0
    speed = float(np.hypot(vx, vy))
    peak_row = max(core, key=lambda r: r["area_px"])

    pol = "brighter" if np.mean(
        [1 if r["polarity"] == "brighter" else 0 for r in core]) >= 0.5 \
        else "darker"

    summary = {
        "event_id": ev.get("event_id", ""),
        "start_frame": int(core[0]["frame"]),
        "peak_frame": int(peak_row["frame"]),
        "end_frame": int(core[-1]["frame"]),
        "n_frames_active": len(core),
        "duration_frames": int(core[-1]["order_index"]
                               - core[0]["order_index"] + 1),
        "polarity": pol,
        "peak_area_px": int(peak_row["area_px"]),
        "mean_area_px": round(float(areas.mean()), 1),
        "peak_snr": round(max(r["peak_snr"] for r in core), 3),
        "max_leading_edge_px": round(max(r["leading_edge_px"] for r in core), 2),
        "onset_cx": round(cx[0], 2), "onset_cy": round(cy[0], 2),
        "peak_cx": round(float(peak_row["centroid_x"]), 2),
        "peak_cy": round(float(peak_row["centroid_y"]), 2),
        "centroid_travel_px": round(float(np.hypot(cx[-1] - cx[0],
                                                   cy[-1] - cy[0])), 2),
        "centroid_vx_px_per_frame": round(vx, 4),
        "centroid_vy_px_per_frame": round(vy, 4),
        "centroid_speed_px_per_frame": round(speed, 4),
    }

    if p.fps:
        summary["centroid_speed_px_per_s"] = round(speed * p.fps, 2)
        summary["duration_us"] = round(
            summary["duration_frames"] / p.fps * 1e6, 3)
        if p.px_size_um:
            px_m = p.px_size_um * 1e-6
            summary["centroid_speed_m_per_s"] = round(speed * p.fps * px_m, 4)
            summary["max_leading_edge_mm"] = round(
                summary["max_leading_edge_px"] * p.px_size_um * 1e-3, 4)
    return summary


# --------------------------------------------------------------------------- #
# Diagnostic plots
# --------------------------------------------------------------------------- #

def make_plots(eid, pack, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _window, snr_stack, idxs, rows, p, thresh = pack
    nwin = snr_stack.shape[0]
    core_pos = [r for r in range(nwin) if rows[r]["in_core"]]
    lim = np.percentile(np.abs(snr_stack), 99.5) or 1.0

    # ---- montage of pooled-SNR difference images across the event ---------
    pick = core_pos if core_pos else list(range(nwin))
    if len(pick) > 6:
        pick = [pick[int(round(j))]
                for j in np.linspace(0, len(pick) - 1, 6)]
    ncol = len(pick)
    fig, axes = plt.subplots(1, ncol, figsize=(2.6 * ncol, 3.0), squeeze=False)
    for ax, r in zip(axes[0], pick):
        ax.imshow(snr_stack[r], cmap="seismic", vmin=-lim, vmax=lim,
                  aspect="auto")
        ax.contour(np.abs(snr_stack[r]) > thresh, levels=[0.5],
                   colors="lime", linewidths=0.8)
        row = rows[r]
        if row["centroid_x"] != "":
            ax.plot(float(row["centroid_x"]), float(row["centroid_y"]),
                    "x", color="black", ms=7, mew=1.5)
        ax.set_title(f"frame {row['frame']}\narea {row['area_px']}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"event {eid}: pooled SNR (red=brighter, blue=darker)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_montage.png", dpi=110)
    plt.close(fig)

    # ---- centroid trajectory + area-vs-time ------------------------------
    act = [r for r in rows if r["area_px"] > 0]
    frames = [r["frame"] for r in act]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    if act:
        cxs = [float(r["centroid_x"]) for r in act]
        cys = [float(r["centroid_y"]) for r in act]
        sc = ax1.scatter(cxs, cys, c=frames, cmap="viridis", s=24)
        ax1.plot(cxs, cys, color="gray", lw=0.6, zorder=0)
        ax1.invert_yaxis()
        ax1.set_xlabel("centroid x (px)"); ax1.set_ylabel("centroid y (px)")
        ax1.set_title("centroid trajectory")
        fig.colorbar(sc, ax=ax1, label="frame")

        ax2.plot(frames, [r["area_px"] for r in act], "-o", ms=3,
                 color="firebrick")
        for r in act:
            if r["in_core"]:
                ax2.axvspan(r["frame"] - 0.5, r["frame"] + 0.5,
                            color="orange", alpha=0.06)
    ax2.set_xlabel("frame"); ax2.set_ylabel("active-pixel area")
    ax2.set_title("activity growth / decay")
    fig.suptitle(f"event {eid}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_trajectory.png", dpi=110)
    plt.close(fig)


def dump_frames(eid, pack, out_dir):
    """
    Write one treated image per frame of the event window into a per-event
    subfolder: the raw grayscale camera frame with the detected active region
    outlined (lime) and the centroid marked (red x). Covers the full analysis
    window (detected core *plus* the +/- margin context frames).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window, snr_stack, idxs, rows, p, thresh = pack
    sub = os.path.join(out_dir, f"event_{eid}_frames")
    os.makedirs(sub, exist_ok=True)

    h, w = window.shape[1], window.shape[2]
    n = 0
    for r in range(window.shape[0]):
        row = rows[r]
        mask = np.abs(snr_stack[r]) > thresh
        if p.edge_margin > 0:
            mask &= cc._edge_mask(snr_stack[r].shape, p.edge_margin)

        fig, ax = plt.subplots(figsize=(w / 100.0, h / 100.0 + 0.4))
        ax.imshow(window[r], cmap="gray", aspect="equal")
        if mask.any():
            ax.contour(mask, levels=[0.5], colors="lime", linewidths=0.9)
        if row["centroid_x"] != "":
            ax.plot(float(row["centroid_x"]), float(row["centroid_y"]),
                    "x", color="red", ms=9, mew=1.6)
        tag = "core" if row["in_core"] else "margin"
        ax.set_title(f"frame {row['frame']} [{tag}]  area {row['area_px']} px "
                     f"({row['polarity'] or 'quiescent'})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(sub, f"frame_{row['frame']:09d}.png"), dpi=110)
        plt.close(fig)
        n += 1
    return sub, n


def make_movie(eid, pack, out_dir, movie_fps=12):
    """
    Render the event window as an MP4: the raw grayscale frame with the
    detected active region outlined (lime) and the centroid marked (red x),
    one video frame per camera frame over the full core+margin window.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    window, snr_stack, idxs, rows, p, thresh = pack
    h, w = window.shape[1], window.shape[2]
    vmin, vmax = float(window.min()), float(window.max())

    fig, ax = plt.subplots(figsize=(w / 100.0, h / 100.0 + 0.4))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(window[0], cmap="gray", aspect="equal", vmin=vmin, vmax=vmax)
    centroid, = ax.plot([], [], "x", color="red", ms=9, mew=1.6)
    title = ax.set_title("")
    state = {"contour": None}

    def draw(r):
        im.set_data(window[r])
        if state["contour"] is not None:
            state["contour"].remove()
            state["contour"] = None
        mask = np.abs(snr_stack[r]) > thresh
        if p.edge_margin > 0:
            mask &= cc._edge_mask(snr_stack[r].shape, p.edge_margin)
        if mask.any():
            state["contour"] = ax.contour(mask, levels=[0.5], colors="lime",
                                          linewidths=0.9)
        row = rows[r]
        if row["centroid_x"] != "":
            centroid.set_data([float(row["centroid_x"])],
                              [float(row["centroid_y"])])
        else:
            centroid.set_data([], [])
        tag = "core" if row["in_core"] else "margin"
        title.set_text(f"event {eid}  frame {row['frame']} [{tag}]  "
                       f"area {row['area_px']} px "
                       f"({row['polarity'] or 'quiescent'})")
        return []

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=window.shape[0], blit=False)
    path = os.path.join(out_dir, f"event_{eid}.mp4")
    anim.save(path, writer=FFMpegWriter(fps=movie_fps, bitrate=2400), dpi=110)
    plt.close(fig)
    return path, window.shape[0]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    ap = argparse.ArgumentParser(
        description="Characterise detected change events.")
    ap.add_argument("input_dir", help="Directory of single-frame TIFFs.")
    ap.add_argument("events_csv", help="CSV from detect_activity.py.")
    ap.add_argument("--pattern", default="*.tif")
    ap.add_argument("--out-dir", default="analysis", help="Output directory.")

    g = ap.add_argument_group("region of interest (match detection)")
    g.add_argument("--row-lo", type=int, default=None)
    g.add_argument("--row-hi", type=int, default=None)
    g.add_argument("--col-lo", type=int, default=None)
    g.add_argument("--col-hi", type=int, default=None)
    g.add_argument("--edge-margin", type=int, default=6)

    g = ap.add_argument_group("change detection (match detection)")
    g.add_argument("--smooth", type=int, default=5)
    g.add_argument("--detrend-band", type=int, default=101)
    g.add_argument("--pix-k", type=float, default=6.0)
    g.add_argument("--noise-floor-frac", type=float, default=0.25)
    g.add_argument("--margin", type=int, default=10,
                   help="Frames padded around each event for the local "
                        "baseline and context.")

    g = ap.add_argument_group("physical calibration (optional)")
    g.add_argument("--fps", type=float, default=None,
                   help="Camera frame rate; auto-read from a .cihx if present.")
    g.add_argument("--px-size-um", type=float, default=None,
                   help="Pixel size in micrometres (for m/s and mm output); "
                        "auto-read from a .cihx when spatially calibrated.")

    ap.add_argument("--plots", action="store_true",
                    help="Write diagnostic PNGs per event.")
    ap.add_argument("--dump-frames", action="store_true",
                    help="Write one treated image per frame (raw frame + "
                         "detected-region outline) into a per-event subfolder.")
    ap.add_argument("--movie", action="store_true",
                    help="Render an MP4 per event (raw frame + detected-region "
                         "outline) over the full core+margin window.")
    ap.add_argument("--movie-fps", type=float, default=12.0,
                    help="Playback frame rate of the MP4 (default 12).")
    return ap


def main(argv=None):
    p = build_parser().parse_args(argv)
    os.makedirs(p.out_dir, exist_ok=True)

    fps, px, _meta, info = cc.resolve_calibration(
        p.input_dir, p.fps, p.px_size_um)
    for line in info:
        print(f"[analyze] {line}", file=sys.stderr)
    p.fps, p.px_size_um = fps, px

    files = cc.list_frames(p.input_dir, p.pattern)
    fnum_to_idx = {cc.frame_number(f): i for i, f in enumerate(files)}

    events = load_events(p.events_csv)
    print(f"[analyze] {len(events)} event(s) from {p.events_csv}",
          file=sys.stderr)
    if not events:
        print("[analyze] nothing to do.", file=sys.stderr)
        return 0

    summaries = []
    for ev in events:
        eid = ev.get("event_id", "?")
        summary, rows, pack = analyze_event(ev, files, fnum_to_idx, p)
        summaries.append(summary)

        per_frame_path = os.path.join(p.out_dir, f"event_{eid}_frames.csv")
        with open(per_frame_path, "w", newline="") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows)

        if p.plots:
            make_plots(eid, pack, os.path.join(p.out_dir, f"event_{eid}"))

        if p.dump_frames:
            sub, n = dump_frames(eid, pack, p.out_dir)
            print(f"[analyze] event {eid}: wrote {n} frame image(s) -> {sub}",
                  file=sys.stderr)

        if p.movie:
            mpath, n = make_movie(eid, pack, p.out_dir, p.movie_fps)
            print(f"[analyze] event {eid}: wrote {n}-frame movie -> {mpath}",
                  file=sys.stderr)

        spd = (f"{summary.get('centroid_speed_m_per_s')} m/s"
               if "centroid_speed_m_per_s" in summary
               else f"{summary['centroid_speed_px_per_frame']} px/frame")
        print(f"[analyze] event {eid}: frames "
              f"{summary['start_frame']}-{summary['end_frame']} "
              f"({summary['n_frames_active']} active), {summary['polarity']}, "
              f"peak area {summary['peak_area_px']} px, centroid {spd}",
              file=sys.stderr)

    summary_path = os.path.join(p.out_dir, "events_summary.csv")
    cols = []
    for s in summaries:
        for k in s:
            if k not in cols:
                cols.append(k)
    with open(summary_path, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=cols)
        wcsv.writeheader()
        for s in summaries:
            wcsv.writerow({c: s.get(c, "") for c in cols})
    print(f"[analyze] wrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

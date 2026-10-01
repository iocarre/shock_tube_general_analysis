"""
event_overview.py
=================

Label each detected event with its most likely cause, and draw one overview
sheet per recording that answers "is anything seen?" at a glance.

Labels (first matching rule wins, strongest evidence first):

  shock-candidate  overlaps a moving front found by the x-t shock search,
                   over the whole field or inside a micro-tube's bore
  slow-drift       lasts a large part of the recording (illumination or
                   background drifting, not a transient)
  camera           the camera changed its timing / exposure on these frames
                   (trigger frame, read from a Phantom .chd)
  flash            the whole frame's brightness jumps (flash, light glitch,
                   sensor recovering from saturation)
  global           covers a large part of the field at once
  stripe           tall, narrow region across the tube: a front caught in one
                   or two frames, or a sensor-row artefact -- check by eye
  vibration        explained by a sub-pixel shift of the whole image
  particle         a small compact blob: dust, droplet, particle
  noise            scattered pixels close to the threshold

The rules use the measurements of ``analyze_activity`` plus the flags written
by ``detect_activity`` (camera / flash frames, shock-search candidates, in
the field and in the tube). They
are deliberately simple and each label comes with a ``label_reason`` giving the
numbers behind it.
"""

from __future__ import annotations

import csv
import os

import numpy as np

import change_common as cc
import xt_diagram as xtd


LABELS = {                      # same colours as the HTML dashboard
    "shock-candidate": "#2a78d6",
    "slow-drift": "#eb6834",
    "camera": "#1baf7a",
    "flash": "#eda100",
    "global": "#e87ba4",
    "stripe": "#008300",
    "vibration": "#4a3aa7",
    "particle": "#e34948",
    "noise": "#898781",
}

FIRST_COLS = ["event_id", "label", "label_reason"]


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

def shift_fit(frame, bg, mask, k=3):
    """
    How much of the change around ``mask`` a small rigid shift of the image
    explains: fit (frame - bg) ~ -(sx * dbg/dx + sy * dbg/dy) + c on the active
    pixels and a 3-px band around them. Returns (R^2, shift in px).
    """
    d = cc.box2d_mean(frame - bg, k)
    gy, gx = np.gradient(cc.box2d_mean(bg, k))
    near = cc.box2d_mean(mask.astype(np.float32), 7) > 1e-3
    if near.sum() < 10:
        return 0.0, 0.0
    A = np.stack([gx[near], gy[near], np.ones(int(near.sum()))], 1)
    y = d[near]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / tot if tot > 0 else 0.0
    return max(r2, 0.0), float(np.hypot(coef[0], coef[1]))


def classify(summary, ev, rows, pack, n_total):
    """
    Add the shape features, ``label`` and ``label_reason`` to ``summary``
    (in place) and return the tile used by the overview sheet.
    """
    window, snr_stack, _idxs, _rows, _p, thresh, valid, bg = pack
    rows_h = int(valid.any(1).sum()) or bg.shape[0]     # measured tube height
    n_valid = int(valid.sum()) or 1

    act = [r for r in range(len(rows)) if rows[r]["in_core"]
           and rows[r]["area_px"] > 0]
    cands = str(ev.get("shock_candidates", "") or "").strip()
    tcands = str(ev.get("tube_candidates", "") or "").strip()
    cam = int(ev.get("camera_frames") or 0)
    fl = int(ev.get("flash_frames") or 0)
    dur = int(summary.get("duration_frames") or 0)

    feat = {"peak_area_frac": 0.0, "blob_w_px": 0, "blob_h_px": 0,
            "blob_frac": 0.0, "shift_r2": 0.0, "shift_px": 0.0,
            "camera_frames": cam, "flash_frames": fl,
            "shock_candidates": cands}
    r_peak = max(act, key=lambda r: rows[r]["area_px"]) if act else None
    mask = np.zeros(valid.shape, bool)
    if r_peak is not None:
        row = rows[r_peak]
        mask = (np.abs(snr_stack[r_peak]) > thresh) & valid
        r2, shift = shift_fit(window[r_peak], bg, mask)
        feat.update(peak_area_frac=round(row["area_px"] / n_valid, 4),
                    blob_w_px=row["blob_w"], blob_h_px=row["blob_h"],
                    blob_frac=round(row["blob_px"] / row["area_px"], 3),
                    shift_r2=round(r2, 3), shift_px=round(shift, 3))

    bw, bh = feat["blob_w_px"], feat["blob_h_px"]
    if cands or tcands:
        label = "shock-candidate"
        why = "; ".join(
            ([f"overlaps shock candidate(s) {cands}"] if cands else [])
            + ([f"overlaps front(s) {tcands} found in the tube"]
               if tcands else []))
    elif dur >= max(100, 0.2 * n_total):
        label, why = "slow-drift", f"lasts {dur} of {n_total} frames"
    elif cam and dur <= 4:
        label, why = "camera", (f"{cam} frame(s) with irregular camera "
                                "timing/exposure")
    elif fl:
        label, why = "flash", f"{fl} frame(s) with a whole-frame brightness jump"
    elif r_peak is None:
        label, why = "noise", "no active pixel against the local baseline"
    elif feat["peak_area_frac"] >= 0.2:
        label, why = "global", (f"peak covers {100 * feat['peak_area_frac']:.0f}"
                                "% of the measured field")
    elif bh >= 0.6 * rows_h and bw <= 0.15 * rows_h:
        label, why = "stripe", f"tall narrow region {bw}x{bh} px"
    elif feat["shift_r2"] >= 0.5:
        label, why = "vibration", (f"a {feat['shift_px']:.2f} px image shift "
                                   f"explains {100 * feat['shift_r2']:.0f}% "
                                   "of the change")
    elif (feat["blob_frac"] >= 0.5 and bh <= 0.5 * rows_h
          and bw <= 0.5 * rows_h):
        label, why = "particle", f"compact blob {bw}x{bh} px"
    else:
        label, why = "noise", (f"scattered: largest blob "
                               f"{rows[r_peak]['blob_px']} of "
                               f"{rows[r_peak]['area_px']} px")
    summary.update(label=label, label_reason=why, **feat)

    if r_peak is None:              # show the event's own peak frame anyway
        r_peak_img = next((k for k, row in enumerate(rows)
                           if row["frame"] == summary.get("peak_frame")),
                          len(rows) // 2)
    else:
        r_peak_img = r_peak
    img = window[r_peak_img]
    frame = rows[r_peak_img]["frame"]
    return {"label": label, "color": LABELS[label],
            "title": f"event {summary.get('event_id')}  frame {frame}  {label}",
            "image": img.astype(np.float32), "mask": mask, "valid": valid,
            "peak_snr": float(summary.get("peak_snr") or 0.0)}


MOMENT_COLOR = "#606060"


def moment_tile(m, rows, pack):
    """Overview tile of one strongest moment (see detect_activity)."""
    window, snr_stack, _idxs, _rows, _p, thresh, valid, _bg = pack
    r = next((k for k, row in enumerate(rows)
              if row["frame"] == int(m["frame"])), len(rows) // 2)
    mask = (np.abs(snr_stack[r]) > thresh) & valid
    return {"label": "moment", "color": MOMENT_COLOR,
            "title": f"moment {m['rank']}  frame {m['frame']}  "
                     f"{m['area_px']} px (below threshold)",
            "image": window[r].astype(np.float32), "mask": mask,
            "valid": valid, "peak_snr": float(m["peak_snr"]),
            "marker": (float(m["cx"]), float(m["cy"]))
            if int(m["area_px"]) > 0 else None}


def save_tile(tile, path):
    """Save one tile (frame, active region outlined, optional red circle)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = tile["image"]
    h, w = img.shape
    vals = img[tile["valid"]] if tile["valid"].any() else img.ravel()
    lo, hi = np.percentile(vals, [1, 99.5])
    fig = plt.figure(figsize=(w / 100.0, h / 100.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, aspect="equal")
    if tile["mask"].any():
        ax.contour(tile["mask"], levels=[0.5], colors="lime", linewidths=0.8)
    if tile.get("marker"):
        ax.plot(*tile["marker"], "o", ms=14, mfc="none", mec="red", mew=1.2)
    ax.set_axis_off()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_moments(events_csv):
    """Strongest moments written by detection next to the events CSV."""
    path = cc.output_base(events_csv) + "_moments.csv"
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# Overview sheet
# --------------------------------------------------------------------------- #

def _read_cands(path, along="x0_px"):
    """Candidates of a _shocks.csv (``along``: the position column drawn)."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [{"candidate_id": int(r["candidate_id"]),
                 "score": float(r["score"]), "t0": int(r["start_index"]),
                 "n_frames": int(r["n_frames"]), "x0_px": float(r[along]),
                 "v_px_per_frame": float(r["v_px_per_frame"]),
                 "start_frame": int(r["start_frame"]),
                 "end_frame": int(r["end_frame"]),
                 "where": r.get("where", ""),
                 "x_frame_px": float(r["x0_px"])}
                for r in csv.DictReader(fh)]


def load_tube(events_csv):
    """The tube found by detection, its x-t diagram's candidates, and the
    numbers of its search: (tube dict or None, reason, candidates)."""
    import json
    base = cc.output_base(events_csv)
    act = None
    if os.path.exists(base + "_activity.npz"):
        act = np.load(base + "_activity.npz")
    if act is not None and "tube_json" in act:
        return (json.loads(str(act["tube_json"])), "",
                _read_cands(base + "_tube_shocks.csv", "s0_px"))
    if act is not None and "tube_reason" in act:
        return None, str(act["tube_reason"]), []
    return None, "not searched (older detection results)", []


def _load_detection(events_csv):
    """The .npz and shock candidates written by detection, if present."""
    base = cc.output_base(events_csv)
    act = None
    if os.path.exists(base + "_activity.npz"):
        act = dict(np.load(base + "_activity.npz"))
    cands = _read_cands(base + "_shocks.csv")
    return act, cands


WHERE = {"inside": "inside the tube",
         "outside too": "also outside the walls: a wave outside the tube, "
                        "seen through it",
         "unclear": "inside or outside: unclear",
         "tube moving": "the tube itself moves in these frames: may be its "
                        "walls moving, not a front"}


def tube_lines(act, tube, reason, tcands):
    """What the overview says about the micro-tube and the fronts in it."""
    if tube is None:
        return [f"Tube: none -- {reason}."]
    w0, w1 = tube["bore_px_ends"]
    bore = (f"{tube['bore_px']:.2f} px" if abs(w1 - w0) < 0.5
            else f"tapered {w0:.1f} -> {w1:.1f} px")
    lines = [f"Tube: bore {bore}, {tube['angle_deg']:+.2f} deg, seen over "
             f"{tube['seen_px']:.0f} px."]
    if act is None or not bool(act.get("shock_search", False)):
        return lines
    if tcands:
        inside = [c for c in tcands if c["where"] == "inside"]
        best = max(inside or tcands, key=lambda c: c["score"])
        lines.append(
            f"Fronts in the tube's bore: {len(tcands)} ({len(inside)} inside "
            f"the tube). Best: score {best['score']:.1f}, frames "
            f"{best['start_frame']}-{best['end_frame']}, "
            f"{best['v_px_per_frame']:+.1f} px/frame along the tube, "
            f"{WHERE.get(best['where'], best['where'])} (details in "
            "_tube_shocks.csv).")
    else:
        sens = {int(k): (a, pct)
                for k, a, pct in act.get("tube_sensitivity", [])}
        s = (f" A front filling the bore of >= {sens[8][1]:.2f}% of the "
             "brightness seen over 8 frames would have been reported."
             if 8 in sens else "")
        lines.append(f"Fronts in the tube's bore: none (best track score "
                     f"{float(act['tube_best_score']):.1f} < "
                     f"{float(act['shock_k']):g}).{s}")
    return lines


def verdict_lines(act, cands, summaries, moments=(), tube_info=None):
    """The plain-language summary printed at the top of the sheet."""
    lines = []
    if act is None or not bool(act.get("shock_search", False)):
        lines.append("Shock search: not run (no detection .npz, or "
                     "--no-shock-search).")
    elif cands:
        best = max(cands, key=lambda c: c["score"])
        lines.append(
            f"Shock search: {len(cands)} candidate(s) above "
            f"{float(act['shock_k']):g} sigma. Best: score {best['score']:.1f}, "
            f"frames {best['start_frame']}-{best['end_frame']}, "
            f"{best['v_px_per_frame']:+.1f} px/frame (yellow guides; details "
            "in _shocks.csv).")
    else:
        sens = {int(k): (a, pct) for k, a, pct in act.get("sensitivity", [])}
        s = ""
        if 8 in sens:
            s = (f" A full-height front of >= {sens[8][1]:.2f}% of the "
                 f"brightness seen over 8 frames would have been reported"
                 + (f" ({sens[4][1]:.2f}% over 4, {sens[32][1]:.2f}% over 32)"
                    if 4 in sens and 32 in sens else "") + ".")
        lines.append(f"Shock search: no moving front found (best track score "
                     f"{float(act['best_score']):.1f} < "
                     f"{float(act['shock_k']):g}).{s}")
    if tube_info is not None:
        lines += tube_lines(act, *tube_info)
    if act is not None:
        fnum = act["fnum"]
        cam = [int(fnum[i]) for i in act.get("camera_frames", [])]
        fl = [int(fnum[i]) for i in act.get("flash_frames", [])]
        trig = int(act.get("trigger_index", -1))
        parts = []
        if trig >= 0 and trig < len(fnum):
            parts.append(f"trigger at frame {int(fnum[trig])}")
        if cam:
            parts.append(f"camera timing/exposure change at frame(s) "
                         f"{cam[:6]}")
        if fl:
            parts.append(f"brightness jump at frame(s) {fl[:8]}"
                         + (" ..." if len(fl) > 8 else ""))
        if parts:
            lines.append("Flagged: " + "; ".join(parts) + ".")
    if summaries:
        counts = {}
        for s in summaries:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
        order = [k for k in LABELS if k in counts]
        lines.append(f"Events: {len(summaries)} -- "
                     + ", ".join(f"{counts[k]} {k}" for k in order) + ".")
    else:
        lines.append("Events: none detected.")
    if moments:
        big = max(moments, key=lambda m: int(m["area_px"]))
        lines.append(f"Strongest moments below the event thresholds: "
                     f"{len(moments)} shown (grey tiles); the largest is "
                     f"{big['area_px']} active px at frame {big['frame']}. "
                     "Pure noise also produces a few such moments.")
    return lines


def make_overview(p, summaries, tiles, files, moments=(), moment_tiles=(),
                  max_tiles=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    act, cands = _load_detection(p.events_csv)
    tube, reason, tcands = load_tube(p.events_csv)
    lines = verdict_lines(act, cands, summaries, moments,
                          (tube, reason, tcands))
    has_tube = act is not None and "tube_xt" in act

    order = list(LABELS)
    shown = sorted(tiles, key=lambda t: (order.index(t["label"]),
                                         -t["peak_snr"]))[:max_tiles]
    n_events_shown = len(shown)
    shown += list(moment_tiles)
    ncol = 3
    ev_rows = int(np.ceil(n_events_shown / ncol))
    nrow = ev_rows + int(np.ceil(len(moment_tiles) / ncol))
    h, w = (shown[0]["image"].shape if shown else (1, 4))
    tile_h = max(1.0, 12.0 / ncol * h / w + 0.35)
    top_h = 6.5 if act is not None else 0.0
    # Zooms on the best fronts, from the field's diagram or the tube's.
    zooms = sorted([("field", c) for c in cands]
                   + [("tube", c) for c in (tcands if has_tube else [])],
                   key=lambda kc: -kc[1]["score"])[:ncol] \
        if act is not None else []
    zoom_h = 3.2 if zooms else 0.0
    head_h = 0.55 + 0.2 * len(lines)
    fig = plt.figure(figsize=(14, head_h + top_h + zoom_h + nrow * tile_h))
    heights = ([head_h] + ([top_h] if act is not None else [])
               + ([zoom_h] if zooms else []) + [tile_h] * nrow)
    gs = fig.add_gridspec(len(heights), 4 * ncol, height_ratios=heights,
                          hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    name = os.path.basename(os.path.normpath(p.input_dir))
    ax.text(0, 1, name, fontsize=14, weight="bold", va="top")
    ax.text(0, 1 - 0.4 / head_h, "\n".join(lines), fontsize=9, va="top",
            wrap=True)

    row0 = 1
    if act is not None:
        fnum = act["fnum"]
        flagged = sorted(set(act.get("camera_frames", []).tolist())
                         | set(act.get("flash_frames", []).tolist()))
        z, _ = xtd.normalise(act["xt"], flagged)
        zt = xtd.normalise(act["tube_xt"], flagged)[0] if has_tube else None
        xbin = int(act["xt_bin"])
        split = 5 if has_tube else 3 * ncol
        ax1 = fig.add_subplot(gs[1, :split])
        xtd.draw_xt(ax1, z, fnum, xbin, cands, flagged)
        ax1.set_title("x-t diagram: a shock is a straight slanted line "
                      "(yellow guides = candidate, grey = flagged frame)",
                      fontsize=9)
        if has_tube:
            axt = fig.add_subplot(gs[1, split:3 * ncol + 1], sharey=ax1)
            xtd.draw_xt(axt, zt, fnum, xbin, tcands, flagged)
            axt.set_xlabel("position along the tube (px)")
            axt.set_title("inside the tube's bore only", fontsize=9)
            axt.tick_params(labelleft=False)
            axt.set_ylabel("")
        ax2 = fig.add_subplot(gs[1, 3 * ncol + (1 if has_tube else 0):],
                              sharey=ax1)
        ax2.plot(act["area"], fnum, lw=0.7, color="black")
        for s in summaries:
            ax2.axhspan(s["start_frame"] - 0.5, s["end_frame"] + 0.5,
                        color=LABELS[s["label"]], alpha=0.5, lw=0)
        ax2.set_xscale("symlog", linthresh=max(1, int(act["min_area"])))
        ax2.set_xlim(0, max(10 * int(act["min_area"]),
                            1.5 * float(act["area"].max())))
        for m in moments:
            ax2.plot([max(1, int(m["area_px"]))], [int(m["frame"])], "<",
                     color=MOMENT_COLOR, ms=5)
        ax2.set_xlabel("active-pixel area")
        ax2.set_title("activity, events coloured by label", fontsize=9)
        ax2.tick_params(labelleft=False)
        used = [k for k in LABELS if any(s["label"] == k for s in summaries)]
        if used:
            ax2.legend(handles=[Patch(color=LABELS[k], label=k) for k in used],
                       fontsize=7, loc="lower right")
        row0 = 2

        # Zoom on the best candidates: at full-recording scale a fast front
        # crosses in a few frames and looks flat; here its slope is visible.
        for c_i, (kind, c) in enumerate(zooms):
            a = max(0, c["t0"] - 20)
            b = min(len(fnum), c["t0"] + c["n_frames"] + 20)
            axz = fig.add_subplot(gs[row0, 4 * c_i:4 * c_i + 4])
            local = dict(c, t0=c["t0"] - a)
            zz = zt if kind == "tube" else z
            xtd.draw_xt(axz, zz[a:b], fnum[a:b], xbin, [local],
                        [r - a for r in flagged if a <= r < b])
            if kind == "tube":
                axz.set_xlabel("position along the tube (px)")
                title = (f"tube front {c['candidate_id']}: score "
                         f"{c['score']:.1f}, {c['v_px_per_frame']:+.1f} "
                         f"px/frame\n{WHERE.get(c['where'], c['where'])}")
            else:
                title = (f"candidate {c['candidate_id']}: score "
                         f"{c['score']:.1f}, {c['v_px_per_frame']:+.1f} "
                         "px/frame")
            axz.set_title(title, fontsize=8, color=LABELS["shock-candidate"])
        if zooms:
            row0 += 1

    for n, t in enumerate(shown):
        # Events first, then the moments starting on a fresh row.
        k = n if n < n_events_shown else ev_rows * ncol + n - n_events_shown
        r, c = divmod(k, ncol)
        ax = fig.add_subplot(gs[row0 + r, 4 * c:4 * c + 4])
        img = t["image"]
        vals = img[t["valid"]] if t["valid"].any() else img.ravel()
        lo, hi = np.percentile(vals, [1, 99.5])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, aspect="equal")
        if t["mask"].any():
            ax.contour(t["mask"], levels=[0.5], colors="lime", linewidths=0.8)
        if t.get("marker"):
            ax.plot(*t["marker"], "o", ms=14, mfc="none", mec="red", mew=1.2)
        ax.set_title(t["title"], fontsize=8, color=t["color"])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(t["color"]); sp.set_linewidth(2)
    if len(tiles) > n_events_shown:
        fig.text(0.5, 0.005, f"showing {n_events_shown} of {len(tiles)} "
                 "events (strongest evidence first)", ha="center", fontsize=8)

    path = p.paths.overview
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return path

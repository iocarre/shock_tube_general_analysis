"""
front_sheet.py  --  one picture per front found inside a micro-tube.
====================================================================

For every front the detection found in a tube's bore (``NAME_tube_shocks.csv``)
the analysis draws ``tube_fronts/NAME_tube_front_<id>.png``, which shows
whether it is really inside the tube:

  * **the frames** along the front's track (up to 4), background-subtracted,
    around the tube, with every front present at that moment marked
    (o inside the tube, square: outside too, diamond: unclear);
  * **two x-t diagrams side by side**, inside the bore and in the band just
    outside the walls, around the front's frames: a front inside the tube
    shows only in the first, a wave outside the tube in both;
  * **position against time** of the fronts in that window: the position
    measured in each frame and the straight track fitted by the search;
  * **the profile across the tube** at the front, in each frame of its track,
    with the walls shaded: a front inside the tube changes the bore only.

It needs the frames, detection's ``NAME_activity.npz`` (the tube and its x-t
diagrams) and ``NAME_tube_shocks.csv``.
"""

from __future__ import annotations

import csv
import json
import os
import warnings

import numpy as np

import change_common as cc
import tube as tubemod
import xt_diagram as xtd

STYLE = {"inside": ("o", "#eb6834", "inside the tube"),
         "outside too": ("s", "#2a78d6", "outside too (a wave outside the "
                                         "tube, seen through it)"),
         "unclear": ("D", "#898781", "inside or outside: unclear"),
         "tube moving": ("X", "#4a3aa7", "the tube itself moves: may be its "
                                         "walls moving, not a front")}
INK, MUTED = "#222222", "#777777"


def _style(c):
    return STYLE.get(c.get("where") or "unclear", STYLE["unclear"])


def load_fronts(path):
    """Rows of a _tube_shocks.csv with numbers parsed."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            c = dict(r)
            for k in ("score", "outside_score", "tube_shift_px", "s0_px",
                      "s_end_px",
                      "v_px_per_frame", "x0_px", "y0_px", "x_end_px",
                      "y_end_px"):
                c[k] = float(r[k]) if r.get(k) not in (None, "") else None
            for k in ("candidate_id", "start_frame", "end_frame", "n_frames",
                      "start_index", "end_index"):
                c[k] = int(r[k])
            out.append(c)
    return out


def _change_maps(p, files, idxs, avoid):
    """Background-subtracted frames in pixel-noise units, each with its median
    over the lit pixels removed (a flash or exposure change brightens the
    whole frame). The background comes from frames away from ``avoid``."""
    n = len(files)
    lo, hi = avoid
    pool = [i for i in cc.sample_indices(n, 80) if i < lo or i > hi]
    pool = pool[:48] if len(pool) >= 8 else cc.sample_indices(n, 48)
    load = lambda i: cc.load_frame(files[i], p.row_lo, p.row_hi, p.col_lo,
                                   p.col_hi, p.rotate)
    bg, sigma = cc.estimate_bg_noise(np.stack([load(i) for i in pool]))
    lit = bg > 0.3 * np.percentile(bg, 99)
    maps = {}
    for i in idxs:
        z = (load(i) - bg) / sigma
        maps[i] = z - np.median(z[lit])
    return maps, bg, lit


def _shown(z, lit):
    """Map for display: 3x3 mean rescaled to unit noise (x3), unlit grey."""
    v = cc.box2d_mean(z, 3) * 3.0
    return np.where(lit, v, np.nan)


def _measured(zb, t, s_pred, xbin, sign, half=3):
    """Position of the front in frame ``t`` of the bore diagram: the
    strongest bin (of the front's polarity) within ``half`` bins of the
    track."""
    W = zb.shape[1]
    b = int(np.floor(s_pred / xbin))
    lo, hi = max(0, b - half), min(W, b + half + 1)
    if lo >= hi:
        return None
    seg = sign * zb[t, lo:hi]
    if not np.isfinite(seg).any() or np.nanmax(seg) <= 0:
        return None
    return (lo + int(np.nanargmax(seg)) + 0.5) * xbin


def make_sheet(p, files, act, tube, fronts, c, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fnum = act["fnum"]
    n = len(fnum)
    t_us = act["time_us"] if "time_us" in act else np.full(n, np.nan)
    xbin = int(act["xt_bin"])
    flagged = sorted(set(act.get("camera_frames", np.array([])).tolist())
                     | set(act.get("flash_frames", np.array([])).tolist()))
    t0, t1 = c["start_index"], c["end_index"]
    sign = 1.0 if c["polarity"] == "brighter" else -1.0
    a = np.deg2rad(tube.angle_deg)
    s_start = tube.s_range[0]

    # Fronts in the same frames (the outer wave next to the inside front, ...)
    w0, w1 = max(0, t0 - 6), min(n - 1, t1 + 10)
    near = [f for f in fronts
            if f["start_index"] <= w1 and f["end_index"] >= w0]
    pos = lambda f, t: f["s0_px"] + f["v_px_per_frame"] * (t - f["start_index"])

    track = list(range(t0, t1 + 1))
    shown = sorted(set(np.linspace(t0, t1, min(4, len(track))).round()
                       .astype(int).tolist()))
    maps, bg, lit = _change_maps(p, files, sorted(set(track) | set(shown)),
                                 (t0 - 10, t1 + 10))
    H, W = bg.shape

    zb = xtd.normalise(act["tube_xt"], flagged)[0]
    zs = xtd.normalise(act["sleeve_xt"], flagged)[0] \
        if "sleeve_xt" in act else None
    # For measuring positions: flagged frames kept (a front away from the
    # flash is still measurable there; such points are drawn hollow).
    zb_raw = xtd.normalise(act["tube_xt"], [], clip=1e9, highpass=0)[0]
    meas = {id(f): {t: _measured(zb_raw, t, pos(f, t), xbin,
                                 1.0 if f["polarity"] == "brighter" else -1.0)
                    for t in range(f["start_index"], f["end_index"] + 1)}
            for f in near}

    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.9, 1.2, 1.15], hspace=0.45,
                          wspace=0.25)

    # --- 1. frames around the tube
    ys = [tube.point(s, r)[1] for s in tube.s_range for r in tube.rho_wall]
    r_lo = int(max(0, min(ys) - 45))
    r_hi = int(min(H, max(ys) + 45))
    for k, t in enumerate(shown):
        ax = fig.add_subplot(gs[0, k])
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#cfcfcf")
        ax.imshow(_shown(maps[t], lit), cmap=cmap, vmin=-4, vmax=4,
                  interpolation="nearest", extent=[0, W, H, 0], aspect="auto")
        ax.set_ylim(r_hi, r_lo)
        for s0, s1 in tube.segments:
            for e in (0, 1):
                (xa, ya) = tube.point(s0, tube.bore_edges(s0)[e])
                (xb, yb) = tube.point(s1, tube.bore_edges(s1)[e])
                ax.plot([xa, xb], [ya, yb], ":", color=MUTED, lw=0.7)
        for f in near:
            if f["start_index"] <= t <= f["end_index"]:
                m, col, _ = _style(f)
                s = s_start + pos(f, t)
                lo, hi = tube.bore_edges(s)
                x, y = tube.point(s, 0.5 * (lo + hi))
                if 0 <= x < W and 0 <= y < H:       # not yet / no longer seen
                    ax.plot(x, y, m, mfc="none", mec=col, mew=2,
                            ms=13 if f is c else 10)
        ax.set_xlim(0, W)
        tt = t_us[t]
        ax.set_title(f"frame {int(fnum[t])}"
                     + (f"   t = {tt:.1f} µs" if np.isfinite(tt) else "")
                     + ("  (flagged)" if t in flagged else ""),
                     fontsize=10, color=INK)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.set_ylabel("row (bore edges dotted)", fontsize=9)
    fig.text(0.5, 0.672,
             "  ".join(f"{dict(o='○', s='□', D='◇', X='✕')[STYLE[w][0]]} "
                       f"{STYLE[w][2]}"
                       for w in STYLE if any((f.get("where") or "unclear") == w
                                             for f in near))
             + "      background-subtracted, red = brighter",
             ha="center", fontsize=10, color=INK)

    # --- 2. x-t inside the bore / just outside the walls
    win = slice(w0, w1 + 1)
    local = [dict(f, t0=f["start_index"] - w0, x0_px=f["s0_px"]) for f in near]
    lf = [r - w0 for r in flagged if w0 <= r <= w1]
    for j, (z, ttl) in enumerate(((zb, "x-t INSIDE the bore"),
                                  (zs, "x-t just OUTSIDE the walls "
                                       "(lit band 2-8 px beyond them)"))):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        if z is None:
            ax.axis("off")
            continue
        xtd.draw_xt(ax, z[win], fnum[win], xbin, local, lf)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("position along the tube (px)")
        ax.set_title(ttl, fontsize=11, color=INK)

    # --- 3a. position against time
    ax = fig.add_subplot(gs[2, 0:2])
    use_t = np.isfinite(t_us[t0:t1 + 1]).all()
    tx = (lambda t: t_us[t]) if use_t else (lambda t: fnum[t])
    speeds = {}
    for f in near:
        m, col, lab = _style(f)
        tr = range(f["start_index"], f["end_index"] + 1)
        pts = [(t, meas[id(f)][t]) for t in tr if meas[id(f)][t] is not None]
        ax.plot([tx(t) for t in tr], [pos(f, t) for t in tr], "--", color=col,
                lw=1.2, alpha=0.7)
        if pts:
            ax.plot([tx(t) for t, _ in pts], [v for _, v in pts], "-",
                    color=col, lw=2)
            for t, v in pts:
                ax.plot(tx(t), v, m, ms=8, mew=1.5,
                        mfc="white" if t in flagged else col,
                        mec=col if t in flagged else "white")
        v = f["v_px_per_frame"]
        if use_t and p.fps:
            v_txt, speeds[f["candidate_id"]] = (f"{v * p.fps / 1e6:+.1f} "
                                                "px/µs", abs(v))
        else:
            v_txt, speeds[f["candidate_id"]] = f"{v:+.1f} px/frame", abs(v)
        ax.plot([], [], m + "-", color=col, lw=2, ms=8, mec="white",
                label=f"front {f['candidate_id']} ({f.get('where') or '?'}): "
                      f"{v_txt}")
    ax.set_xlabel("time from trigger (µs)" if use_t else "frame")
    ax.set_ylabel("position along the tube (px)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ins = [f for f in near if f.get("where") == "inside"]
    out = [f for f in near if f.get("where") == "outside too"]
    if ins and out:
        r = (max(abs(f["v_px_per_frame"]) for f in ins)
             / max(abs(f["v_px_per_frame"]) for f in out))
        ttl = f"Inside front {'faster' if r >= 1 else 'slower'} than the " \
              f"outside one: x{r:.2f}"
    else:
        ttl = ("Measured position (markers) and fitted track (dashed)")
    ax.set_title(ttl, fontsize=11, color=INK)
    if any(t in flagged for f in near for t in meas[id(f)]
           if meas[id(f)][t] is not None):
        ax.text(0.01, 0.01, "hollow marker: flagged frame (trigger / flash)",
                transform=ax.transAxes, fontsize=8, color=MUTED)

    # --- 3b. profile across the tube at the front
    ax = fig.add_subplot(gs[2, 2:4])
    half = max(14.0, 0.5 * (tube.outer_px if np.isfinite(tube.outer_px)
                            else tube.wall_sep_px) + 8)
    rr = np.arange(-half, half + 0.01, 0.25)
    prof_t = sorted(set(np.linspace(t0, t1, min(8, len(track))).round()
                        .astype(int).tolist()))
    for k, t in enumerate(prof_t):
        m_ = meas[id(c)][t]
        s = s_start + (m_ if m_ is not None else pos(c, t))
        lo, hi = tube.bore_edges(s)
        cen = 0.5 * (lo + hi)
        ss = s + np.array([-1.5, -0.5, 0.5, 1.5])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.nanmean(tubemod._sample(maps[t], a, ss[None, :],
                                              cen + rr[:, None]), axis=1)
        ax.plot(prof, rr, "-", lw=1.6, color=_style(c)[1],
                alpha=0.35 + 0.6 * (k + 1) / len(prof_t))
    s_mid = s_start + pos(c, 0.5 * (t0 + t1))
    lo, hi = tube.bore_edges(s_mid)
    cen = 0.5 * (lo + hi)
    for wr, name in ((tube.rho_wall[0], "wall"), (tube.rho_wall[1], "wall")):
        ax.axhspan(wr - cen - 1.5, wr - cen + 1.5, color="#dddddd", zorder=0)
        ax.text(0.98, wr - cen, name, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8, color=MUTED)
    for e in (lo, hi):
        ax.axhline(e - cen, color=MUTED, lw=0.7, ls=":")
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.invert_yaxis()
    ax.set_xlim(-4, 8) if sign > 0 else ax.set_xlim(-8, 4)
    ax.set_xlabel("change / pixel noise (σ)")
    ax.set_ylabel("across the tube (px, 0 = bore centre)")
    ax.set_title(f"Profile across the tube at front {c['candidate_id']} "
                 + ("(each frame of its track" if len(prof_t) == len(track)
                    else f"({len(prof_t)} frames of its track")
                 + "; darker = later)", fontsize=11, color=INK)
    ax.text(0.01, 0.01, "taken at the measured position, 4 px along the tube",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    m, col, lab = _style(c)
    out_s = (f", outside the walls {c['outside_score']:.1f}"
             if c.get("outside_score") is not None else "")
    if c.get("tube_shift_px") is not None:
        out_s += f", tube moved up to {c['tube_shift_px']:.2f} px"
    fig.suptitle(f"{p.paths.name} — front {c['candidate_id']} in the tube: "
                 f"{lab}  (score {c['score']:.1f}{out_s}; frames "
                 f"{c['start_frame']}-{c['end_frame']})", fontsize=14,
                 color=INK, y=0.97)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return path


def make_sheets(p, files):
    """Draw one sheet per front in the tube; returns the paths written."""
    paths = p.paths
    if not os.path.exists(paths.activity):
        return []
    act = dict(np.load(paths.activity))
    if "tube_json" not in act or "tube_xt" not in act:
        return []
    tube = tubemod.Tube.from_dict(json.loads(str(act["tube_json"])))
    fronts = load_fronts(paths.tube_shocks)
    out = []
    for c in fronts:
        out.append(make_sheet(p, files, act, tube, fronts, c,
                              paths.tube_front_sheet(c["candidate_id"])))
    return out

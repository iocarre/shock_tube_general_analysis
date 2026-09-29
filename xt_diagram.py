"""
xt_diagram.py
=============

Space-time (x-t) diagram of a recording and a search for moving fronts in it.

The tube axis must be horizontal in the (rotated) frame -- use ``--rotate``.
For every frame the pooled SNR map (see ``change_common.activity_snr``) is
averaged over the tube height, column by column, over the measured pixels only.
Stacking these profiles over time gives the classic shock-tube x-t diagram:

  * a front spanning the tube height (a shock) keeps its full strength under
    the height average and draws a **straight slanted line**; the slope is its
    speed;
  * a small object (dust) is diluted by the average and leaves short specks;
  * a change of illumination hits every x at once: a **horizontal** band.

The shock search sums the x-t diagram along straight tracks x = x0 + v (t - t0)
for every speed in [--shock-vmin, --shock-vmax] (both directions) over up to
32 frames, and reports the tracks whose summed signal is a many-sigma outlier.
Summing along the track is what makes a faint front visible: over K frames the
noise grows as sqrt(K) but a real front grows as K.

To keep things cheap the diagram is stored with ``--xt-bin`` columns merged
(4 px by default), and track sums are built by doubling (1, 2, 4, ... frames),
so the search costs a few seconds per thousand frames.
"""

from __future__ import annotations

import numpy as np

import change_common as cc


# --------------------------------------------------------------------------- #
# Building the diagram
# --------------------------------------------------------------------------- #

def xt_row(snr, valid, xbin):
    """
    Height-averaged pooled SNR of one frame, with ``xbin`` columns merged.
    Bins with too few measured pixels are NaN.
    """
    h, w = snr.shape
    nb = w // xbin
    num = np.where(valid, snr, 0.0)[:, :nb * xbin].sum(0)
    cnt = valid[:, :nb * xbin].sum(0)
    num = num.reshape(nb, xbin).sum(1)
    cnt = cnt.reshape(nb, xbin).sum(1).astype(np.float64)
    out = np.full(nb, np.nan, dtype=np.float32)
    ok = cnt >= 0.3 * h * xbin
    out[ok] = num[ok] / cnt[ok]
    return out


def _robust_scale(a):
    """Per-column median and 1.4826 * MAD, NaN for columns never measured."""
    med = np.full(a.shape[1], np.nan)
    mad = np.full(a.shape[1], np.nan)
    ok = np.isfinite(a).any(0)
    if ok.any():
        med[ok] = np.nanmedian(a[:, ok], axis=0)
        mad[ok] = 1.4826 * np.nanmedian(np.abs(a[:, ok] - med[ok]), axis=0)
    return med, mad


def normalise(xt, bad_rows=(), clip=5.0, highpass=33):
    """
    Turn the raw diagram into a unit-noise one: each column is centred and
    scaled by its robust (median / MAD) spread over time, values are clipped to
    +/- ``clip`` sigma (a flash cannot dominate a track sum), ``bad_rows``
    (flagged frames) are zeroed, and a running mean over ``highpass`` frames is
    removed so that slow, stationary changes do not look like a slow front.
    Returns (z, column_scale); column_scale converts raw units to z units.
    """
    xt = np.asarray(xt, dtype=np.float64)
    med, mad = _robust_scale(xt)
    dead = ~np.isfinite(mad) | (mad <= 0)
    mad = np.where(dead, 1.0, mad)
    z = np.clip((xt - med) / mad, -clip, clip)
    z[:, dead] = 0.0
    z[~np.isfinite(z)] = 0.0
    bad = [r for r in bad_rows if 0 <= r < z.shape[0]]
    z[bad] = 0.0
    if highpass and z.shape[0] > highpass:
        half = highpass // 2
        c = np.cumsum(np.pad(z, ((half + 1, half), (0, 0)), mode="edge"), 0)
        z = z - (c[highpass:] - c[:-highpass]) / highpass
        z[bad] = 0.0
    col_scale = np.where(dead, 0.0, 1.0 / mad)
    return z.astype(np.float32), col_scale


# --------------------------------------------------------------------------- #
# Straight-track search
# --------------------------------------------------------------------------- #

def _shifted(S, dt, dx):
    """out[t, x] = S[t + dt, x + dx], zero where that falls outside."""
    T, W = S.shape
    out = np.zeros_like(S)
    if dt >= T or abs(dx) >= W:
        return out
    out[:T - dt, max(0, -dx):W - max(0, dx)] = S[dt:, max(0, dx):W + min(0, dx)]
    return out


def speed_grid(nbins, umin, umax, kmin=4, kmax=32):
    """
    Speeds (bins/frame) to test, each with its track length K (a power of two):
    K is the number of frames the front stays in view, capped to [kmin, kmax],
    and consecutive speeds differ by 1/K so no track drifts by more than half a
    bin from the tested line.
    """
    grid, u = [], umin
    while u <= umax:
        span = nbins / u
        if span < kmin:
            break
        K = int(min(max(2 ** int(np.floor(np.log2(span))), kmin), kmax))
        grid.append((u, K))
        u += 1.0 / K
    return grid


def search(z, umin, umax, kmin=4, kmax=32, chunk=4096):
    """
    Best straight-track score starting at every (frame, bin) of ``z``.

    score = |sum of z along the track| / sqrt(K), i.e. ~N(0, 1) for pure noise.
    Returns (score, speed_bins_per_frame, K) arrays shaped like ``z``, plus the
    number of speeds tested. Works in time chunks to bound memory.
    """
    T, W = z.shape
    grid = speed_grid(W, umin, umax, kmin, kmax)
    best = np.zeros((T, W), np.float32)
    best_v = np.zeros((T, W), np.float32)
    best_k = np.zeros((T, W), np.int16)
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        zc = z[a:min(T, b + kmax)]
        nkeep = b - a
        for u, K in grid:
            levels = int(np.log2(K))
            for v in (u, -u):
                S = zc.copy()
                for j in range(levels):
                    half = 2 ** j
                    S = S + _shifted(S, half, int(round(v * half)))
                S = np.abs(S[:nkeep]) / np.sqrt(K)
                m = S > best[a:b]
                best[a:b][m] = S[m]
                best_v[a:b][m] = v
                best_k[a:b][m] = K
    return best, best_v, best_k, len(grid)


def candidates(score, speed, klen, z, thresh, xbin, max_n=50, merge_px=64.0):
    """
    Tracks scoring above ``thresh``, strongest first, one per front: a track
    whose path stays within ``merge_px`` (on average, over the frames they
    share) of an already reported track is the same front seen with a slightly
    different start or speed, and is dropped.
    """
    T, W = score.shape
    s = score.copy()
    merge = merge_px / xbin
    out, kept = [], []                   # kept: (t0, t1, x0, v) in bins
    for _ in range(20000):
        if len(out) >= max_n:
            break
        i = int(np.argmax(s))
        t0, x0 = divmod(i, W)
        if s[t0, x0] < thresh:
            break
        v, K = float(speed[t0, x0]), int(klen[t0, x0])
        t1 = min(T - 1, t0 + K - 1)
        s[t0, max(0, x0 - 2):min(W, x0 + 3)] = 0.0
        dup = False
        for (a0, a1, ax, av) in kept:
            lo, hi = max(t0, a0), min(t1, a1)
            if lo > hi:
                continue
            tt = np.arange(lo, hi + 1)
            d = np.abs((x0 + v * (tt - t0)) - (ax + av * (tt - a0))).mean()
            if d <= merge:
                dup = True
                break
        if dup:
            continue
        kept.append((t0, t1, x0, v))
        # Sign of the summed signal: brighter or darker than the background.
        xs = np.clip(np.round(x0 + v * np.arange(K)).astype(int), 0, W - 1)
        ts = np.clip(t0 + np.arange(K), 0, T - 1)
        polarity = "brighter" if z[ts, xs].sum() > 0 else "darker"
        out.append({"t0": t0, "x0_px": (x0 + 0.5) * xbin,
                    "v_px_per_frame": v * xbin, "n_frames": K,
                    "score": float(score[t0, x0]), "polarity": polarity})
    return out


# --------------------------------------------------------------------------- #
# Sensitivity: the faintest front the search would still report
# --------------------------------------------------------------------------- #

def sensitivity(model, p, col_scale, xbin, thresh, width_px=2.0,
                n_frames=(4, 8, 32)):
    """
    Amplitude of the faintest full-height front the search would still report,
    for tracks of ``n_frames`` frames. The front is a Gaussian line
    (``width_px`` sigma) of amplitude ``a`` x the local pixel noise; its x-t
    response is computed through the same processing chain (pooling, band-pass,
    height average, binning, column scaling), averaged over positions across
    the measured field and over sub-bin phases.

    Returns {K: (a_min in pixel-noise units, a_min as % of the brightness)}.
    """
    h, w = model.bg.shape
    cols = np.nonzero(model.valid.any(0))[0]
    if cols.size == 0:
        return {}
    xs = np.linspace(cols[0] + 3 * width_px, cols[-1] - 3 * width_px, 9)
    responses = []
    x = np.arange(w)
    for xc0 in xs:
        for phase in np.linspace(0, xbin, 4, endpoint=False):
            xc = xc0 + phase
            prof = np.exp(-0.5 * ((x - xc) / width_px) ** 2).astype(np.float32)
            line = model.sigma * prof[None, :]
            snr = cc.activity_snr(model.bg + line, model.bg, model.sigma2,
                                  p.smooth, p.detrend_band)
            row = xt_row(snr, model.valid, xbin)
            b = int(xc // xbin)
            if 0 <= b < row.size and np.isfinite(row[b]) and col_scale[b] > 0:
                responses.append(max(np.nanmax(row[max(0, b - 1):b + 2]), 0)
                                 * col_scale[b])
    if not responses:
        return {}
    r = float(np.median(responses))
    if r <= 0:
        return {}
    rel = float(np.median((model.sigma / np.maximum(model.bg, 1e-6))
                          [model.valid]))
    return {K: (thresh / (r * np.sqrt(K)), 100.0 * rel * thresh
                / (r * np.sqrt(K))) for K in n_frames}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def display_rows(z, max_rows=2000):
    """Rows to show: all of them, or blocks reduced to their strongest value
    (sign kept) when the recording is long, so a front is never averaged away."""
    T = z.shape[0]
    if T <= max_rows:
        return z, 1
    f = int(np.ceil(T / max_rows))
    n = T // f
    blk = z[:n * f].reshape(n, f, -1)
    idx = np.abs(blk).argmax(1)
    return np.take_along_axis(blk, idx[:, None, :], 1)[:, 0, :], f


def draw_xt(ax, z, frames, xbin, cands=(), flagged=(), vlim=4.0):
    """Draw the x-t diagram on ``ax`` (x horizontal, frame number downwards),
    with flagged frames shaded grey and candidate tracks bracketed by two
    dashed yellow guides."""
    shown, f = display_rows(z)
    T, W = z.shape
    f0, f1 = frames[0], frames[-1]
    ax.imshow(shown, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto",
              interpolation="nearest",
              extent=[0, W * xbin, f1 + 0.5, f0 - 0.5])
    for r in flagged:
        ax.axhspan(frames[r] - 0.5, frames[r] + 0.5, color="0.5", alpha=0.35,
                   lw=0)
    for c in cands:
        # Two guides on either side of the track, leaving the signal visible.
        t = np.array([c["t0"], min(T - 1, c["t0"] + c["n_frames"] - 1)])
        x = c["x0_px"] + c["v_px_per_frame"] * (t - c["t0"])
        for off in (-4 * xbin, 4 * xbin):
            ax.plot(x + off, [frames[i] for i in t], color="black", lw=2.0,
                    alpha=0.7)
            ax.plot(x + off, [frames[i] for i in t], color="yellow", lw=1.0,
                    ls="--")
    ax.set_xlabel("x along the tube (px)")
    ax.set_ylabel("frame")
    ax.set_xlim(0, W * xbin)

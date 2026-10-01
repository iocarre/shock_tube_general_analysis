#!/usr/bin/env python3
"""
tube.py  --  find a straight micro-tube in the field of view.
=============================================================

Some shots are taken in open air (to test the shadowgraph); most have a
straight micro-tube in the field, whose position, angle, length and diameter
vary. In a shadowgraph the tube always looks the same: two thin, straight,
parallel **dark lines** (its walls) with a **bright strip** between them (the
bore). This module finds that pattern in the background image of a shot:

  1. thin-dark-line map: pixels darker than their lit neighbourhood (a broad
     dark object -- a rod, a block, an unlit area -- is not a thin line);
  2. straight-line search (Hough) over every angle;
  3. pairs of parallel lines 3-40 px apart with a bright bore between them,
     scored by the length over which both walls *and* the bright bore are seen
     continuously (scattered noise never forms a long continuous run);
  4. refinement of the angle (0.1 deg) and of the wall / bore edges (sub-pixel)
     from the profile across the tube.

Geometry is expressed along the tube: ``s`` (position along the axis) and
``rho`` (distance across it), with ``s = x cos(a) + y sin(a)`` and
``rho = -x sin(a) + y cos(a)`` for the axis angle ``a`` (pixel coordinates of
the frame as the pipeline sees it, i.e. after ``--rotate``).

Run on its own to check the detection before analysing anything:

    python3 tube.py Donnerstag --rotate 90       # every shot below, any depth
    python3 tube.py Donnerstag/t400/Test1/26284_1_50 --rotate 90

Each shot gets ``NAME_analysis/NAME_tube.json`` (the geometry, or why no tube
was found) and ``NAME_analysis/NAME_tube.png`` (bore drawn on the background).
A folder of shots also gets ``CAMPAIGN_tubes.png``, one thumbnail per shot.

Detection (``detect_activity.py``) runs the same detection on its background
model, then uses :class:`BoreSampler` to turn every frame into a profile along
the bore -- and along a band just outside the walls -- to search for fronts
inside the tube and to tell them from waves outside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np

import change_common as cc

MIN_GAP, MAX_GAP = 3, 40        # wall-to-wall distance searched (px)
MIN_RUN = 60                    # shortest continuous visible stretch (px)
MIN_CONTRAST = 20.0             # scene contrast / pixel noise below: unusable


@dataclass
class Tube:
    """A straight tube: axis angle, wall and bore edges across it (``rho``),
    and the stretch along it (``s``) where the bore is seen. Each bore edge is
    a straight line of its own (``rho_bore`` at the middle of the stretch,
    ``bore_slope`` along it), so a tapered bore is followed too."""
    angle_deg: float            # axis direction, deg (0 = along +x, + = down)
    rho_wall: tuple             # wall centre lines (rho), sub-pixel
    rho_bore: tuple             # bore edges (rho) at the middle of s_range
    rho_outer: tuple            # outer edges (rho); NaN if not measurable
    segments: list              # visible stretches along the axis [(s0, s1)]
    score: float                # continuous length over which it was found
    bore_slope: tuple = (0.0, 0.0)   # d(rho)/ds of each bore edge

    @property
    def s_mid(self):
        return 0.5 * (self.s_range[0] + self.s_range[1])

    def bore_edges(self, s):
        """Bore edges (rho_lo, rho_hi) at position(s) ``s`` along the tube."""
        ds = np.asarray(s, np.float64) - self.s_mid
        return (self.rho_bore[0] + self.bore_slope[0] * ds,
                self.rho_bore[1] + self.bore_slope[1] * ds)

    def bore_at(self, s):
        lo, hi = self.bore_edges(s)
        return hi - lo

    @property
    def s_range(self):
        """From the start of the first visible stretch to the end of the last."""
        return self.segments[0][0], self.segments[-1][1]

    @property
    def seen_px(self):
        return _seen(self.segments)

    @property
    def bore_px(self):
        return self.rho_bore[1] - self.rho_bore[0]

    @property
    def wall_sep_px(self):
        return self.rho_wall[1] - self.rho_wall[0]

    @property
    def outer_px(self):
        return self.rho_outer[1] - self.rho_outer[0]

    def coords(self, shape):
        """(s, rho) of every pixel of a frame of this ``shape``."""
        return _coords(shape, np.deg2rad(self.angle_deg))

    def bore_mask(self, shape):
        """Pixels whose centre lies inside the bore, where the bore is seen."""
        s, rho = self.coords(shape)
        along = np.zeros(shape, bool)
        for lo, hi in self.segments:
            along |= (s >= lo) & (s <= hi)
        lo, hi = self.bore_edges(s)
        return (rho > lo) & (rho < hi) & along

    def point(self, s, rho):
        """Pixel (x, y) of a point given along / across the tube."""
        a = np.deg2rad(self.angle_deg)
        return (s * np.cos(a) - rho * np.sin(a),
                s * np.sin(a) + rho * np.cos(a))

    def to_dict(self):
        d = asdict(self)
        s0, s1 = self.s_range
        d.update(found=True, bore_px=self.bore_px,
                 bore_px_ends=(self.bore_at(s0), self.bore_at(s1)),
                 wall_sep_px=self.wall_sep_px, outer_px=self.outer_px,
                 s_range=self.s_range, seen_px=self.seen_px)
        for key, s_ in (("axis_start_xy", s0), ("axis_end_xy", s1)):
            lo, hi = self.bore_edges(s_)
            d[key] = self.point(s_, 0.5 * (lo + hi))
        return _round(d)

    @classmethod
    def from_dict(cls, d):
        if not d or not d.get("found"):
            return None
        return cls(d["angle_deg"], tuple(d["rho_wall"]), tuple(d["rho_bore"]),
                   tuple(np.nan if v is None else v for v in d["rho_outer"]),
                   [tuple(g) for g in d["segments"]], d["score"],
                   tuple(d.get("bore_slope", (0.0, 0.0))))


def _round(x):
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 3)
    return x


def _coords(shape, a):
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float64)
    return xx * np.cos(a) + yy * np.sin(a), -xx * np.sin(a) + yy * np.cos(a)


def _runs(ok, close=6):
    """Runs (first, last) of True in a 1-D mask, gaps <= ``close`` closed."""
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    runs = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - runs[-1][1] > close + 1:
            runs.append([i, i])
        else:
            runs[-1][1] = i
    return [tuple(r) for r in runs]


def _longest(runs):
    return max(runs, key=lambda r: r[1] - r[0]) if runs else (0, 0)


def _subpix(p, i):
    """Parabolic refinement of an extremum of ``p`` at index ``i``."""
    if i <= 0 or i >= len(p) - 1:
        return float(i)
    a, b, c = p[i - 1], p[i], p[i + 1]
    d = a - 2 * b + c
    return i + (0.5 * (a - c) / d if d else 0.0)


def _cross(p, i, step, level):
    """Sub-pixel index where ``p`` first crosses ``level``, walking from ``i``."""
    while 0 <= i + step < len(p) and (p[i] - level) * (p[i + step] - level) > 0:
        i += step
    j = i + step
    if not 0 <= j < len(p) or p[j] == p[i]:
        return np.nan
    return i + (level - p[i]) / (p[j] - p[i]) * step


def line_map(b):
    """Thin dark lines: darker than a lit 9x9 neighbourhood by > 15 %."""
    loc = cc.box2d_mean(b, 9)
    return ((loc - b) > 0.15) & (loc > 0.25)


def _sample(b, a, s, rho):
    """Bilinear sample of ``b`` at (s, rho) points; NaN outside the frame."""
    x = s * np.cos(a) - rho * np.sin(a)
    y = s * np.sin(a) + rho * np.cos(a)
    h, w = b.shape
    ok = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 2)
    fx, fy = x - x0, y - y0
    v = (b[y0, x0] * (1 - fx) * (1 - fy) + b[y0, x0 + 1] * fx * (1 - fy)
         + b[y0 + 1, x0] * (1 - fx) * fy + b[y0 + 1, x0 + 1] * fx * fy)
    return np.where(ok, v, np.nan)


def _profile(b, a, s_lo, s_hi, centre, half, step=0.25):
    """
    Brightness across the tube every ``step`` px of rho: median along the
    stretch [s_lo, s_hi]. Every rho is sampled at the same positions along
    the tube, so brightness changes along it (a dark object above part of the
    tube) cannot bias one rho against another.
    """
    rho = centre + np.arange(-half, half + step / 2, step)
    s = np.arange(s_lo, s_hi + 0.5, 1.0)
    v = _sample(b, a, s[None, :], rho[:, None])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.nanmedian(v, axis=1)
    return prof, rho


def _pair_run(b, ridge_s, ridge_r, a, r1, r2, shape):
    """Continuous stretch (s0, s1) over which both walls (lines at rho r1, r2)
    and a bright bore between them are seen."""
    s_min, s_max = _extent_along(shape, a)
    n = int(s_max - s_min) + 1
    cov = []
    for r in (r1, r2):
        on = np.abs(ridge_r - r) <= 1.0
        c = np.zeros(n, bool)
        c[np.clip((ridge_s[on] - s_min).astype(int), 0, n - 1)] = True
        cov.append(c)
    # bore brighter than the walls, sampled on the lines (nearest pixel)
    ss = s_min + np.arange(n)
    def sample(r):
        x = np.rint(ss * np.cos(a) - r * np.sin(a)).astype(int)
        y = np.rint(ss * np.sin(a) + r * np.cos(a)).astype(int)
        ok = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
        v = np.full(n, np.nan)
        v[ok] = b[y[ok], x[ok]]
        return v
    mid = sample(0.5 * (r1 + r2))
    walls = np.fmax(sample(r1), sample(r2))
    with np.errstate(invalid="ignore"):
        bright = (mid > 0.25) & (mid > 1.3 * walls)
    i0, i1 = _longest(_runs(cov[0] & cov[1] & bright))
    return s_min + i0, s_min + i1


def _visible(b, a, rho_lo, rho_hi, slack=0.0, min_seg=15):
    """
    Stretches [(s0, s1), ...] along the tube where the bore -- between
    ``rho_lo`` and ``rho_hi`` -- is seen: lit and brighter than both walls
    just outside it. A tube can be hidden in places (a holder in front of
    it), so every stretch of ``min_seg`` px or more is kept. ``slack`` widens
    the search for the walls (for rough wall positions); gaps of a few px (a
    speck on the window) are closed.
    """
    import warnings
    warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
    smin, smax = _extent_along(b.shape, a)
    s = np.arange(smin, smax + 1.0)[None, :]
    mid, gap = 0.5 * (rho_lo + rho_hi), rho_hi - rho_lo
    inner = np.arange(-0.25, 0.26, 0.25) * gap + mid
    bore = np.nanmax(_sample(b, a, s, inner[:, None]), axis=0) \
        if gap > 3 else _sample(b, a, s, np.array([[mid]]))[0]
    wall = []
    for edge, d in ((rho_lo, -1), (rho_hi, 1)):
        rr = edge + d * np.arange(0.5, 2.6 + slack, 0.5)
        wall.append(np.nanmin(_sample(b, a, s, rr[:, None]), axis=0))
    with np.errstate(invalid="ignore"):
        ok = (bore > 0.25) & (bore > 1.5 * np.fmax(wall[0], wall[1]))
    return [(float(smin + i0), float(smin + i1 + 1)) for i0, i1 in _runs(ok)
            if i1 + 1 - i0 >= min_seg]


def _seen(segs):
    return sum(hi - lo for lo, hi in segs)


def _extent_along(shape, a):
    h, w = shape
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    s = corners[:, 0] * np.cos(a) + corners[:, 1] * np.sin(a)
    return np.floor(s.min()), np.ceil(s.max())


def detect_tube(bg, sigma=None):
    """
    Look for a straight tube in a background image.

    Returns ``(tube, reason)``: a :class:`Tube` and ``""``, or ``None`` and a
    sentence saying why no tube was found. ``sigma`` (per-pixel noise) lets an
    unusable image -- unlit, or noise only -- be recognised as such.
    """
    bg = np.asarray(bg, np.float64)
    lo, hi = np.percentile(bg, [1, 99])
    if sigma is not None:
        contrast = (hi - lo) / max(float(np.median(sigma)), 1e-9)
        if contrast < MIN_CONTRAST:
            return None, (f"image unusable: scene contrast is only "
                          f"{contrast:.1f}x the pixel noise (unlit or empty?)")
    b = (bg - lo) / max(hi - lo, 1e-9)
    ridge = line_map(b)
    ry, rx = np.nonzero(ridge)
    if rx.size < 2 * MIN_RUN:
        return None, "no thin dark lines in the image"
    shape = b.shape

    # Hough: for every axis angle, count line pixels per rho (1 px bins) and
    # keep pairs of local maxima a plausible wall distance apart.
    pairs = []
    for deg in np.arange(-89.5, 90.01, 0.5):
        a = np.deg2rad(deg)
        rho = -rx * np.sin(a) + ry * np.cos(a)
        r0 = np.floor(rho.min())
        cnt = np.bincount((rho - r0).astype(int))
        pk = [i for i in range(len(cnt)) if cnt[i] >= MIN_RUN // 2
              and cnt[i] >= cnt[max(i - 1, 0)]
              and cnt[i] >= cnt[min(i + 1, len(cnt) - 1)]]
        pk = sorted(pk, key=lambda i: -cnt[i])[:8]
        for i in pk:
            for j in pk:
                if MIN_GAP <= j - i <= MAX_GAP:
                    pairs.append((min(cnt[i], cnt[j]), deg, r0 + i + 0.5,
                                  r0 + j + 0.5))
    pairs.sort(reverse=True)

    best = None
    for _, deg, r1, r2 in pairs[:60]:
        a = np.deg2rad(deg)
        rs = rx * np.cos(a) + ry * np.sin(a)
        rr = -rx * np.sin(a) + ry * np.cos(a)
        s0, s1 = _pair_run(b, rs, rr, a, r1, r2, shape)
        if best is None or s1 - s0 > best[0]:
            best = (s1 - s0, deg, r1, r2, s0, s1)
    if best is None or best[0] < MIN_RUN:
        got = f" (longest: {best[0]:.0f} px)" if best else ""
        return None, (f"no pair of parallel walls with a bright bore seen over "
                      f"{MIN_RUN} px or more{got}")

    _, deg, r1, r2, _, _ = best
    cen, gap = 0.5 * (r1 + r2), r2 - r1
    a = np.deg2rad(deg)
    segs = _visible(b, a, cen - gap / 2, cen + gap / 2, slack=1.0)
    if _seen(segs) < MIN_RUN:
        return None, (f"bore seen over only {_seen(segs):.0f} px "
                      f"(need {MIN_RUN})")
    s0, s1 = segs[0][0], segs[-1][1]

    # Refine the angle: the wall/bore profile across the tube is sharpest
    # (largest variance) when taken exactly along the tube. Rotate about the
    # middle of the visible stretch, so the tube stays at the same rho there.
    sm = 0.5 * (s0 + s1)
    xm, ym = sm * np.cos(a) - cen * np.sin(a), sm * np.sin(a) + cen * np.cos(a)
    half_len = 0.5 * (s1 - s0)
    # Only the visible stretches enter the profiles (hidden parts are flat).
    def cross(ad, r_c, s_c, half):
        sh = s_c - sm
        profs = [_profile(b, ad, lo + sh, hi + sh, r_c, half)
                 for lo, hi in segs]
        w = np.array([hi - lo for lo, hi in segs])[:, None]
        return (np.nansum(np.array([p for p, _ in profs]) * w, 0) / w.sum(),
                profs[0][1])
    best_v = None
    for d in np.arange(deg - 1.0, deg + 1.001, 0.05):
        ad = np.deg2rad(d)
        s_c = xm * np.cos(ad) + ym * np.sin(ad)
        r_c = -xm * np.sin(ad) + ym * np.cos(ad)
        prof, _ = cross(ad, r_c, s_c, gap / 2 + 2)
        v = np.nanvar(prof)
        if best_v is None or v > best_v[0]:
            best_v = (v, d, s_c, r_c)
    _, deg, sm, cen = best_v
    a = np.deg2rad(deg)
    half = gap / 2 + 9
    prof, rho_ax = cross(a, cen, sm, half)
    if not np.isfinite(prof).all():
        return None, "cross profile runs out of the frame"
    step = rho_ax[1] - rho_ax[0]
    at = lambda i: rho_ax[0] + step * i

    # Bore: the bright plateau in the middle; its edges at half level between
    # it and the darkest point of each wall. Walls: darkest point within 4 px
    # beyond each bore edge. Outer edges: half level between the wall and the
    # brightness 4-7 px beyond it (NaN where a dark object lies against it).
    core = np.flatnonzero(np.abs(rho_ax - cen) <= gap / 2)
    top = int(core[np.argmax(prof[core])])
    k4, k7 = int(round(4 / step)), int(round(7 / step))
    bore, walls, outer = [], [], []
    for d in (-1, 1):
        wall_zone = np.flatnonzero((d * (rho_ax - rho_ax[top]) > 0)
                                   & (np.abs(rho_ax - cen) <= gap / 2 + 3))
        if wall_zone.size == 0:
            return None, "walls could not be resolved in the cross profile"
        lvl = 0.5 * (prof[top] + prof[wall_zone].min())
        e = _cross(prof, top, d, lvl)
        if not np.isfinite(e):
            return None, "bore edges could not be resolved in the cross profile"
        ie = int(round(e))
        zone = np.arange(ie, ie + d * (k4 + 1), d)
        zone = zone[(zone >= 0) & (zone < len(prof))]
        iw = int(zone[np.argmin(prof[zone])])
        win = prof[max(iw - k7, 0):max(iw - k4, 0)] if d < 0 else \
            prof[iw + k4:iw + k7]
        out = np.median(win) if win.size else np.nan
        o = (_cross(prof, iw, d, 0.5 * (out + prof[iw]))
             if np.isfinite(out) and out > prof[iw] + 0.1 else np.nan)
        bore.append(at(e))
        walls.append(at(_subpix(prof, iw)))
        outer.append(at(o) if np.isfinite(o) else np.nan)
    if bore[1] - bore[0] < 1.0:
        return None, "bore narrower than a pixel"

    segs = _visible(b, a, bore[0], bore[1])
    if _seen(segs) < MIN_RUN:
        return None, (f"bore seen over only {_seen(segs):.0f} px "
                      f"(need {MIN_RUN})")
    tube = Tube(float(round(deg, 2)), tuple(walls), tuple(bore),
                tuple(outer), [tuple(g) for g in segs], float(best[0]))
    _fit_edges(b, a, tube, gap)
    return tube, ""


def _edge_at(prof, rho_ax, cen, width, gap):
    """Bore edges in one cross profile: half level between the bore's
    brightest point and the darkest point of each wall."""
    step = rho_ax[1] - rho_ax[0]
    core = np.flatnonzero(np.abs(rho_ax - cen) <= max(width / 2 - 0.5, 0.5))
    if core.size == 0 or not np.isfinite(prof).all():
        return None
    top = int(core[np.argmax(prof[core])])
    out = []
    for d in (-1, 1):
        zone = np.flatnonzero((d * (rho_ax - rho_ax[top]) > 0)
                              & (np.abs(rho_ax - cen) <= gap / 2 + 3))
        if zone.size == 0:
            return None
        e = _cross(prof, top, d, 0.5 * (prof[top] + prof[zone].min()))
        if not np.isfinite(e):
            return None
        out.append(rho_ax[0] + step * e)
    return out


def _fit_edges(b, a, tube, gap, block=24):
    """
    Measure the bore edges in ``block``-px pieces along the visible stretches
    and fit a straight line to each edge (one re-fit without the pieces more
    than 0.5 px -- or 3 robust sigma -- off, so a local defect cannot tilt
    it). Updates ``tube.rho_bore`` / ``bore_slope``.
    """
    pts = []
    for lo, hi in tube.segments:
        n = max(1, int((hi - lo) // block))
        edges = np.linspace(lo, hi, n + 1)
        for s0, s1 in zip(edges[:-1], edges[1:]):
            sm = 0.5 * (s0 + s1)
            e_lo, e_hi = tube.bore_edges(sm)
            cen = 0.5 * (e_lo + e_hi)
            prof, rho_ax = _profile(b, a, s0, s1, cen, gap / 2 + 4)
            e = _edge_at(prof, rho_ax, cen, e_hi - e_lo, gap)
            if e is not None:
                pts.append((sm, e[0], e[1]))
    if len(pts) < 3:
        return
    p = np.array(pts)
    ds = p[:, 0] - tube.s_mid
    fits = []
    for k in (1, 2):
        c = np.polyfit(ds, p[:, k], 1)
        res = np.abs(np.polyval(c, ds) - p[:, k])
        ok = res <= max(0.5, 3 * 1.4826 * np.median(res))
        if ok.sum() >= 3:
            c = np.polyfit(ds[ok], p[ok, k], 1)
        fits.append(c)
    tube.rho_bore = (float(fits[0][1]), float(fits[1][1]))
    tube.bore_slope = (float(fits[0][0]), float(fits[1][0]))


# --------------------------------------------------------------------------- #
# The bore as a 1-D signal: one x-t diagram row per frame
# --------------------------------------------------------------------------- #

class BoreSampler:
    """
    Turns a frame into a profile along the bore: the change of every bore
    pixel in units of its own noise, ``(frame - bg) / sigma``, minus the
    frame's median over the bore (a flash or exposure change brightens the
    whole bore at once), averaged in ``xbin``-px bins along the tube and
    multiplied by sqrt(pixels) so that a quiet bin is ~N(0, 1).

    No spatial smoothing is applied: the bore is only a few px wide, and any
    pooling box would mix in the walls and the flow outside the tube.
    Bin ``k`` covers ``s_start + [k, k+1) * xbin`` along the tube; bins where
    the bore is hidden are NaN.
    """

    def __init__(self, tube, shape, xbin, mask=None):
        self.tube, self.xbin = tube, int(xbin)
        s, _ = tube.coords(shape)
        mask = tube.bore_mask(shape) if mask is None else mask
        self.idx = np.flatnonzero(mask)
        self.s_start = float(tube.s_range[0])
        k = np.floor((s.ravel()[self.idx] - self.s_start) / self.xbin)
        keep = (k >= 0) & (k <= (tube.s_range[1] - self.s_start) // self.xbin)
        self.idx, self.k = self.idx[keep], k[keep].astype(int)
        self.nbins = int((tube.s_range[1] - self.s_start) // self.xbin) + 1
        self.count = np.bincount(self.k, None, self.nbins)
        self.seen = self.count > 0

    @classmethod
    def sleeve(cls, tube, shape, xbin, bg, gap=(2.0, 8.0), lit=0.3):
        """
        The same sampling for a band just *outside* the tube: ``gap`` px
        beyond each wall's outer side, on both sides, lit pixels only (a dark
        holder against the tube is left out), over the tube's whole length.
        A wave travelling outside the tube shows here as well as in the bore
        (the bore is seen through it); a front inside the tube does not.
        """
        s, rho = tube.coords(shape)
        out_lo = tube.rho_outer[0] if np.isfinite(tube.rho_outer[0]) \
            else tube.rho_wall[0] - 2.0
        out_hi = tube.rho_outer[1] if np.isfinite(tube.rho_outer[1]) \
            else tube.rho_wall[1] + 2.0
        band = (((rho <= out_lo - gap[0]) & (rho >= out_lo - gap[1]))
                | ((rho >= out_hi + gap[0]) & (rho <= out_hi + gap[1])))
        ref = np.percentile(bg, 99)
        return cls(tube, shape, xbin, band & (bg > lit * ref))

    def row(self, frame, bg, sigma):
        z = (frame.ravel()[self.idx] - bg.ravel()[self.idx]) \
            / sigma.ravel()[self.idx]
        z = z - np.median(z)
        out = np.full(self.nbins, np.nan, np.float32)
        out[self.seen] = (np.bincount(self.k, z, self.nbins)[self.seen]
                          / np.sqrt(self.count[self.seen]))
        return out

    def s_of(self, pos_px):
        """Position along the tube (s) of a distance from the bin origin."""
        return self.s_start + pos_px

    def xy(self, pos_px):
        """Pixel (x, y) on the bore axis at a distance from the bin origin."""
        s = self.s_of(pos_px)
        lo, hi = self.tube.bore_edges(s)
        return self.tube.point(s, 0.5 * (lo + hi))

    def sensitivity(self, bg, sigma, col_scale, thresh, width_px=2.0,
                    n_frames=(4, 8, 32)):
        """
        Amplitude of the faintest front filling the bore that the search
        would still report, for tracks of ``n_frames`` frames, as in
        ``xt_diagram.sensitivity``: a Gaussian line across the bore
        (``width_px`` sigma along it) of amplitude ``a`` x the pixel noise,
        put through this same sampling at positions along the bore.
        Returns {K: (a_min in pixel-noise units, a_min as % of brightness)}.
        """
        bins = np.flatnonzero(self.seen & (col_scale[:self.nbins] > 0))
        if bins.size == 0:
            return {}
        s, _ = self.tube.coords(bg.shape)
        resp = []
        for b in bins[np.linspace(0, bins.size - 1, 12).astype(int)]:
            for phase in np.linspace(0, self.xbin, 4, endpoint=False):
                sc = self.s_start + b * self.xbin + phase
                line = sigma * np.exp(-0.5 * ((s - sc) / width_px) ** 2)
                row = self.row(bg + line, bg, sigma)
                lo, hi = max(0, b - 1), min(self.nbins, b + 2)
                v = np.nanmax(row[lo:hi]) if np.isfinite(row[lo:hi]).any() \
                    else 0.0
                resp.append(max(v, 0.0) * col_scale[b])
        r = float(np.median(resp))
        if r <= 0:
            return {}
        rel = float(np.median(sigma.ravel()[self.idx]
                              / np.maximum(bg.ravel()[self.idx], 1e-6)))
        return {K: (thresh / (r * np.sqrt(K)),
                    100.0 * rel * thresh / (r * np.sqrt(K))) for K in n_frames}


class TubeShift:
    """
    How far the tube moved across its axis in a frame (px, + towards larger
    rho): a rigid shift ``d`` changes the image around the tube by about
    ``-d * dbg/drho``, so ``d`` is the least-squares fit of that to the
    change in a band covering the tube and its walls (a uniform brightness
    change is fitted out). A tube shaken by the shot moves its walls across
    the bore's pixels, and the bore's x-t diagram then shows "fronts" that
    are only the walls moving.
    """

    def __init__(self, tube, bg):
        s, rho = tube.coords(bg.shape)
        lo, hi = tube.bore_edges(s)
        cen = 0.5 * (lo + hi)
        width = tube.outer_px if np.isfinite(tube.outer_px) \
            else tube.wall_sep_px
        along = np.zeros(bg.shape, bool)
        for s0, s1 in tube.segments:
            along |= (s >= s0) & (s <= s1)
        band = (np.abs(rho - cen) <= 0.5 * width + 4) & along
        a = np.deg2rad(tube.angle_deg)
        gy, gx = np.gradient(bg.astype(np.float64))
        g = (-np.sin(a) * gx + np.cos(a) * gy)[band]
        self.idx = np.flatnonzero(band)
        self.g = g - g.mean()
        self.gg = float((self.g * self.g).sum()) or 1.0

    def shift(self, frame, bg):
        d = frame.ravel()[self.idx] - bg.ravel()[self.idx]
        return float(-((d - d.mean()) * self.g).sum() / self.gg)


# --------------------------------------------------------------------------- #
# Files: geometry JSON and check pictures
# --------------------------------------------------------------------------- #

def background(input_dir, pattern="*.tif", rotate=0, n_sample=24):
    """Median background and per-pixel noise of a shot (uniform sample)."""
    files = cc.list_frames(input_dir, pattern)
    stack = np.stack([cc.load_frame(files[i], rotate=rotate)
                      for i in cc.sample_indices(len(files), n_sample)])
    return cc.estimate_bg_noise(stack)


def save(path, tube, reason, extra=None):
    d = tube.to_dict() if tube else {"found": False, "reason": reason}
    d.update(extra or {})
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)


def load(path):
    """The :class:`Tube` saved at ``path`` (None if absent or no tube)."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return Tube.from_dict(json.load(fh))


def _bore_text(tube):
    """'4.90 px', or 'tapered 7.9 -> 4.8 px' (along s) when the ends differ."""
    w0, w1 = tube.bore_at(tube.s_range[0]), tube.bore_at(tube.s_range[1])
    if abs(w1 - w0) >= 0.5:
        return f"tapered {w0:.1f} -> {w1:.1f} px"
    return f"{tube.bore_px:.2f} px"


def draw(ax, bg, tube, title, reason=""):
    """Background with the bore edges (orange) and the ends of each visible
    stretch (blue) drawn on it."""
    lo, hi = np.percentile(bg, [1, 99.5])
    ax.imshow(bg, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if tube is None:
        import textwrap
        ax.set_title(f"{title}\n" + textwrap.fill(f"no tube: {reason}", 60),
                     fontsize=8, color="#555")
        return
    for s0, s1 in tube.segments:
        (a0, b0), (a1, b1) = tube.bore_edges(s0), tube.bore_edges(s1)
        for r0, r1 in ((a0, a1), (b0, b1)):
            (xa, ya), (xb, yb) = tube.point(s0, r0), tube.point(s1, r1)
            ax.plot([xa, xb], [ya, yb], "-", color="#eb6834", lw=1.0)
        for s, (lo, hi) in ((s0, (a0, b0)), (s1, (a1, b1))):
            (xa, ya), (xb, yb) = tube.point(s, lo - 5), tube.point(s, hi + 5)
            ax.plot([xa, xb], [ya, yb], "-", color="#2a78d6", lw=1.2)
    ax.set_xlim(-0.5, bg.shape[1] - 0.5)
    ax.set_ylim(bg.shape[0] - 0.5, -0.5)
    n = len(tube.segments)
    ax.set_title(f"{title}\nbore {_bore_text(tube)}, {tube.angle_deg:+.2f}"
                 f" deg, seen over {tube.seen_px:.0f} px"
                 + (f" in {n} stretches" if n > 1 else ""), fontsize=8)


def describe(tube, reason=""):
    """One line saying what was found."""
    if tube is None:
        return f"no tube ({reason})"
    return (f"bore {_bore_text(tube)}, {tube.angle_deg:+.2f} deg, seen over "
            f"{tube.seen_px:.0f} px in {len(tube.segments)} stretch(es)")


def write(paths, bg, tube, reason, extra=None):
    """Write ``NAME_tube.json`` and ``NAME_tube.png`` of a shot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(paths.dir, exist_ok=True)
    save(paths.tube, tube, reason, extra)
    h, w = bg.shape
    fig, ax = plt.subplots(figsize=(min(14, 2 + w / 30), 1.2 + h / 30))
    draw(ax, bg, tube, paths.name, reason)
    fig.tight_layout()
    fig.savefig(paths.tube_png, dpi=100)
    plt.close(fig)


def process(shot, p):
    """Detect the tube of one shot and write its JSON and picture."""
    paths = cc.shot_paths(shot)
    bg, sigma = background(shot, p.pattern, p.rotate)
    tube, reason = detect_tube(bg, sigma)
    write(paths, bg, tube, reason, {"rotate": p.rotate})
    print(f"[tube] {paths.name}: {describe(tube, reason)}", file=sys.stderr)
    return bg, tube, reason


def contact_sheet(campaign_dir, results):
    """One thumbnail per shot, to check the detections at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n, cols = len(results), 4
    rows = int(np.ceil(n / cols))
    h, w = results[0][1].shape
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols,
                                                 rows * (1.2 + 4.6 * h / w)))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, (name, bg, tube, reason) in zip(np.ravel(axs), results):
        ax.axis("on")
        draw(ax, bg, tube, name, reason)
    fig.tight_layout()
    path = os.path.join(campaign_dir, os.path.basename(
        os.path.normpath(campaign_dir)) + "_tubes.png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Find a straight micro-tube in each shot and draw it.")
    ap.add_argument("input_dir", help="A shot folder, or a folder of shot "
                                      "folders nested at any depth.")
    ap.add_argument("--pattern", default="*.tif", help="Glob for frames.")
    ap.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270),
                    help="Rotate frames clockwise, as for the analysis "
                         "(degrees).")
    p = ap.parse_args(argv)
    shots, batch = cc.find_shots(p.input_dir, p.pattern)
    results = []

    def one(shot):
        bg, tube, reason = process(shot, p)
        results.append((os.path.relpath(shot, p.input_dir) if batch
                        else os.path.basename(shot), bg, tube, reason))

    code = cc.run_shots("tube", shots, batch, one)
    if batch and results:
        print(f"[tube] wrote {contact_sheet(p.input_dir, results)}",
              file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

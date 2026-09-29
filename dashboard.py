#!/usr/bin/env python3
"""
dashboard.py  --  self-contained HTML dashboards of the analysis results.
=========================================================================

Two pages, both single files with every image and number embedded (no movies),
so they can be opened offline, moved or e-mailed on their own:

  * **shot dashboard** ``NAME_analysis/NAME_dashboard.html``: the verdict, the
    camera and analysis settings, an interactive x-t diagram linked to the
    activity and brightness curves (drag to zoom, click an event), the events
    table with each event's images and curve, the strongest moments below the
    thresholds, and the shock-search candidates;
  * **campaign dashboard** ``CAMPAIGN/CAMPAIGN_dashboard.html``: one row per
    shot, automatic data checks (duplicate recordings, incomplete exports,
    missing metadata, shots not analysed yet) and plots comparing the shots
    (sensitivity, noise, brightness, events, activity around the trigger).
    Its links to the shot dashboards work while the folders stay together.

The analysis writes them automatically (``--no-dashboard`` to skip). To rebuild
them from existing results, without re-running any analysis:

    python3 dashboard.py shockTube_Marseille            # campaign + every shot
    python3 dashboard.py shockTube_Marseille/26284_1_9  # one shot
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as _dt
import glob
import io
import json
import os
import sys

import numpy as np

import change_common as cc
import xt_diagram as xtd

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dashboard_template.html")
MAX_ROWS = 3000          # frames shown at full resolution before binning


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _uri(path, max_w=1600, fmt="JPEG"):
    """Image file -> data URI (re-encoded, downscaled if wider than max_w)."""
    if not path or not os.path.exists(path):
        return None
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, fmt, quality=85) if fmt == "JPEG" else im.save(buf, fmt)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _num(v):
    """CSV string -> int / float when it is one, else the string."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in str(v) else f


def _block(a, step, how):
    """Reduce a 1-D array in blocks of ``step`` (last block may be short)."""
    a = np.asarray(a, dtype=np.float64)
    if step <= 1:
        return a
    n = int(np.ceil(a.size / step))
    pad = n * step - a.size
    fill = {"max": -np.inf, "first": np.nan, "mean": np.nan}[how]
    b = np.concatenate([a, np.full(pad, fill)]).reshape(n, step)
    if how == "max":
        return b.max(1)
    if how == "first":
        return b[:, 0]
    return np.nanmean(b, 1)


def _clean(x):
    """numpy scalars / NaN -> JSON-safe Python values."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _jsonload(act, key):
    try:
        return json.loads(str(act[key])) if key in act else {}
    except (ValueError, TypeError):
        return {}


# --------------------------------------------------------------------------- #
# Reading one shot's analysis folder
# --------------------------------------------------------------------------- #

def load_shot(adir):
    """Everything the dashboards need from ``NAME_analysis/``."""
    adir = os.path.normpath(adir)
    name = os.path.basename(adir)
    name = name[:-len("_analysis")] if name.endswith("_analysis") else name
    paths = cc.ShotPaths(adir, name)
    act = dict(np.load(paths.activity)) if os.path.exists(paths.activity) \
        else None
    return {"name": name, "dir": adir, "paths": paths, "act": act,
            "events": _read_csv(paths.events),
            "summary": _read_csv(paths.summary),
            "shocks": _read_csv(paths.shocks),
            "moments": _read_csv(paths.moments),
            "meta": _jsonload(act, "meta_json") if act else {},
            "params": _jsonload(act, "params_json") if act else {},
            "stats": _jsonload(act, "stats_json") if act else {}}


def _flagged(act):
    return sorted(set(act.get("camera_frames", np.array([])).tolist())
                  | set(act.get("flash_frames", np.array([])).tolist()))


def _verdict(shot):
    act = shot["act"]
    labels = {}
    for s in shot["summary"]:
        lab = s.get("label") or "unlabelled"      # results of older versions
        labels[lab] = labels.get(lab, 0) + 1
    v = {"n_events": len(shot["summary"]), "labels": labels,
         "n_cands": len(shot["shocks"]), "search": False}
    if act is not None and bool(act.get("shock_search", False)):
        sens = [[int(k), float(a), float(p)]
                for k, a, p in act.get("sensitivity", [])]
        v.update(search=True, best_score=float(act["best_score"]),
                 shock_k=float(act["shock_k"]), sens=sens,
                 sens8=next((p for k, a, p in sens if k == 8), None))
    if shot["moments"]:
        big = max(shot["moments"], key=lambda m: int(m["area_px"]))
        v["largest_moment"] = {"frame": int(big["frame"]),
                               "area": int(big["area_px"])}
    return v


def _series(act, step):
    """Per-frame curves reduced to display rows."""
    fnum = act["fnum"]
    level = act["level"].astype(np.float64)
    med = np.median(level) if level.size else 1.0
    t = act.get("time_us", np.full(fnum.size, np.nan))
    return {"frame": _block(fnum, step, "first").astype(int).tolist(),
            "area": _block(act["area"], step, "max").astype(int).tolist(),
            "level_pct": _block(100.0 * (level / med - 1.0), step, "mean"),
            "t_us": _block(t, step, "first")}


def _xt_payload(act, step):
    """The unit-noise x-t diagram, block-reduced (strongest value kept) and
    quantised to int8 for embedding."""
    z, _ = xtd.normalise(act["xt"], _flagged(act))
    T, W = z.shape
    n = int(np.ceil(T / step))
    zp = np.concatenate([z, np.zeros((n * step - T, W), z.dtype)])
    blk = zp.reshape(n, step, W)
    idx = np.abs(blk).argmax(1)
    red = np.take_along_axis(blk, idx[:, None, :], 1)[:, 0, :]
    q = np.clip(np.round(red / 4.0 * 127), -127, 127).astype(np.int8)
    return {"rows": n, "cols": W, "xbin": int(act["xt_bin"]),
            "data": base64.b64encode(q.tobytes()).decode()}


def _thumb(act, w=280, h=110):
    """Small x-t picture for the campaign table."""
    from PIL import Image
    z, _ = xtd.normalise(act["xt"], _flagged(act))
    v = np.clip(z / 4.0, -1, 1)
    neg = np.array([0x2a, 0x78, 0xd6], float)
    mid = np.array([0xf0, 0xef, 0xec], float)
    pos = np.array([0xe3, 0x49, 0x48], float)
    a = np.abs(v)[..., None]
    rgb = np.where(v[..., None] < 0, mid + (neg - mid) * a, mid + (pos - mid) * a)
    im = Image.fromarray(rgb.astype(np.uint8)).resize((w, h), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Shot dashboard
# --------------------------------------------------------------------------- #

def shot_payload(shot, campaign_href=None):
    act, paths = shot["act"], shot["paths"]
    fps = shot["meta"].get("fps")
    data = {"page": "shot", "name": shot["name"],
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "campaign_href": campaign_href, "meta": shot["meta"],
            "params": shot["params"], "stats": shot["stats"],
            "verdict": _verdict(shot), "has_detection": act is not None}

    fr_time = {}
    if act is not None:
        n = act["fnum"].size
        step = max(1, int(np.ceil(n / MAX_ROWS)))
        data["series"] = _series(act, step)
        data["step"] = step
        data["min_area"] = int(act["min_area"])
        data["xt"] = _xt_payload(act, step)
        data["width_px"] = int(act["frame_shape"][1])
        fnum = act["fnum"]
        cam = [int(fnum[i]) for i in act.get("camera_frames", [])]
        fl = [int(fnum[i]) for i in act.get("flash_frames", [])]
        trig = int(act.get("trigger_index", -1))
        data["flags"] = {"camera": cam, "flash": fl,
                         "trigger": int(fnum[trig]) if 0 <= trig < n else None}
        t = act.get("time_us", np.full(n, np.nan))
        fr_time = {int(f): float(tt) for f, tt in zip(fnum, t)}

    events = []
    for s in shot["summary"]:
        eid = s["event_id"]
        pre = paths.event_prefix(eid)
        rows = _read_csv(pre + "_frames.csv")
        peak = _num(s.get("peak_frame"))
        events.append({
            "id": _num(eid), "label": s.get("label", ""),
            "reason": s.get("label_reason", ""),
            "fields": {k: _num(v) for k, v in s.items()},
            "t_peak_us": fr_time.get(peak),
            "images": {"peak": _uri(pre + "_peak.png"),
                       "montage": _uri(pre + "_montage.png"),
                       "trajectory": _uri(pre + "_trajectory.png", 1100)},
            "curve": {"frame": [int(r["frame"]) for r in rows],
                      "area": [int(float(r["area_px"] or 0)) for r in rows],
                      "core": [r["in_core"] == "True" for r in rows]}})
    data["events"] = events
    data["moments"] = [
        {"rank": int(m["rank"]), "frame": int(m["frame"]),
         "area": int(m["area_px"]), "snr": _num(m["peak_snr"]),
         "cx": _num(m["cx"]), "cy": _num(m["cy"]),
         "polarity": m.get("polarity", ""),
         "t_us": fr_time.get(int(m["frame"])),
         "image": _uri(paths.moment_image(m["rank"]), 1000)}
        for m in shot["moments"]]
    data["cands"] = [{k: _num(v) for k, v in c.items()} for c in shot["shocks"]]
    data["static"] = {"overview": _uri(paths.overview, 1800),
                      "timeline": _uri(paths.timeline, 1400)}
    return data


def _render(data, path, title):
    html = open(TEMPLATE, encoding="utf-8").read()
    blob = json.dumps(_clean(data), separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("__TITLE__", title).replace("__DATA__", blob)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


def _campaign_file(campaign_dir):
    campaign_dir = os.path.normpath(campaign_dir)
    return os.path.join(campaign_dir,
                        os.path.basename(campaign_dir) + "_dashboard.html")


def build_shot(adir):
    """Write ``NAME_dashboard.html`` in the analysis folder; returns its path."""
    shot = load_shot(adir)
    parent = os.path.dirname(os.path.normpath(adir))
    camp = _campaign_file(parent)
    href = "../" + os.path.basename(camp) if os.path.exists(camp) else None
    return _render(shot_payload(shot, href), shot["paths"].dashboard,
                   f"{shot['name']} dashboard")


# --------------------------------------------------------------------------- #
# Campaign dashboard
# --------------------------------------------------------------------------- #

def _runs(idx):
    """Consecutive integer runs of a sorted index list -> [(first, last)]."""
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [tuple(r) for r in out]


def data_checks(shots, campaign_dir):
    """Automatic checks on the recordings of a campaign."""
    checks = []
    have = [s for s in shots if s["act"] is not None
            and "frame_hash" in s["act"]]
    # Same recording exported twice (identical frames in two shots).
    for i, a in enumerate(have):
        ha = a["act"]["frame_hash"]
        for b in have[i + 1:]:
            hb = b["act"]["frame_hash"]
            common = np.intersect1d(ha, hb)
            if common.size < 3:
                continue
            ia = np.nonzero(np.isin(ha, common))[0]
            ib = np.nonzero(np.isin(hb, common))[0]
            ra = ", ".join(f"{x}-{y}" for x, y in _runs(ia.tolist())[:3])
            rb = ", ".join(f"{x}-{y}" for x, y in _runs(ib.tolist())[:3])
            if ia.size == ha.size and ib.size == hb.size:
                what = f"{a['name']} and {b['name']} are the same recording"
            elif ia.size == ha.size:
                what = f"{a['name']} is part of {b['name']}"
            elif ib.size == hb.size:
                what = f"{b['name']} is part of {a['name']}"
            else:
                what = f"{a['name']} and {b['name']} share frames"
            checks.append({"level": "warning", "title": what,
                           "detail": f"{common.size} identical frames: "
                                     f"{a['name']} files {ra} = {b['name']} "
                                     f"files {rb}."})
    for s in shots:
        m = s["meta"]
        if s["act"] is None:
            checks.append({"level": "info", "title": f"{s['name']}: older "
                           "analysis format", "detail": "No detection data "
                           "(.npz) in its analysis folder: re-run the analysis "
                           "for its x-t diagram, curves and checks."})
            continue
        if m.get("header_images") and m.get("n_files") and \
                m["header_images"] != m["n_files"]:
            checks.append({"level": "warning", "title": f"{s['name']}: "
                           "incomplete export?", "detail":
                           f"The camera header lists {m['header_images']} "
                           f"images but {m['n_files']} files were exported."})
        if not m.get("fps"):
            checks.append({"level": "warning", "title": f"{s['name']}: no "
                           "frame rate", "detail": "No camera metadata file "
                           "(.cihx or .chd) and no --fps: times are unknown."})
        irr = m.get("irregular_frames") or []
        if len(irr) > 1:
            checks.append({"level": "info", "title": f"{s['name']}: "
                           f"{len(irr)} frames with irregular timing or "
                           "exposure", "detail": f"File indices {irr[:10]}."})
        if s["events"] and not s["summary"]:
            checks.append({"level": "warning", "title": f"{s['name']}: "
                           "analysis incomplete", "detail": "Events were "
                           "detected but not analysed."})
    # Frame folders without results.
    done = {s["name"] for s in shots}
    for d in sorted(glob.glob(os.path.join(campaign_dir, "*"))):
        n = os.path.basename(d)
        if os.path.isdir(d) and not n.endswith("_analysis") and \
                n not in done and glob.glob(os.path.join(d, "*.tif")):
            checks.append({"level": "warning", "title": f"{n}: not analysed",
                           "detail": "Frames present but no analysis folder."})
    # Settings that differ between shots.
    for key, lab in (("fps", "frame rate"), ("exposure_us", "exposure")):
        vals = {s["meta"].get(key) for s in shots if s["meta"].get(key)}
        if len(vals) > 1:
            checks.append({"level": "info", "title": f"Different {lab} "
                           "across shots", "detail": ", ".join(
                               f"{s['name']}: {s['meta'].get(key):g}"
                               for s in shots if s["meta"].get(key))})
    if not checks:
        checks.append({"level": "good", "title": "No problem found",
                       "detail": "No duplicate, incomplete export or "
                                 "missing metadata."})
    return checks


def _aligned(act, max_pts=1500):
    """Activity and brightness against time from the trigger, for overlays."""
    t = act.get("time_us")
    if t is None or not np.isfinite(t).any():
        return None
    step = max(1, int(np.ceil(t.size / max_pts)))
    level = act["level"].astype(np.float64)
    return {"t_us": _block(t, step, "first"),
            "area": _block(act["area"], step, "max"),
            "level_pct": _block(100.0 * (level / np.median(level) - 1.0),
                                step, "mean")}


def campaign_payload(shots, campaign_dir):
    rows = []
    for s in shots:
        act, m = s["act"], s["meta"]
        rel = os.path.relpath(s["paths"].dashboard, campaign_dir)
        row = {"name": s["name"], "href": rel.replace(os.sep, "/"),
               "verdict": _verdict(s), "meta": m, "stats": s["stats"],
               "has_detection": act is not None}
        if act is not None:
            row["n_frames"] = int(act["fnum"].size)
            trig = int(act.get("trigger_index", -1))
            row["trigger_frame"] = (int(act["fnum"][trig])
                                    if 0 <= trig < act["fnum"].size else None)
            row["thumb"] = _thumb(act)
            row["aligned"] = _aligned(act)
        else:
            row["n_frames"] = None
        rows.append(row)
    return {"page": "campaign",
            "name": os.path.basename(os.path.normpath(campaign_dir)),
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "shots": rows, "checks": data_checks(shots, campaign_dir)}


def analysis_dirs(campaign_dir):
    return sorted(d for d in glob.glob(os.path.join(campaign_dir, "*_analysis"))
                  if os.path.isdir(d))


def build_campaign(campaign_dir, shots_too=True):
    """Write the campaign dashboard (and, by default, every shot dashboard)."""
    dirs = analysis_dirs(campaign_dir)
    path = _campaign_file(campaign_dir)
    shots = [load_shot(d) for d in dirs]
    _render(campaign_payload(shots, campaign_dir), path,
            f"{os.path.basename(os.path.normpath(campaign_dir))} dashboard")
    if shots_too:                       # now the back-links can point to it
        for d in dirs:
            build_shot(d)
    return path


def after_run(input_dir, batch, enabled):
    """Called by the scripts after a run: refresh the campaign dashboard after
    a batch, or after a single shot when its campaign already has one."""
    if not enabled:
        return
    campaign = input_dir if batch else os.path.dirname(
        os.path.normpath(input_dir))
    if batch or os.path.exists(_campaign_file(campaign)):
        path = build_campaign(campaign, shots_too=batch)
        print(f"[dashboard] wrote {path}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Rebuild the HTML dashboards from existing results.")
    ap.add_argument("path", help="A campaign folder, a shot frame folder, or "
                                 "a NAME_analysis folder.")
    p = ap.parse_args(argv)
    path = os.path.normpath(p.path)
    if path.endswith("_analysis"):
        print(build_shot(path))
    elif os.path.isdir(path + "_analysis"):
        print(build_shot(path + "_analysis"))
    elif analysis_dirs(path):
        print(build_campaign(path))
    else:
        sys.exit(f"no analysis results found for {p.path!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

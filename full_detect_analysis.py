#!/usr/bin/env python3
"""
full_detect_analysis.py  --  run Script 1 then Script 2 in one command.
======================================================================

Chains ``detect_activity.py`` (find the events) and ``analyze_activity.py``
(characterise each one) on a shot folder, with a single set of options: the
region of interest, ``--rotate`` and the change-detection settings are passed
to both steps, so they always match.

For a frame folder ``NAME`` everything goes into ``NAME_analysis/`` next to
it (``--out-dir`` to change), every file prefixed with ``NAME``:

  * ``NAME_dashboard.html``      -- interactive, self-contained: start here
  * ``NAME_overview.png``        -- is anything seen? as one picture
  * ``NAME_events_summary.csv``  -- labelled events
  * ``NAME_events.csv``          -- detected events
  * ``NAME_shocks.csv``          -- shock-search candidates (``--no-shock-search``)
  * ``NAME_timeline.png``        -- x-t diagram + activity (``--no-plot-timeline``)
  * ``NAME_activity.npz``        -- per-frame arrays, x-t diagram
  * ``NAME_moments.csv``         -- strongest moments below the thresholds
  * ``NAME_tube.png`` / ``.json`` -- the micro-tube found (``--tube off``)
  * ``NAME_tube_shocks.csv``     -- fronts found inside the tube's bore
  * ``event_<id>/``              -- per-event CSV, plots, frames and MP4
                                    (``--no-plots``, ``--no-dump-frames``,
                                    ``--no-movie`` to trim)

If the input folder holds no frames itself, every folder below it that does, at
any depth, is treated as a shot (batch mode), and ``CAMPAIGN_dashboard.html`` is written in it
at the end. A shot whose ``NAME_events.csv`` and
``NAME_events_summary.csv`` both exist is skipped; one with only
``NAME_events.csv`` gets the analysis step only. ``--force`` redoes everything. A failing shot does not stop the others.

Both scripts remain usable on their own; this one only wires them together.

Example
-------
    python3 full_detect_analysis.py shockTube_Marseille/26284_1_5 --rotate 90
    python3 full_detect_analysis.py shockTube_Marseille --rotate 90   # every shot
"""

from __future__ import annotations

import argparse
import os
import sys

import analyze_activity
import change_common as cc
import dashboard
import detect_activity


def build_parser():
    ap = detect_activity.build_parser(
        description="Detect change events, then characterise each of them.")
    analyze_activity.add_analysis_options(ap)
    return ap


def main(argv=None):
    ap = build_parser()
    p = ap.parse_args(argv)
    shots, batch = cc.find_shots(p.input_dir, p.pattern)
    cc.check_batch_args(ap, p, batch, ["--out-dir"])
    analyze_activity.check_movie(p)

    def process(shot):
        q = argparse.Namespace(**vars(p))
        q.input_dir = shot
        detect_activity.resolve_outputs(q)
        q.events_csv = q.out
        analyze_activity.resolve_outputs(q)

        have_events = os.path.exists(q.events_csv)
        have_analysis = os.path.exists(q.paths.summary)
        if batch and not q.force and have_events and have_analysis:
            return "events CSV and events summary exist; use --force to redo"

        if batch and not q.force and have_events:
            print(f"[full] {q.events_csv} exists, running the analysis only",
                  file=sys.stderr)
        else:
            # Each step gets its own copy: detection fills in fps / min-area.
            detect_activity.run(argparse.Namespace(**vars(q)))
        analyze_activity.run(argparse.Namespace(**vars(q)))

    code = cc.run_shots("full", shots, batch, process)
    dashboard.after_run(p.input_dir, batch, p.dashboard)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

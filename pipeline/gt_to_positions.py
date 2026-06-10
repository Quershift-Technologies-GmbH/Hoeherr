#!/usr/bin/env python3
"""
TeamTrack/SportsLabKit-GT-CSVs -> positions.parquet (gleiches Schema wie run_tracking.py).

Mehrere aufeinanderfolgende 30s-Segmente werden zu einer kontinuierlichen Timeline
verkettet (Segment-Offset aus dem Dateinamen: D_..._0030_0060 -> Start bei 30s).

WICHTIG (Nadir): Position = Bbox-ZENTRUM (aus der Senkrechten ist das Zentrum die
Bodenposition; bottom-center wäre um die halbe Körperprojektion verschoben).

tracker_id = TeamID*100 + PlayerID (GT-stabil). Ball wird übersprungen.

    .venv/bin/python pipeline/gt_to_positions.py \
        --annotations teamtrack/annotations --pattern "D_20220220_1_0*" \
        --end-sec 300 --fps 29.97 --out positions_gt.parquet
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def parse_segment(path: str, fps: float) -> pd.DataFrame:
    name = os.path.basename(path)
    m = re.search(r"_(\d{4})_(\d{4})\.csv$", name)
    offset_sec = int(m.group(1))
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    rows = []
    # Spalten: MultiIndex (TeamID, PlayerID, Attribute)
    players = sorted({(t, p) for t, p, _ in df.columns if t != "BALL" and p != "BALL"})
    for team, player in players:
        try:
            sub = df[(team, player)]
        except KeyError:
            continue
        need = {"bb_left", "bb_top", "bb_width", "bb_height"}
        if not need.issubset(set(sub.columns)):
            continue
        x = sub["bb_left"] + sub["bb_width"] / 2.0
        y = sub["bb_top"] + sub["bb_height"] / 2.0
        valid = sub["bb_width"].notna() & (sub["bb_width"] > 0)
        frames = df.index[valid].to_numpy()
        tid = int(team) * 100 + int(player)
        for fr, xi, yi, wi, hi in zip(frames, x[valid], y[valid],
                                      sub["bb_width"][valid], sub["bb_height"][valid]):
            rows.append((int(fr) + round(offset_sec * fps),
                         offset_sec + int(fr) / fps,
                         tid, float(xi), float(yi), float(wi), float(hi),
                         1.0, np.nan, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["frame", "t_sec", "tracker_id", "x_px", "y_px",
                                       "w_px", "h_px", "conf", "hue", "sat", "val"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="teamtrack/annotations")
    ap.add_argument("--pattern", default="D_20220220_1_0*")
    ap.add_argument("--end-sec", type=float, default=300.0)
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--stride", type=int, default=3, help="Sampling wie CV-Pipeline")
    ap.add_argument("--out", default="positions_gt.parquet")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.annotations, args.pattern + ".csv")))
    parts = []
    for p in paths:
        m = re.search(r"_(\d{4})_(\d{4})\.csv$", p)
        if int(m.group(1)) >= args.end_sec:
            continue
        seg = parse_segment(p, args.fps)
        parts.append(seg)
        print(f"[seg] {os.path.basename(p)}: {len(seg)} Samples, "
              f"{seg.tracker_id.nunique()} GT-Spieler")
    df = pd.concat(parts, ignore_index=True).sort_values(["frame", "tracker_id"])
    df = df[df.frame % args.stride == 0]
    df = df[df.t_sec <= args.end_sec]
    df.to_parquet(args.out, index=False)
    print(f"[done] {len(df)} Zeilen, {df.tracker_id.nunique()} Spieler, "
          f"{df.t_sec.max():.1f}s -> {args.out}")


if __name__ == "__main__":
    main()

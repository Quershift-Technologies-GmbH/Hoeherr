#!/usr/bin/env python3
"""
Offline-Retracking: rohe Detections (run_tracking.py --raw-out) durch ByteTrack
replayen — Parameter-Experimente in Sekunden statt erneuter YOLO-Inferenz.
Danach optionales Gap-Stitching: Track endet bei (X,Y), anderer startet kurz
danach in der Nähe -> gleiche Person (statische Nadir-Kamera macht das robust).

    .venv/bin/python pipeline/retrack.py --raw detections_raw.parquet \
        --homography homography.npz --out positions_cv2.parquet \
        --min-conf 0.10 --activation 0.30 --buffer 60 --stitch-gap 4.0 --stitch-dist 2.5
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import supervision as sv


def dedup(raw: pd.DataFrame, H: np.ndarray, radius_m: float) -> pd.DataFrame:
    """SAHI-Kachel-Duplikate entfernen: pro Frame greedy höchste Conf zuerst,
    alles im Umkreis radius_m (Meter) fliegt raus."""
    pts = raw[["x_px", "y_px"]].to_numpy(np.float64).reshape(-1, 1, 2)
    XY = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    raw = raw.copy()
    raw["_X"], raw["_Y"] = XY[:, 0], XY[:, 1]
    keep_idx = []
    for _, g in raw.groupby("frame", sort=False):
        g = g.sort_values("conf", ascending=False)
        xy = g[["_X", "_Y"]].to_numpy()
        taken = np.zeros(len(g), dtype=bool)
        for i in range(len(g)):
            if taken[i]:
                continue
            keep_idx.append(g.index[i])
            d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
            taken |= d < radius_m
    out = raw.loc[sorted(keep_idx)].drop(columns=["_X", "_Y"])
    print(f"[dedup] {len(raw)} -> {len(out)} Detections (r={radius_m}m)")
    return out


def replay_bytetrack(raw: pd.DataFrame, fps_eff: float, activation: float,
                     buffer_frames: int, match_thresh: float,
                     min_consecutive: int, inflate: float = 1.0) -> pd.DataFrame:
    tracker = sv.ByteTrack(
        track_activation_threshold=activation,
        lost_track_buffer=buffer_frames,
        minimum_matching_threshold=match_thresh,
        frame_rate=max(1, int(round(fps_eff))),
        minimum_consecutive_frames=min_consecutive,
    )
    out = []
    for frame_no, g in raw.groupby("frame", sort=True):
        # inflate: Boxen NUR für die IoU-Assoziation aufblasen (Buffered-IoU-
        # Trick für winzige schnelle Objekte) — Positionen bleiben Original.
        wi = g.w_px * inflate / 2
        hi = g.h_px * inflate / 2
        xyxy = np.stack([g.x_px - wi, g.y_px - hi,
                         g.x_px + wi, g.y_px + hi], axis=1).astype(np.float32)
        det = sv.Detections(
            xyxy=xyxy,
            confidence=g.conf.to_numpy(np.float32),
            class_id=np.zeros(len(g), dtype=int),
        )
        # sv 0.28 reicht data{} nicht durch den Tracker -> Rückmapping über xyxy
        key2idx = {tuple(np.round(b, 1)): i for b, i in zip(xyxy, g.index)}
        det = tracker.update_with_detections(det)
        if det.tracker_id is None:
            continue
        for tid, box in zip(det.tracker_id, det.xyxy):
            idx = key2idx.get(tuple(np.round(box, 1)))
            if idx is not None:
                out.append((int(idx), int(tid)))
    m = pd.DataFrame(out, columns=["idx", "tracker_id"]).set_index("idx")
    df = raw.drop(columns=["tracker_id"]).join(m, how="inner")
    return df


def stitch_tracks(df: pd.DataFrame, H: np.ndarray, max_gap_s: float,
                  max_dist_m: float) -> pd.DataFrame:
    """Greedy: Track-Ende -> nächster Track-Start (zeitlich) im Umkreis mergen."""
    pts = df[["x_px", "y_px"]].to_numpy(np.float64).reshape(-1, 1, 2)
    XY = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    df = df.copy()
    df["X"], df["Y"] = XY[:, 0], XY[:, 1]

    info = []
    for tid, g in df.groupby("tracker_id"):
        g = g.sort_values("t_sec")
        info.append({
            "tid": int(tid),
            "t0": g.t_sec.iloc[0], "t1": g.t_sec.iloc[-1],
            "x0": g.X.iloc[0], "y0": g.Y.iloc[0],
            "x1": g.X.iloc[-1], "y1": g.Y.iloc[-1],
        })
    info.sort(key=lambda r: r["t0"])

    parent = {r["tid"]: r["tid"] for r in info}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    ends = sorted(info, key=lambda r: r["t1"])
    merged = 0
    used_starts = set()
    for end in ends:
        best = None
        for start in info:  # info ist nach t0 sortiert -> break erlaubt
            dt = start["t0"] - end["t1"]
            if dt > max_gap_s:
                break
            if start["tid"] == end["tid"] or start["tid"] in used_starts:
                continue
            if dt < -0.15:
                continue
            d = np.hypot(start["x0"] - end["x1"], start["y0"] - end["y1"])
            if d > max_dist_m:
                continue
            score = d + dt * 0.3
            if best is None or score < best[0]:
                best = (score, start["tid"])
        if best is not None:
            tgt = best[1]
            if find(tgt) != find(end["tid"]):
                parent[find(tgt)] = find(end["tid"])
                used_starts.add(tgt)
                merged += 1
    df["tracker_id"] = df["tracker_id"].map(lambda t: find(int(t)))
    print(f"[stitch] {merged} Merges -> {df.tracker_id.nunique()} Tracks")
    return df.drop(columns=["X", "Y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="detections_raw.parquet")
    ap.add_argument("--homography", default="homography.npz")
    ap.add_argument("--out", default="positions_cv2.parquet")
    ap.add_argument("--min-conf", type=float, default=0.10)
    ap.add_argument("--activation", type=float, default=0.30)
    ap.add_argument("--buffer", type=int, default=60, help="lost_track_buffer (Frames)")
    ap.add_argument("--match", type=float, default=0.8)
    ap.add_argument("--min-consecutive", type=int, default=2)
    ap.add_argument("--stitch-gap", type=float, default=0.0, help="0 = kein Stitching")
    ap.add_argument("--stitch-dist", type=float, default=2.5)
    ap.add_argument("--inflate", type=float, default=1.0,
                    help="Box-Inflation nur für IoU-Matching (z.B. 2.5)")
    ap.add_argument("--dedup-m", type=float, default=0.0,
                    help="Duplikat-Radius in Metern (z.B. 0.6), 0 = aus")
    args = ap.parse_args()

    raw = pd.read_parquet(args.raw)
    raw = raw[raw.conf >= args.min_conf].reset_index(drop=True)
    ts = np.sort(raw.t_sec.unique())
    fps_eff = 1.0 / float(np.median(np.diff(ts)))
    print(f"[raw] {len(raw)} Detections, {fps_eff:.1f}fps effektiv")
    if args.dedup_m > 0:
        raw = dedup(raw, np.load(args.homography)["H"], args.dedup_m)

    df = replay_bytetrack(raw, fps_eff, args.activation, args.buffer,
                          args.match, args.min_consecutive, args.inflate)
    print(f"[track] {df.tracker_id.nunique()} Tracks, {len(df)} Samples")

    if args.stitch_gap > 0:
        H = np.load(args.homography)["H"]
        df = stitch_tracks(df, H, args.stitch_gap, args.stitch_dist)

    df.to_parquet(args.out, index=False)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()

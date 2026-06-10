#!/usr/bin/env python3
"""
Annotierter Demo-Clip aus konsolidierten Positionen: Spieler-Marker + IDs
im Video, dazu Pitch-Radar (Mini-Map) unten links.

    .venv/bin/python pipeline/render_demo.py --source clip5m_drone.mp4 \
        --positions positions_consolidated.parquet --homography homography.npz \
        --out demo_annotated.mp4 --seconds 30
"""
import argparse

import cv2
import numpy as np
import pandas as pd

TEAM_COLOR = {True: (60, 60, 230), False: (230, 160, 40)}  # BGR: rot / cyan-blau
RADAR_W, RADAR_H = 420, 280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--homography", default="homography.npz")
    ap.add_argument("--out", default="demo_annotated.mp4")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    hom = np.load(args.homography)
    H = hom["H"]
    L, W = float(hom["pitch_length"]), float(hom["pitch_width"])
    df = pd.read_parquet(args.positions)
    pts = df[["x_px", "y_px"]].to_numpy(np.float64).reshape(-1, 1, 2)
    XY = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    df["X"], df["Y"] = XY[:, 0], XY[:, 1]
    by_frame = dict(tuple(df.groupby("frame")))

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    Wv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
    Hv = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wv, Hv))

    n_frames = int(args.seconds * fps)
    last = {}
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if i in by_frame:
            last = {int(r.tracker_id): r for r in by_frame[i].itertuples()}
        frame = cv2.resize(frame, (Wv, Hv))
        # Spieler-Marker
        for tid, r in last.items():
            x, y = int(r.x_px * args.scale), int(r.y_px * args.scale)
            col = TEAM_COLOR[tid < 100]
            cv2.circle(frame, (x, y), 14, col, 2)
            cv2.putText(frame, str(tid % 100), (x + 12, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        # Radar
        rx, ry = 20, Hv - RADAR_H - 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx, ry), (rx + RADAR_W, ry + RADAR_H), (20, 40, 20), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        cv2.rectangle(frame, (rx, ry), (rx + RADAR_W, ry + RADAR_H), (255, 255, 255), 1)
        cv2.line(frame, (rx + RADAR_W // 2, ry), (rx + RADAR_W // 2, ry + RADAR_H),
                 (200, 200, 200), 1)
        cv2.circle(frame, (rx + RADAR_W // 2, ry + RADAR_H // 2), 24, (200, 200, 200), 1)
        for tid, r in last.items():
            px = rx + int(r.X / L * RADAR_W)
            py = ry + int(r.Y / W * RADAR_H)
            cv2.circle(frame, (px, py), 5, TEAM_COLOR[tid < 100], -1)
        cv2.putText(frame, "hoeherr v0 - supervision pipeline (Tier A)",
                    (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out.write(frame)
    out.release()
    print(f"[done] {n_frames} Frames -> {args.out}")


if __name__ == "__main__":
    main()

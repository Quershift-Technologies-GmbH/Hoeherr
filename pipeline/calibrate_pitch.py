#!/usr/bin/env python3
"""
hoeherr Pipeline — Stufe 2: Pixel -> Pitch-Meter (Homography).

Input: points.json mit >=4 Punktpaaren
    {"pitch_length": 105.0, "pitch_width": 68.0,
     "points": [{"px": [x, y], "pitch": [X, Y]}, ...]}
Pitch-Koordinaten: X entlang der Länge (0..length), Y entlang der Breite (0..width),
Ursprung = linke obere Ecke aus Kamerasicht.

Output: homography.npz (Matrix H, Maße) + overlay_check.png (projizierte
Spielfeldlinien auf Referenzframe — visuelle Verifikation!).

    .venv/bin/python pipeline/calibrate_pitch.py --frame ref.png \
        --points points.json --out homography.npz
"""
import argparse
import json

import cv2
import numpy as np


def pitch_lines(length: float, width: float) -> list[np.ndarray]:
    """Standard-Spielfeldlinien als Polylines in Pitch-Metern."""
    L, W = length, width
    lines = [
        np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], dtype=float),  # Außenlinien
        np.array([[L / 2, 0], [L / 2, W]], dtype=float),                  # Mittellinie
        # Strafräume (16,5m tief, 40,32m breit)
        np.array([[0, W / 2 - 20.16], [16.5, W / 2 - 20.16],
                  [16.5, W / 2 + 20.16], [0, W / 2 + 20.16]], dtype=float),
        np.array([[L, W / 2 - 20.16], [L - 16.5, W / 2 - 20.16],
                  [L - 16.5, W / 2 + 20.16], [L, W / 2 + 20.16]], dtype=float),
        # Fünfmeterräume (5,5m tief, 18,32m breit)
        np.array([[0, W / 2 - 9.16], [5.5, W / 2 - 9.16],
                  [5.5, W / 2 + 9.16], [0, W / 2 + 9.16]], dtype=float),
        np.array([[L, W / 2 - 9.16], [L - 5.5, W / 2 - 9.16],
                  [L - 5.5, W / 2 + 9.16], [L, W / 2 + 9.16]], dtype=float),
    ]
    # Mittelkreis (r=9,15m)
    th = np.linspace(0, 2 * np.pi, 64)
    circle = np.stack([L / 2 + 9.15 * np.cos(th), W / 2 + 9.15 * np.sin(th)], axis=1)
    lines.append(circle)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="Referenzframe PNG")
    ap.add_argument("--points", required=True, help="points.json")
    ap.add_argument("--out", default="homography.npz")
    ap.add_argument("--overlay", default="overlay_check.png")
    args = ap.parse_args()

    cfg = json.load(open(args.points))
    L = float(cfg.get("pitch_length", 105.0))
    W = float(cfg.get("pitch_width", 68.0))
    src = np.array([p["px"] for p in cfg["points"]], dtype=np.float64)
    dst = np.array([p["pitch"] for p in cfg["points"]], dtype=np.float64)
    assert len(src) >= 4, "mind. 4 Punktpaare nötig"

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    assert H is not None, "findHomography fehlgeschlagen"

    # Reprojektionsfehler auf den Kalibrierpunkten
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1)
    print(f"[calib] {len(src)} Punkte, Reprojektion (Meter): "
          f"mean={err.mean():.2f} max={err.max():.2f}, Inlier={int(mask.sum())}/{len(src)}")

    np.savez(args.out, H=H, pitch_length=L, pitch_width=W)

    # Overlay: Pitch-Linien zurück ins Bild projizieren
    frame = cv2.imread(args.frame)
    Hinv = np.linalg.inv(H)
    for line in pitch_lines(L, W):
        px = cv2.perspectiveTransform(line.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        pts = px.astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
    for s in src.astype(int):
        cv2.circle(frame, tuple(s), 8, (0, 255, 255), 2)
    cv2.imwrite(args.overlay, frame)
    print(f"[done] H -> {args.out}, Kontrollbild -> {args.overlay}")


if __name__ == "__main__":
    main()

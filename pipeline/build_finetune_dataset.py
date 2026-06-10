#!/usr/bin/env python3
"""
Fine-Tune-Dataset aus TeamTrack-GT bauen: 4K-Frames -> 640x640-Kacheln + YOLO-Labels.

Klassen: 0=player, 1=ball. Kacheln ohne Spieler werden bis auf jede 10. verworfen
(Negativ-Beispiele). Box bleibt, wenn >=50% der Fläche in der Kachel liegt.

    .venv/bin/python pipeline/build_finetune_dataset.py \
        --videos teamtrack/videos --annotations teamtrack/annotations \
        --train D_20220220_1_0600_0630 D_20220220_1_0660_0690 \
                D_20220220_1_0720_0750 D_20220220_1_0780_0810 \
        --val D_20220220_1_0900_0930 --frame-step 30 --out dataset
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TILE = 640


def load_boxes(csv_path: str) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """frame -> [(cls, x1, y1, x2, y2)] mit cls 0=player, 1=ball."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    entities = sorted({(t, p) for t, p, _ in df.columns})
    boxes: dict[int, list] = {}
    for team, player in entities:
        sub = df[(team, player)]
        if "bb_left" not in sub.columns:
            continue
        cls = 1 if str(team) == "BALL" else 0
        valid = sub["bb_width"].notna() & (sub["bb_width"] > 0)
        for fr in df.index[valid]:
            r = sub.loc[fr]
            x1, y1 = float(r["bb_left"]), float(r["bb_top"])
            boxes.setdefault(int(fr), []).append(
                (cls, x1, y1, x1 + float(r["bb_width"]), y1 + float(r["bb_height"])))
    return boxes


def tile_origins(W: int, H: int) -> list[tuple[int, int]]:
    xs = [round(i * (W - TILE) / max(1, (W - TILE) // TILE)) for i in
          range((W - TILE) // TILE + 1)]
    ys = [round(i * (H - TILE) / max(1, round((H - TILE) / TILE + 0.5)))
          for i in range(round((H - TILE) / TILE + 0.5) + 1)]
    xs = sorted(set(min(x, W - TILE) for x in xs))
    ys = sorted(set(min(y, H - TILE) for y in ys))
    return [(x, y) for y in ys for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="teamtrack/videos")
    ap.add_argument("--annotations", default="teamtrack/annotations")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--frame-step", type=int, default=30)
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": [0, 0], "val": [0, 0]}  # [tiles, boxes]
    empty_counter = 0
    for split, segs in (("train", args.train), ("val", args.val)):
        for seg in segs:
            vid = os.path.join(args.videos, seg + ".mp4")
            boxes = load_boxes(os.path.join(args.annotations, seg + ".csv"))
            cap = cv2.VideoCapture(vid)
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            origins = tile_origins(W, H)
            for fr in range(0, n, args.frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, frame = cap.read()
                if not ok or fr not in boxes:
                    continue
                for ox, oy in origins:
                    labels = []
                    for cls, x1, y1, x2, y2 in boxes[fr]:
                        cx1, cy1 = max(x1, ox), max(y1, oy)
                        cx2, cy2 = min(x2, ox + TILE), min(y2, oy + TILE)
                        if cx2 - cx1 < 3 or cy2 - cy1 < 3:
                            continue
                        if (cx2 - cx1) * (cy2 - cy1) < 0.5 * (x2 - x1) * (y2 - y1):
                            continue
                        bw, bh = cx2 - cx1, cy2 - cy1
                        labels.append(f"{cls} {(cx1 - ox + bw / 2) / TILE:.6f} "
                                      f"{(cy1 - oy + bh / 2) / TILE:.6f} "
                                      f"{bw / TILE:.6f} {bh / TILE:.6f}")
                    has_player = any(l.startswith("0 ") for l in labels)
                    if not has_player:
                        empty_counter += 1
                        if empty_counter % 10:
                            continue
                    name = f"{seg}_f{fr:05d}_x{ox}_y{oy}"
                    tile = frame[oy:oy + TILE, ox:ox + TILE]
                    cv2.imwrite(str(out / "images" / split / (name + ".jpg")), tile,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    (out / "labels" / split / (name + ".txt")).write_text(
                        "\n".join(labels) + ("\n" if labels else ""))
                    stats[split][0] += 1
                    stats[split][1] += len(labels)
            cap.release()
            print(f"[{split}] {seg}: kumulativ {stats[split][0]} Kacheln, "
                  f"{stats[split][1]} Boxen", flush=True)

    yaml = out / "teamtrack_tiles.yaml"
    yaml.write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: player\n  1: ball\n")
    print(f"[done] {stats} -> {yaml}")


if __name__ == "__main__":
    main()

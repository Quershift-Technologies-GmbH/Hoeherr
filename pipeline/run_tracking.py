#!/usr/bin/env python3
"""
hoeherr Pipeline — Stufe 1: Video -> Detections -> Tracks -> positions.parquet

Output-Schema (eine Zeile pro Track pro Sample-Frame):
    frame, t_sec, tracker_id, x_px, y_px (Fußpunkt = bottom-center bbox),
    w_px, h_px, conf, hue, sat, val (Trikotfarbe, Median des oberen Bbox-Drittels)

Beispiel:
    .venv/bin/python pipeline/run_tracking.py --source clip5m.mkv \
        --out positions.parquet --stride 3 --imgsz 1536 --device mps
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

PERSON = 0


def jersey_hsv(frame: np.ndarray, xyxy: np.ndarray) -> tuple[float, float, float]:
    """Median-HSV des oberen Drittels der Box (Trikot), zentrale 60% Breite."""
    x1, y1, x2, y2 = xyxy.astype(int)
    h = y2 - y1
    w = x2 - x1
    if h < 6 or w < 4:
        return (np.nan, np.nan, np.nan)
    cx1 = x1 + int(0.2 * w)
    cx2 = x2 - int(0.2 * w)
    cy1 = y1 + int(0.10 * h)
    cy2 = y1 + int(0.45 * h)
    crop = frame[max(cy1, 0):cy2, max(cx1, 0):cx2]
    if crop.size == 0:
        return (np.nan, np.nan, np.nan)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    med = np.median(hsv, axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))


def _nms_merge(detections: sv.Detections, iou_threshold: float = 0.5) -> sv.Detections:
    """Greedy NMS über eine Liste von Detections (nach Multi-Scale-Merge)."""
    if len(detections) == 0:
        return detections
    xyxy = detections.xyxy
    confs = detections.confidence
    order = np.argsort(-confs)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        area_r = (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_threshold]
    idx = np.array(keep)
    return sv.Detections(
        xyxy=xyxy[idx],
        confidence=confs[idx],
        class_id=detections.class_id[idx] if detections.class_id is not None else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="positions.parquet")
    ap.add_argument("--model", default="models/yolov8n_teamtrack_tiles_v0.pt")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--stride", type=int, default=3, help="jeden n-ten Frame verarbeiten")
    ap.add_argument("--slice", action="store_true", help="SAHI-Tiling (sv.InferenceSlicer)")
    ap.add_argument("--multi-scale", action="store_true",
                    help="Multi-Scale SAHI: 2 Durchläufe (320px + 640px Tiles) mit NMS-Merge")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit auf verarbeitete (gesampelte) Frames, 0=alle")
    ap.add_argument("--video-out", default="", help="optionales annotiertes Kontrollvideo")
    ap.add_argument("--nadir", action="store_true",
                    help="Top-Down-Sicht: Bodenposition = Bbox-Zentrum statt bottom-center")
    ap.add_argument("--raw-out", default="",
                    help="zusätzlich ALLE Detections (vor Tracking) speichern -> "
                         "Offline-Retracking ohne erneute YOLO-Inferenz (retrack.py)")
    args = ap.parse_args()

    model = YOLO(args.model)
    vi = sv.VideoInfo.from_video_path(args.source)
    eff_fps = vi.fps / args.stride
    print(f"[info] {vi.width}x{vi.height} @ {vi.fps:.2f}fps, {vi.total_frames} frames "
          f"-> Sampling-Stride {args.stride} = {eff_fps:.2f}fps effektiv")

    tracker = sv.ByteTrack(frame_rate=max(1, int(round(eff_fps))))

    def yolo_detect(frame: np.ndarray) -> sv.Detections:
        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                            device=args.device, classes=[PERSON], verbose=False)[0]
        return sv.Detections.from_ultralytics(res)

    if args.multi_scale:
        # ---- Multi-Scale SAHI: 2 Tile-Größen + NMS-Merge ----
        slicer_small = sv.InferenceSlicer(
            callback=yolo_detect, slice_wh=(320, 320),
            overlap_wh=(64, 64), iou_threshold=0.5,
        )
        slicer_large = sv.InferenceSlicer(
            callback=yolo_detect, slice_wh=(640, 640),
            overlap_wh=(128, 128), iou_threshold=0.5,
        )

        def multi_scale_detect(frame: np.ndarray) -> sv.Detections:
            det_small = slicer_small(frame)
            det_large = slicer_large(frame)
            if len(det_small) == 0:
                return det_large
            if len(det_large) == 0:
                return det_small
            merged = sv.Detections(
                xyxy=np.vstack([det_small.xyxy, det_large.xyxy]),
                confidence=np.concatenate([det_small.confidence, det_large.confidence]),
                class_id=np.concatenate([
                    det_small.class_id if det_small.class_id is not None else np.zeros(len(det_small), dtype=int),
                    det_large.class_id if det_large.class_id is not None else np.zeros(len(det_large), dtype=int),
                ]),
            )
            return _nms_merge(merged, iou_threshold=0.5)

        detect = multi_scale_detect
        print("[info] Multi-Scale SAHI AKTIV (320px + 640px Tiles, NMS-Merge)")
    elif args.slice:
        slicer = sv.InferenceSlicer(callback=yolo_detect, slice_wh=(640, 640),
                                    overlap_wh=(128, 128), iou_threshold=0.5)
        detect = slicer
        print("[info] SAHI-Tiling AKTIV")
    else:
        detect = yolo_detect

    sink = None
    annotators = None
    if args.video_out:
        out_info = sv.VideoInfo(width=vi.width, height=vi.height,
                                fps=max(1, int(round(eff_fps))))
        sink = sv.VideoSink(args.video_out, out_info)
        sink.__enter__()
        annotators = (sv.EllipseAnnotator(thickness=2),
                      sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=3))

    rows = []
    raw_rows = []
    processed = 0
    for i, frame in enumerate(sv.get_video_frames_generator(args.source)):
        if i % args.stride:
            continue
        if args.max_frames and processed >= args.max_frames:
            break
        det = detect(frame)
        if args.raw_out and len(det):
            t_raw = i / vi.fps
            for xyxy, conf in zip(det.xyxy, det.confidence):
                hue, sat, val = jersey_hsv(frame, xyxy)
                pos_y = (xyxy[1] + xyxy[3]) / 2 if args.nadir else xyxy[3]
                raw_rows.append((i, t_raw, -1,
                                 float((xyxy[0] + xyxy[2]) / 2), float(pos_y),
                                 float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1]),
                                 float(conf), hue, sat, val))
        det = tracker.update_with_detections(det)
        t_sec = i / vi.fps
        if det.tracker_id is not None:
            for xyxy, conf, tid in zip(det.xyxy, det.confidence, det.tracker_id):
                hue, sat, val = jersey_hsv(frame, xyxy)
                pos_y = (xyxy[1] + xyxy[3]) / 2 if args.nadir else xyxy[3]
                rows.append((i, t_sec, int(tid),
                             float((xyxy[0] + xyxy[2]) / 2), float(pos_y),
                             float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1]),
                             float(conf), hue, sat, val))
        if sink is not None:
            ann = annotators[0].annotate(frame.copy(), det)
            labels = [f"#{tid}" for tid in det.tracker_id] if det.tracker_id is not None else []
            ann = annotators[1].annotate(ann, det, labels=labels)
            sink.write_frame(ann)
        processed += 1
        if processed % 100 == 0:
            print(f"[{processed}] t={t_sec:.1f}s tracks={len(det)}", flush=True)

    if sink is not None:
        sink.__exit__(None, None, None)

    cols = ["frame", "t_sec", "tracker_id", "x_px", "y_px",
            "w_px", "h_px", "conf", "hue", "sat", "val"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_parquet(args.out, index=False)
    if args.raw_out:
        pd.DataFrame(raw_rows, columns=cols).to_parquet(args.raw_out, index=False)
        print(f"[raw] {len(raw_rows)} Detections -> {args.raw_out}")
    n_tracks = df.tracker_id.nunique() if len(df) else 0
    print(f"\n[done] {len(df)} Zeilen, {n_tracks} Tracks, "
          f"{processed} Frames verarbeitet -> {args.out}")


if __name__ == "__main__":
    main()

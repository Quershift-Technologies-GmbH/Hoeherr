#!/usr/bin/env python3
"""
Offline-Retracking: rohe Detections (run_tracking.py --raw-out) durch ByteTrack
replayen — Parameter-Experimente in Sekunden statt erneuter YOLO-Inferenz.
Danach optionales Gap-Stitching: Track endet bei (X,Y), anderer startet kurz
danach in der Nähe -> gleiche Person (statische Nadir-Kamera macht das robust).

Appearance-Features: Optionaler frozen MobileNetV3-Feature-Extraktor für
Re-ID-basiertes Track-Stitching (--video + --alpha).

Track-Interpolation: Fehlende Detektionen in Gaps <= N Frames werden linear
interpoliert (--max-interp-gap, default 3).

    .venv/bin/python pipeline/retrack.py --raw detections_raw.parquet \
        --homography homography.npz --out positions_cv2.parquet \
        --min-conf 0.05 --activation 0.20 --buffer 60 --stitch-gap 4.0 --stitch-dist 2.5
    (Defaults sind jetzt GT-validiert — obige Flags nur nötig wenn man abweichen will)

    # Mit Appearance-Features:
    .venv/bin/python pipeline/retrack.py --raw detections_raw.parquet \
        --homography homography.npz --out positions_cv2.parquet \
        --video clip5m.mkv --alpha 0.7
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import supervision as sv

# ---- Appearance Feature Extractor (lazy-loaded) ----

_appearance_extractor = None


class AppearanceExtractor:
    """Frozen MobileNetV3-Small Feature-Extraktor für Re-ID Embeddings."""

    def __init__(self, device: str = "cpu"):
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        self.device = torch.device(device)
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        backbone.eval()
        # Entferne Classifier, behalte nur Feature-Extraktor + Pool
        self.model = torch.nn.Sequential(
            backbone.features,
            backbone.avgpool,
            torch.nn.Flatten(),
        ).to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._torch = torch

    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extrahiere Embeddings für eine Liste von BGR-Crops.
        Returns: (N, D) float32 numpy array, L2-normiert."""
        import torch
        if len(crops) == 0:
            return np.zeros((0, 576), dtype=np.float32)
        with torch.no_grad():
            tensors = []
            for c in crops:
                if c.size == 0 or min(c.shape[:2]) < 4:
                    tensors.append(self.transform(np.zeros((64, 64, 3), dtype=np.uint8)))
                else:
                    rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                    tensors.append(self.transform(rgb))
            batch = torch.stack(tensors).to(self.device)
            emb = self.model(batch).cpu().numpy()
            # L2-Normierung
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
            return (emb / norms).astype(np.float32)


def _get_extractor(device: str = "cpu") -> AppearanceExtractor:
    global _appearance_extractor
    if _appearance_extractor is None:
        _appearance_extractor = AppearanceExtractor(device)
    return _appearance_extractor


def compute_track_embeddings(df: pd.DataFrame, video_path: str,
                             device: str = "cpu") -> dict[int, np.ndarray]:
    """Berechne mittleres Appearance-Embedding pro Track aus Video-Crops.
    Samplet bis zu 8 Frames pro Track für Effizienz."""
    extractor = _get_extractor(device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[warn] Video {video_path} nicht lesbar, Appearance deaktiviert")
        return {}

    track_crops: dict[int, list[np.ndarray]] = {}
    max_samples = 8

    for tid, g in df.groupby("tracker_id"):
        tid = int(tid)
        g = g.sort_values("frame")
        # Sample gleichmäßig verteilt
        indices = np.linspace(0, len(g) - 1, min(max_samples, len(g)), dtype=int)
        sampled = g.iloc[indices]
        crops = []
        for _, row in sampled.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row.frame))
            ret, frame = cap.read()
            if not ret:
                continue
            x, y, w, h = row.x_px, row.y_px, row.w_px, row.h_px
            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h))
            x2 = min(frame.shape[1], int(x + w / 2))
            y2 = min(frame.shape[0], int(y))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        if crops:
            track_crops[tid] = crops

    cap.release()

    # Embeddings berechnen
    track_embeddings = {}
    for tid, crops in track_crops.items():
        embs = extractor.extract(crops)
        track_embeddings[tid] = embs.mean(axis=0)  # Mittleres Embedding

    print(f"[appearance] {len(track_embeddings)} Track-Embeddings berechnet")
    return track_embeddings


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
        wi = g.w_px * inflate / 2
        hi = g.h_px * inflate / 2
        xyxy = np.stack([g.x_px - wi, g.y_px - hi,
                         g.x_px + wi, g.y_px + hi], axis=1).astype(np.float32)
        det = sv.Detections(
            xyxy=xyxy,
            confidence=g.conf.to_numpy(np.float32),
            class_id=np.zeros(len(g), dtype=int),
        )
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
                  max_dist_m: float, alpha: float = 1.0,
                  track_embeddings: dict | None = None) -> pd.DataFrame:
    """Greedy: Track-Ende -> nächster Track-Start (zeitlich) im Umkreis mergen.
    Velocity-aware + optional Appearance-aware."""
    pts = df[["x_px", "y_px"]].to_numpy(np.float64).reshape(-1, 1, 2)
    XY = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    df = df.copy()
    df["X"], df["Y"] = XY[:, 0], XY[:, 1]

    info = []
    for tid, g in df.groupby("tracker_id"):
        g = g.sort_values("t_sec")
        tail = g.tail(min(5, len(g)))
        if len(tail) >= 2:
            dt_tail = tail.t_sec.iloc[-1] - tail.t_sec.iloc[0]
            if dt_tail > 0:
                vx = (tail.X.iloc[-1] - tail.X.iloc[0]) / dt_tail
                vy = (tail.Y.iloc[-1] - tail.Y.iloc[0]) / dt_tail
            else:
                vx, vy = 0.0, 0.0
        else:
            vx, vy = 0.0, 0.0
        info.append({
            "tid": int(tid),
            "t0": g.t_sec.iloc[0], "t1": g.t_sec.iloc[-1],
            "x0": g.X.iloc[0], "y0": g.Y.iloc[0],
            "x1": g.X.iloc[-1], "y1": g.Y.iloc[-1],
            "vx": vx, "vy": vy,
        })
    info.sort(key=lambda r: r["t0"])

    parent = {r["tid"]: r["tid"] for r in info}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    use_appearance = (track_embeddings is not None and len(track_embeddings) > 0
                      and alpha < 1.0)

    ends = sorted(info, key=lambda r: r["t1"])
    merged = 0
    used_starts = set()
    for end in ends:
        best = None
        end_emb = track_embeddings.get(end["tid"]) if use_appearance else None
        for start in info:
            dt = start["t0"] - end["t1"]
            if dt > max_gap_s:
                break
            if start["tid"] == end["tid"] or start["tid"] in used_starts:
                continue
            if dt < -0.15:
                continue
            d_raw = np.hypot(start["x0"] - end["x1"], start["y0"] - end["y1"])
            pred_x = end["x1"] + end["vx"] * dt
            pred_y = end["y1"] + end["vy"] * dt
            d_pred = np.hypot(start["x0"] - pred_x, start["y0"] - pred_y)
            d = min(d_raw, d_pred)
            if d > max_dist_m:
                continue
            pos_score = d / max_dist_m + dt * 0.3 / max_gap_s
            if use_appearance and end_emb is not None:
                start_emb = track_embeddings.get(start["tid"])
                if start_emb is not None:
                    cos_sim = float(np.dot(end_emb, start_emb))
                    app_score = 1.0 - cos_sim
                    score = alpha * pos_score + (1 - alpha) * app_score
                else:
                    score = pos_score
            else:
                score = pos_score
            if best is None or score < best[0]:
                best = (score, start["tid"])
        if best is not None:
            tgt = best[1]
            if find(tgt) != find(end["tid"]):
                parent[find(tgt)] = find(end["tid"])
                used_starts.add(tgt)
                merged += 1
    df["tracker_id"] = df["tracker_id"].map(lambda t: find(int(t)))
    mode = f"alpha={alpha}" if use_appearance else "position-only"
    print(f"[stitch] {merged} Merges -> {df.tracker_id.nunique()} Tracks ({mode})")
    return df.drop(columns=["X", "Y"])


def interpolate_tracks(df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Lineare Interpolation fehlender Detektionen für Gaps <= max_gap Frames.

    Für jeden Track werden aufeinanderfolgende Frame-Paare geprüft. Wenn
    zwischen zwei Detektionen ein Gap von 1..max_gap Frames liegt, werden
    die fehlenden Frames linear interpoliert (Bbox-Koordinaten x_px, y_px,
    w_px, h_px). Die Track-ID bleibt erhalten.

    Args:
        df: DataFrame mit Spalten frame, t_sec, tracker_id, x_px, y_px, w_px, h_px, conf, ...
        max_gap: Maximale Lücke in Frames für Interpolation (default: 3)

    Returns:
        DataFrame mit interpolierten Zeilen eingefügt, sortiert nach (tracker_id, frame)
    """
    if max_gap <= 0 or len(df) == 0:
        return df

    all_frames = sorted(df.frame.unique())
    if len(all_frames) < 2:
        return df
    # Frame-Abstände (Stride) aus den Daten ableiten
    frame_diffs = np.diff(all_frames)
    stride = int(np.median(frame_diffs)) if len(frame_diffs) > 0 else 1
    stride = max(1, stride)

    # t_sec pro Frame (Lookup)
    frame_to_t = dict(zip(df.frame, df.t_sec))
    # Fehlende t_sec-Werte interpolieren
    if len(all_frames) >= 2:
        fps_eff = (frame_to_t.get(all_frames[-1], 0) - frame_to_t.get(all_frames[0], 0))
        fps_eff = fps_eff / (all_frames[-1] - all_frames[0]) if all_frames[-1] != all_frames[0] else 0

    interpolated_rows = []
    interp_cols = ["x_px", "y_px", "w_px", "h_px"]
    carry_cols = [c for c in df.columns if c not in
                  ["frame", "t_sec", "tracker_id"] + interp_cols + ["conf"]]

    for tid, g in df.groupby("tracker_id"):
        g = g.sort_values("frame")
        frames = g.frame.values
        for k in range(len(frames) - 1):
            f_start = frames[k]
            f_end = frames[k + 1]
            gap_frames = (f_end - f_start) // stride - 1
            if gap_frames < 1 or gap_frames > max_gap:
                continue
            row_start = g[g.frame == f_start].iloc[0]
            row_end = g[g.frame == f_end].iloc[0]
            for j in range(1, gap_frames + 1):
                alpha_j = j / (gap_frames + 1)
                f_interp = f_start + j * stride
                t_interp = row_start.t_sec + alpha_j * (row_end.t_sec - row_start.t_sec)
                new_row = {"frame": int(f_interp), "t_sec": float(t_interp),
                           "tracker_id": int(tid),
                           "conf": float(min(row_start.conf, row_end.conf) * 0.8)}
                # Lineare Interpolation der Bbox-Koordinaten
                for col in interp_cols:
                    new_row[col] = float(row_start[col] + alpha_j * (row_end[col] - row_start[col]))
                # Carry-Spalten (z.B. hue, sat, val): Werte vom näheren Endpunkt
                for col in carry_cols:
                    if col in g.columns:
                        new_row[col] = row_start[col] if alpha_j <= 0.5 else row_end[col]
                interpolated_rows.append(new_row)

    if not interpolated_rows:
        print("[interp] keine Gaps <= {max_gap} Frames gefunden")
        return df

    df_interp = pd.DataFrame(interpolated_rows)
    df_out = pd.concat([df, df_interp], ignore_index=True)
    df_out = df_out.sort_values(["tracker_id", "frame"]).reset_index(drop=True)
    print(f"[interp] {len(interpolated_rows)} Frames interpoliert "
          f"(max_gap={max_gap}, {df.tracker_id.nunique()} Tracks)")
    return df_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="detections_raw.parquet")
    ap.add_argument("--homography", default="homography.npz")
    ap.add_argument("--out", default="positions_cv2.parquet")
    ap.add_argument("--min-conf", type=float, default=0.05)
    ap.add_argument("--activation", type=float, default=0.20)
    ap.add_argument("--buffer", type=int, default=60, help="lost_track_buffer (Frames)")
    ap.add_argument("--match", type=float, default=0.8)
    ap.add_argument("--min-consecutive", type=int, default=1)
    ap.add_argument("--stitch-gap", type=float, default=4.0, help="0 = kein Stitching")
    ap.add_argument("--stitch-dist", type=float, default=2.5)
    ap.add_argument("--inflate", type=float, default=3.5,
                    help="Box-Inflation nur für IoU-Matching (z.B. 2.5)")
    ap.add_argument("--dedup-m", type=float, default=0.6,
                    help="Duplikat-Radius in Metern (z.B. 0.6), 0 = aus")
    # ---- Appearance Re-ID ----
    ap.add_argument("--video", default="",
                    help="Pfad zum Quellvideo für Appearance-Feature-Extraktion")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="Gewichtung IoU vs. Appearance: cost = alpha*pos + (1-alpha)*app "
                         "(1.0 = nur Position, 0.0 = nur Appearance, default 0.7)")
    ap.add_argument("--appearance-device", default="cpu",
                    help="Device für Feature-Extraktor (cpu/cuda/mps)")
    # ---- Track Interpolation ----
    ap.add_argument("--max-interp-gap", type=int, default=3,
                    help="Maximale Lücke in Frames für lineare Track-Interpolation "
                         "(0 = aus, default 3)")
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
        # Appearance-Embeddings berechnen, falls Video angegeben
        track_embs = None
        if args.video and args.alpha < 1.0:
            track_embs = compute_track_embeddings(df, args.video, args.appearance_device)
        df = stitch_tracks(df, H, args.stitch_gap, args.stitch_dist,
                           alpha=args.alpha, track_embeddings=track_embs)

    # ---- Track Interpolation: fehlende Frames auffüllen ----
    if args.max_interp_gap > 0:
        df = interpolate_tracks(df, max_gap=args.max_interp_gap)

    df.to_parquet(args.out, index=False)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()

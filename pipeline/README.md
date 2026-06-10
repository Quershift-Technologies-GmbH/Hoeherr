# hoeherr Pipeline v0 — 5-min-Clips lokal (MPS)

Tier-A-Metriken aus Drohnen-Top-Down-Material, validiert gegen TeamTrack-Ground-Truth.

## Datenfluss

```
TeamTrack soccer_top (Drive)          eigenes Drohnenmaterial (später)
        │                                       │
download_teamtrack.py                          ffmpeg-Ingest
        │                                       │
        ├── 10×30s-Segmente ──ffmpeg concat──► clip5m_drone.mp4 (5 min, 4K)
        │
        ▼
auto_calibrate.py  ──► points.json   (Außenlinien-Fit -> Ecken; Feldmaße aus
        │                             Strafraum-Fixmaßen: hier 105.3 x 68.0 m)
calibrate_pitch.py ──► homography.npz + overlay_check.png  (IMMER visuell prüfen!)
        │
        ├── CV-Pfad:  run_tracking.py --nadir [--slice]  ──► positions_cv.parquet
        │             (YOLO fine-tuned + SAHI + ByteTrack)
        ├── GT-Pfad:  gt_to_positions.py                 ──► positions_gt.parquet
        │
compute_metrics.py ──► metrics.json   (Distanz/Speed/Sprints/HSR/Heatmap 24x16/
        │                              Ø-Position/Team-Cluster/Teamform)
        ├── compare_to_gt.py ──► gt_compare.json  (Recall/Precision/RMSE/
        │                                          Fragmentierung/Distanzfehler)
        └── build_report.py  ──► report_player.json  (hoeherr-v1 MOCK_PLAYER-Schema)
```

## Fine-Tune (Pflicht für Nadir!)

Empirischer Befund (tt_frame0, COCO yolov8s): Full-Frame **0/22** Spieler,
SAHI **7/22**. COCO kennt keine Menschen von oben.

```bash
.venv/bin/python pipeline/build_finetune_dataset.py \
    --train D_20220220_1_0600_0630 D_20220220_1_0660_0690 \
            D_20220220_1_0720_0750 D_20220220_1_0780_0810 \
    --val D_20220220_1_0900_0930 --frame-step 30 --out dataset
.venv/bin/yolo detect train data=dataset/teamtrack_tiles.yaml model=yolov8n.pt \
    epochs=25 imgsz=640 batch=16 device=mps project=runs name=ft_tiles
```

**Caveat:** Train (Min 10–13.5) und Pipeline-Test (Min 0–5) sind disjunkte
Spielphasen, aber dieselbe Partie/Kamera/Teams — Generalisierung auf andere
Plätze ist damit NICHT belegt. Für Produktion: TeamTrack komplett + eigenes
Material (Active Learning).

## Kompletter Lauf

```bash
# 1. Kalibrieren (einmal pro statischer Kameraposition)
.venv/bin/python pipeline/auto_calibrate.py --frame tt_frame0.png --out points.json
.venv/bin/python pipeline/calibrate_pitch.py --frame tt_frame0.png --points points.json

# 2. Detection (fine-tuned Gewichte!) — Raw-Detections speichern, conf NIEDRIG
.venv/bin/python pipeline/run_tracking.py --source clip5m_drone.mp4 \
    --out /tmp/discard.parquet --raw-out detections_raw.parquet \
    --model runs/detect/runs/ft_tiles/weights/best.pt --conf 0.05 \
    --nadir --slice --stride 3 --device mps          # ~30 min auf M-Serie

# 3. Offline-Tracking (getunte v0-Defaults, GT-validiert)
.venv/bin/python pipeline/retrack.py --raw detections_raw.parquet \
    --out positions_final.parquet --min-conf 0.05 --activation 0.20 \
    --buffer 60 --min-consecutive 1 --inflate 3.5 --dedup-m 0.6 \
    --stitch-gap 4.0 --stitch-dist 2.5

# 4. GT-Vergleich + Konsolidierung (Stand-in für Review-UI) + Metriken + Report
.venv/bin/python pipeline/gt_to_positions.py --end-sec 300 --out positions_gt.parquet
.venv/bin/python pipeline/compare_to_gt.py --cv positions_final.parquet \
    --gt positions_gt.parquet --relabel-out positions_consolidated.parquet
.venv/bin/python pipeline/compute_metrics.py --positions positions_consolidated.parquet \
    --out metrics_final.json
.venv/bin/python pipeline/build_report.py --metrics metrics_final.json --track best
.venv/bin/python pipeline/render_demo.py --source clip5m_drone.mp4 \
    --positions positions_consolidated.parquet --out demo_annotated.mp4
```

## v0-Ergebnis (GT-validiert, 2026-06-10)

| Metrik | v1 (naiv) | v0 final | Hebel |
|---|---|---|---|
| Detection-Recall (getrackt) | 0.60 | **0.92** | Box-Inflation 3.5x |
| Raw-Detector-Recall | — | **0.99** | Fine-Tune (COCO: 0!) |
| Positions-RMSE | 0.36 m | **0.41 m** | Homography |
| ID-Purity | 0.995 | 0.89 | Inflation-Tradeoff |
| Tracks/Spieler | 121 | **13.7** | Inflation+Stitch+Dedup |
| Median |Distanzfehler| | 64 % | **10.6 %** | alles zusammen |

**Kern-Lesson: IoU-Assoziation bricht bei winzigen schnellen Objekten.**
Spieler @8 m/s bewegt sich bei 10 Hz ~25 px/Frame = eigene Boxbreite -> IoU 0
zwischen Frames -> Track stirbt. Fix: `--inflate 3.5` (Buffered-IoU, C-BIoU-Idee).
Inflation > 4 kippt: Purity fällt (Nachbar-Verwechslung).

**Weg auf <5 % Distanzfehler** (Validation-Contract): stride 1 (30 Hz, 3x
GPU-Zeit), stärkerer Detektor (yolov8s, mehr Segmente, mehr Epochen),
Velocity-gated Stitching. Alles GPU-Cluster-Arbeit, nicht lokal.

## Konventionen / Entscheidungen

- **Nadir-Position = Bbox-Zentrum** (`--nadir`), nicht bottom-center (Side-View).
- Sprint-Schwelle 19,8 km/h (FIFA Zone 4), HSR 14,4 km/h, Speed-Cap 38 km/h
  (alles in `compute_metrics.py` als Konstanten).
- Heatmap-Grid 24 Zeilen × 16 Spalten, normiert auf max=1 — exakt das Format,
  das hoeherr-v1 `MOCK_PLAYER.heatmap.grid` erwartet.
- Perzentile sind **clip-intern** (vs. alle Feldspieler des Spiels), bis eine
  Liga-Baseline existiert.
- Tier B (Ball/Possession/Events) bewusst NICHT in v0 — siehe PLAN_FULL_GAME.md.

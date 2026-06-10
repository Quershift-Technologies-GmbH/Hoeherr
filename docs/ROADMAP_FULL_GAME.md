# höherr — Plan: Ganze Spiele per Drohne analysieren (supervision-Orchestrierung)

Stand: 2026-06-10 · Basis: `track_football.py`-Demo (2026-06-08), supervision-Eval in Memory,
Report-Contract gescrapt von https://hoeherr-v1.vercel.app (`MOCK_PLAYER`, index.html:1772)

## 1. Ziel & Daten-Contract

Die hoeherr-v1-Seite verkauft **Spielerentwicklungs-Reports** (6 Sektionen). Das Mock-Objekt
`MOCK_PLAYER` ist der JSON-Contract, den die Pipeline pro Spieler & Spiel liefern muss.

**Machbarkeits-Mapping (Report-Feld → Pipeline-Tier):**

| Report-Feld | Quelle | Tier |
|---|---|---|
| Laufdistanz, Sprints, Top-Speed | Position + Homography | A — sofort |
| Heatmap-Grid 16×24, Centroid | Position | A — sofort |
| Ø-Position, Mitspieler-Distanzen | Position aller Tracks | A — sofort |
| Positionsdisziplin (% Soll) | Position + Formations-Raster | A — definierbar |
| Raumgewinn | Position (pro Lauf statt pro Ballaktion) | A* — Definition anpassen |
| Passoptionen, Entscheidungsqualität, Pressing-Trigger | Ball + Events | B/C — später |
| Decision-Timeline + Video-Clips | Events / Coach-Tagging | Hybrid in v1 |
| Stärken/Schwächen-Texte, Fazit | LLM aus Metriken | A + LLM |
| Perzentile | Vergleichsbasis | Team-intern v1, Liga später |

**v1-Versprechen:** Sektionen 1–3 vollautomatisch (Tier A), Sektion 4 Coach-getaggt im
synchronisierten Video-Player, Sektionen 5–6 LLM-generiert aus Tier-A-Metriken + Coach-Input.
Ein Flug = Reports für ALLE ~14+ eigenen Spieler gleichzeitig (Kosten amortisieren über Team).

## 2. Warum die Demo nicht reicht (Gap-Analyse)

1. **Scale:** 90 min @ 50fps = 270k Frames. Demo-SAHI auf MPS: 0,18 s/Frame → 13,5 h. Unbrauchbar.
2. **Multi-Clip:** Mavic-Akku ~35-40 min → 3+ Segmente pro Spiel (Akkuwechsel, Halbzeit).
   Demo kennt nur ein Video.
3. **Track-Persistenz:** ByteTrack-IDs brechen bei Okklusion/Bildrand; über 90 min braucht es
   Track-Stitching + Identität (wer ist Leon?). Demo hat nur Frame-zu-Frame-IDs.
4. **Keine Pitch-Koordinaten:** Demo arbeitet in Pixeln. Alle Metriken brauchen Meter →
   Homography (bewegte Drohne → pro Keyframe neu).
5. **Kein CMC:** `sv.ByteTrack` hat keine Camera-Motion-Compensation → bewegte Drohne
   erzeugt Geister-Bewegung. Fix: `BoTSORTTracker` aus `roboflow/trackers`.
6. **COCO-Weights:** Nadir-Performance ~5,9 % Precision pre-finetune → TeamTrack-Fine-Tune Pflicht.

## 3. Ziel-Architektur (Orchestrierung)

```
Upload (R2) ──► Orchestrator (FastAPI + Job-Queue, Cloud Run)
   │
   ├─ [CPU] Ingest: ffmpeg normalize, Segmente alignen, in 10-15-min-Chunks schneiden
   ├─ [GPU ×N parallel, RunPod Serverless 4090] pro Chunk:
   │     Sampling 12,5 fps → YOLO (fine-tuned, SAHI nur für Ball)
   │     → BoT-SORT+CMC → Keyframe-Homography → pitch-koordinaten
   │     → positions.parquet + Spieler-Crops (für Re-ID/Team)
   ├─ [CPU] Merge: Cross-Chunk-Track-Stitching
   │     (Appearance-Embeddings + TeamClassifier (SigLIP+UMAP+KMeans) + Spatial-Kontinuität)
   ├─ [Human ≤10 min] Review-UI: Track→Spieler-Zuordnung bestätigen (Jersey-OCR ab nadir unzuverlässig!)
   ├─ [CPU] Metrics-Engine: Tier-A-Metriken, Perzentile, Heatmap-Grids
   ├─ [CPU+LLM] Report-Builder: MOCK_PLAYER-JSON + Texte + ffmpeg-Clips
   └─ Publish: JSON → hoeherr-v1 (MOCK_PLAYER durch fetch ersetzen), Share-Link pro Spieler
```

**Budget-Schätzung:** 67.500 Frames (90 min @ 12,5 fps), 4090 ≈ 0,1-0,15 s/Frame mit SAHI
→ ~2,3 GPU-h ≈ 1-2 €/Spiel; 6 parallele Chunk-Worker → ~25-30 min Wall-Clock.
Nach Fine-Tune ggf. ohne Spieler-SAHI (nur Ball) → 3-5× billiger.

## 4. Phasenplan

**Phase 0 — Contract einfrieren (0,5 Tag)**
JSON-Schema aus MOCK_PLAYER extrahieren, Validation-Contract als prüfbare Assertions (s. §5).

**Phase 1 — Full-Game-Batch-Runner, COCO-Weights (2-3 Tage)**
Chunked Batch-Pipeline lokal/RunPod: Sampling, Chunking, positions.parquet, naive Stitching.
⚠️ Voraussetzung: **echtes Nadir-Testspiel von Mustafa** (Demo lief auf YouTube-Proxy!).
Output: Baseline-Qualitätsreport (ID-Switches/Spieler, Detection-Recall, Laufzeit, Kosten).

**Phase 2 — Modellqualität (1-2 Wochen, parallel zu 3)**
YOLO-Fine-Tune auf TeamTrack (MIT, 4,37M bboxes, Drohnen-Top-Down) · BoT-SORT-CMC aus
`roboflow/trackers` · Homography: 4-Ecken-`cv2.findHomography` (32-Keypoint-Modell failt auf Nadir),
Keypoint-Fine-Tune später.

**Phase 3 — Identität (1 Woche)**
Cross-Chunk-Re-ID (Embeddings + TeamClassifier), minimale Review-UI (Track-Galerie → Name klicken).
Gimbal-Empfehlung an Mustafa: ~30° off-nadir testen (Rückennummern sichtbar, Homography bleibt ok).

**Phase 4 — Metrics + Report (3-5 Tage)**
Tier-A-Engine (supervision-Primitive: PolygonZone, speed-Rezept, HeatMap), team-interne Perzentile,
LLM-Textgenerierung mit Guardrails, Report-JSON → hoeherr-v1 anbinden.

**Phase 5 — Produktiv-Orchestrierung (3-5 Tage)**
RunPod Serverless Worker-Image, R2, Queue/State-Machine, Retry, Status-Webhooks, Kosten-Telemetrie.

**Phase 6 — Tier B (später)**
Ball-Detektor (SAHI, eigener Fine-Tune), Possession-Modell → Passoptionen/Pressing/Decision-Auto-Tagging.

## 5. Validation-Contract (Done = Assertions grün)

- [ ] ≥85 min Material in <1 h Wall-Clock und <2 € GPU-Kosten verarbeitet
- [ ] ≥95 % der Spieler-Sekunden haben Pitch-Koordinate (nach Stitching)
- [ ] Laufdistanz-Fehler vs. GPS-Uhr-Referenz <5 % (Literatur: 2,36 % erreichbar, PMC9040709)
- [ ] Manuelle Review ≤10 min/Spiel; ID-Switches nach Review = 0
- [ ] Report-JSON validiert gegen Schema und rendert in hoeherr-v1 ohne Code-Änderung
- [ ] Alle 6 Report-Sektionen für ≥1 echten Spieler aus echtem Drohnenmaterial gefüllt

## 6. Risiken

| Risiko | Mitigation |
|---|---|
| Kein echtes Nadir-Material | **Blocker №1** — Testspiel-Aufnahme mit Mustafa terminieren |
| Jersey-OCR ab nadir unmöglich | Off-Nadir-Gimbal + Human-in-the-loop (eingepreist) |
| TeamTrack ≠ unsere Flughöhe/Auflösung | Active-Learning: eigene Frames in Roboflow annotieren, nachtunen |
| Wind/Drohnen-Drift bricht Homography | Keyframe-Refresh + DetectionsSmoother + Ausreißer-Filter |
| Perzentile ohne Liga-Daten unseriös | v1 ehrlich als „vs. Team" labeln |

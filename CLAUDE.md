# höherr — Drohnenbasierte Fußballanalyse

## Projekt-Identität

- **Produkt**: Spieler-Entwicklungsberichte aus Drohnenaufnahmen für Amateurvereine.
  Vision & Report-Format: https://hoeherr-v1.vercel.app (das `MOCK_PLAYER`-Objekt dort
  ist der Daten-Contract — `docs/results/report_player.json` erfüllt ihn bereits).
- **Team**: Mustafa Kaan (Gründer höherr: Drohne, Aufnahmen, Vertrieb, Nutzertests) ·
  Quershift Technologies (Tech: CV-Pipeline, Backend, Infrastruktur).
- **Org/Repo**: `Quershift-Technologies-GmbH/Hoeherr`, Branch `main`.

## Wo der Code wirklich lebt (WICHTIG)

| Bereich | Status |
|---|---|
| `pipeline/` | **Kanonisch & GT-validiert (2026-06-10).** Hier weiterarbeiten. |
| `models/yolov8n_teamtrack_tiles_v0.pt` | Fine-tuned Nadir-Detektor (player+ball), einsatzbereit |
| `src/`, `scripts/`, `configs/` | Trainings-Framework vom April (Experimente v1–v5, Cluster-orientiert). Wertvoll für Phase 2, aber NICHT der validierte Pfad. |
| `api/` | Skeleton, noch nicht angebunden |

## Setup & Kern-Kommandos

```bash
uv venv --python 3.12   # System-Python 3.14 hat keine torch-Wheels!
uv pip install --python .venv/bin/python supervision ultralytics pandas pyarrow scipy gdown

.venv/bin/python pipeline/download_teamtrack.py      # Daten (TeamTrack soccer_top, MIT)
# Kompletter Lauf Video->Report: pipeline/README.md, Abschnitt "Kompletter Lauf"
```

Device ist `mps` (Apple Silicon). Auf NVIDIA: `--device cuda`.

## Validierter Stand (nicht raten — nachmessen!)

GT-Vergleich auf 5 min TeamTrack-Nadir (22 Spieler): Raw-Detection-Recall 0,99 ·
Tracking-Recall 0,92 · RMSE 0,41 m · Median-Distanzfehler 10,6 %
(Details: `docs/results/gt_compare_final.json`). Jede Änderung an Detection/Tracking
MUSS gegen diese Baseline laufen: `pipeline/compare_to_gt.py`. Verschlechterung = Stopp.

## Hart erarbeitete Konventionen (nicht rückgängig machen)

1. **Nadir-Position = Bbox-Zentrum** (`--nadir`-Flag), niemals bottom-center.
2. **COCO-Weights funktionieren NICHT auf Drohnen-Nadir** (0/22 Spieler). Immer die
   fine-getunten Gewichte aus `models/` nutzen bzw. weitertrainieren
   (`pipeline/build_finetune_dataset.py` + ultralytics).
3. **Detections einmal roh speichern (`--raw-out`, conf 0.05), Tracking offline tunen**
   (`pipeline/retrack.py`) — Sekunden statt 25-min-GPU-Schleifen.
4. **Box-Inflation 3.5 für die IoU-Assoziation** ist Absicht (Buffered-IoU): winzige
   schnelle Objekte haben sonst IoU 0 zwischen Frames. Inflation >4 kippt die
   ID-Purity. Getunte v0-Parameter stehen in `pipeline/README.md`.
5. **Homography immer visuell verifizieren** (`overlay_check.png` ansehen!). Feldmaße
   aus Strafraum-Fixmaßen ableiten (`pipeline/auto_calibrate.py`), nicht annehmen.
6. **YouTube-„Drohnen"-Videos sind Schwenk-Kameras** (auch „tactical cam" und
   Veo-Exporte). Für statisches Material: TeamTrack-Datensatz oder eigene Aufnahmen.
7. Heatmap-Format: 24 Zeilen × 16 Spalten, max-normiert — exakt das
   hoeherr-v1-Frontend-Format. Nicht ändern ohne Frontend-Anpassung.
8. Sprint >19,8 km/h, HSR >14,4 km/h, Speed-Cap 38 km/h (Konstanten in
   `pipeline/compute_metrics.py`).

## Nächste Aufgaben (priorisiert)

1. **Phase 2 — Distanzfehler <5 %**: stride 1 (30 Hz), yolov8s-Fine-Tune mit mehr
   TeamTrack-Segmenten (`--frame-step 15`), Velocity-Gating im Stitcher. GPU nötig.
2. **Phase 3 — Review-UI**: Track-Galerie → Spielername zuweisen (ersetzt den
   GT-Relabel-Stand-in in `compare_to_gt.py --relabel-out`). Ziel ≤10 min/Spiel.
3. **Phase 4 — echtes Material**: Testspiel mit Mustafas Drohne aufnehmen
   (Nadir-Hover + ~30° Off-Nadir testen wegen Rückennummern), Active-Learning-Loop.
4. **Tier B — Ball**: Ball-Recall ist aktuell schwach (~35 GT-Ball-Boxen im
   Trainingsset reichen nicht). Mehr Ball-Daten + eigene Ball-Klasse trainieren.

## Do / Don't

- ✅ Vor jedem Merge: `compare_to_gt.py` gegen die Baseline + Zahlen in den PR/Commit.
- ✅ Neue Erkenntnisse in `pipeline/README.md` dokumentieren (lebendes Dokument).
- ❌ Keine Metrik-Behauptungen ohne GT-Messung.
- ❌ `dataset/`, `teamtrack/`, `*.parquet`, Videos nicht committen (siehe .gitignore) —
  Downloads sind reproduzierbar.
- ❌ Tier-B-Features (Passoptionen, Possession, Events) nicht „nebenbei" anfangen —
  erst Tier A auf <5 % bringen (Roadmap: `docs/ROADMAP_FULL_GAME.md`).

## Autonomie- & Risiko-Vertrag

Maßgeblich ist der zentrale Skill `quershift-skills:autonomy-contract` (org-distribuiert via
`quershift-skills@quershift`). Kurzfassung, verbindlich für Main-Loop, Subagents und Hermes:

- **Zwei-Phasen-Modell:** Genau EIN gebündeltes Rückfrage-Fenster sofort nach dem Prompt (nur
  echte, nicht selbst auflösbare Mehrdeutigkeiten) — danach volle Autonomie, null Rückfragen,
  am Ende berichten.
- **Risiko-Affinität:** Handeln schlägt fragen. Fehler im Reversiblen sind gratis und erwartet
  — reinhauen, selbst per Gate (Tests / `verify-gate`) erwischen, fixen, weiter. Sicher nur,
  WEIL das Gate fängt; gegated wird ausschließlich das Irreversible.
- **Kein nacktes A/B:** Gegen das Ziel rankbar → empfehlen UND machen, nicht stehenbleiben.
- **Harte Ausnahmen (immer vorher Freigabe):** E-Mails/externe Nachrichten · vertrauliche Daten
  nach außen · irreversible Infra/Daten (prod-DB-delete, force-push main/shared, Schema-Wipe
  Live-DB, Traffic-100%-auf-ungeprüfte-Revision) · Geld/Verträge.
- **Verifikation Pflicht:** Kein Erfolgsclaim ohne frisch ausgeführten Beleg
  (Exit-Code/Output/Diff). SKIP ist kein PASS.

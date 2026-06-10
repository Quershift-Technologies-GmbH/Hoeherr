#!/usr/bin/env python3
"""
hoeherr Pipeline — Stufe 3: positions.parquet + homography.npz -> metrics.json

Pro Track (Tier A): Distanz, Ø-/Top-Speed, Sprints, High-Speed-Running,
Heatmap-Grid (24x16, wie hoeherr-v1 MOCK_PLAYER), Ø-Position, Spielzeit.
Pro Team (HSV-KMeans über Trikotfarben): Breite/Länge (Kompaktheit), Besetzung.

    .venv/bin/python pipeline/compute_metrics.py --positions positions.parquet \
        --homography homography.npz --out metrics.json
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd

SPRINT_KMH = 19.8        # FIFA-Zone 4/5-Grenze; für Amateure als "Sprint" gelabelt
HSR_KMH = 14.4           # High-Speed-Running-Schwelle
SPEED_CAP_KMH = 38.0     # alles darüber = Homography-/Track-Jitter -> verwerfen
MIN_TRACK_SEC = 5.0      # kürzere Fragmente fliegen aus der Spieler-Statistik
PITCH_MARGIN_M = 2.0     # Punkte außerhalb Feld+Margin = Zuschauer/Betreuer


def project(df: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    pts = df[["x_px", "y_px"]].to_numpy(dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    df = df.copy()
    df["X"] = out[:, 0]
    df["Y"] = out[:, 1]
    return df


def smooth_series(s: pd.Series, win: int = 5) -> pd.Series:
    return s.rolling(win, center=True, min_periods=1).median() \
            .rolling(win, center=True, min_periods=1).mean()


def track_metrics(g: pd.DataFrame, sample_dt: float, L: float, W: float) -> dict | None:
    g = g.sort_values("t_sec")
    dur = g.t_sec.iloc[-1] - g.t_sec.iloc[0]
    if dur < MIN_TRACK_SEC or len(g) < 10:
        return None
    X = smooth_series(g.X)
    Y = smooth_series(g.Y)
    t = g.t_sec.to_numpy()
    dx = np.diff(X)
    dy = np.diff(Y)
    dt = np.diff(t)
    seg = np.hypot(dx, dy)
    valid = (dt > 0) & (dt <= 3.5 * sample_dt)          # Lücken nicht interpolieren
    speed_kmh = np.where(valid, seg / np.where(dt == 0, np.nan, dt) * 3.6, np.nan)
    speed_kmh = np.where(speed_kmh <= SPEED_CAP_KMH, speed_kmh, np.nan)
    dist = float(np.nansum(np.where(np.isfinite(speed_kmh), seg, 0.0)))

    # Top-Speed über ~1s-Fenster glätten gegen Einzel-Frame-Spikes
    win = max(1, int(round(1.0 / sample_dt)))
    sp = pd.Series(speed_kmh).rolling(win, min_periods=max(1, win // 2)).mean()
    top_speed = float(np.nanmax(sp)) if np.isfinite(sp).any() else 0.0

    # Sprints: zusammenhängende Phasen >= SPRINT_KMH über >= 0.8s
    sprints = 0
    run = 0.0
    for v, d in zip(speed_kmh, dt):
        if np.isfinite(v) and v >= SPRINT_KMH:
            run += d
        else:
            if run >= 0.8:
                sprints += 1
            run = 0.0
    if run >= 0.8:
        sprints += 1
    hsr_sec = float(np.nansum(np.where(np.isfinite(speed_kmh) & (speed_kmh >= HSR_KMH),
                                       dt, 0.0)))

    # Heatmap 24 rows x 16 cols (hoeherr-v1-Konvention), normiert auf max=1
    gx = np.clip((X / L * 16).astype(int), 0, 15)
    gy = np.clip((Y / W * 24).astype(int), 0, 23)
    grid = np.zeros((24, 16))
    np.add.at(grid, (gy, gx), 1.0)
    if grid.max() > 0:
        grid = grid / grid.max()

    return {
        "duration_sec": round(float(dur), 1),
        "n_samples": int(len(g)),
        "distance_m": round(dist, 1),
        "avg_speed_kmh": round(dist / dur * 3.6, 2) if dur > 0 else 0.0,
        "top_speed_kmh": round(top_speed, 1),
        "sprints": sprints,
        "hsr_sec": round(hsr_sec, 1),
        "avg_position": {"x": round(float(X.mean() / L), 3),
                         "y": round(float(Y.mean() / W), 3)},
        "heatmap_grid": np.round(grid, 3).tolist(),
        "jersey_hsv": [round(float(v), 1) if np.isfinite(v) else None
                       for v in (g.hue.median(), g.sat.median(), g.val.median())],
    }


def cluster_teams(track_info: dict) -> dict:
    """KMeans k=3 über Trikot-HSV (Hue zirkulär kodiert). 2 größte Cluster = Teams.
    GT-Fallback: ohne Farben (NaN) steckt das Team in der ID (team*100+player)."""
    tids = list(track_info)
    if all(track_info[t]["jersey_hsv"][0] is None for t in tids):
        return {t: ("team_a" if t < 100 else "team_b") for t in tids}
    feats = []
    for tid in tids:
        h, s, v = track_info[tid]["jersey_hsv"]
        ang = h / 180.0 * 2 * np.pi
        feats.append([np.cos(ang) * s, np.sin(ang) * s, v])
    feats = np.array(feats, dtype=np.float32)
    k = min(3, len(tids))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, _ = cv2.kmeans(feats, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    sizes = np.bincount(labels, minlength=k)
    team_clusters = np.argsort(sizes)[::-1][:2]
    mapping = {}
    for tid, lab in zip(tids, labels):
        if lab == team_clusters[0]:
            mapping[tid] = "team_a"
        elif len(team_clusters) > 1 and lab == team_clusters[1]:
            mapping[tid] = "team_b"
        else:
            mapping[tid] = "other"   # Schiri / Keeper / Ausreißer
    return mapping


def team_shape(df: pd.DataFrame, tids: list[int]) -> dict:
    sub = df[df.tracker_id.isin(tids)]
    if sub.empty:
        return {}
    per_frame = sub.groupby("frame").agg(
        length=("X", lambda v: v.max() - v.min()),
        width=("Y", lambda v: v.max() - v.min()),
        n=("X", "size"))
    pf = per_frame[per_frame.n >= 5]
    if pf.empty:
        return {}
    return {"avg_length_m": round(float(pf.length.mean()), 1),
            "avg_width_m": round(float(pf.width.mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="positions.parquet")
    ap.add_argument("--homography", default="homography.npz")
    ap.add_argument("--out", default="metrics.json")
    args = ap.parse_args()

    hom = np.load(args.homography)
    H = hom["H"]
    L = float(hom["pitch_length"])
    W = float(hom["pitch_width"])

    df = pd.read_parquet(args.positions)
    n_raw = len(df)
    df = project(df, H)
    on_pitch = ((df.X >= -PITCH_MARGIN_M) & (df.X <= L + PITCH_MARGIN_M) &
                (df.Y >= -PITCH_MARGIN_M) & (df.Y <= W + PITCH_MARGIN_M))
    df = df[on_pitch]
    print(f"[filter] {n_raw} -> {len(df)} Samples auf dem Feld "
          f"({n_raw - len(df)} außerhalb verworfen)")

    ts = np.sort(df.t_sec.unique())
    sample_dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.1
    total_sec = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

    tracks = {}
    for tid, g in df.groupby("tracker_id"):
        m = track_metrics(g, sample_dt, L, W)
        if m is not None:
            tracks[int(tid)] = m
    print(f"[tracks] {df.tracker_id.nunique()} roh, {len(tracks)} >= {MIN_TRACK_SEC}s")

    team_of = cluster_teams(tracks) if tracks else {}
    for tid in tracks:
        tracks[tid]["team"] = team_of.get(tid, "other")

    teams = {}
    for team in ("team_a", "team_b"):
        tids = [tid for tid in tracks if tracks[tid]["team"] == team]
        teams[team] = {
            "n_tracks": len(tids),
            "total_distance_m": round(sum(tracks[t]["distance_m"] for t in tids), 1),
            **team_shape(df, tids),
        }

    out = {
        "meta": {
            "source_samples": n_raw,
            "clip_seconds": round(total_sec, 1),
            "sample_dt_sec": round(sample_dt, 4),
            "pitch": {"length_m": L, "width_m": W},
            "thresholds": {"sprint_kmh": SPRINT_KMH, "hsr_kmh": HSR_KMH,
                           "speed_cap_kmh": SPEED_CAP_KMH},
            "raw_track_count": int(df.tracker_id.nunique()),
            "kept_track_count": len(tracks),
        },
        "teams": teams,
        "tracks": tracks,
    }
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    # Kompakte Konsolen-Zusammenfassung
    print(f"\n=== METRIKEN ({total_sec:.0f}s Clip) ===")
    for team in ("team_a", "team_b"):
        t = teams[team]
        print(f"{team}: {t['n_tracks']} Tracks, {t['total_distance_m']:.0f} m gesamt, "
              f"Form {t.get('avg_length_m', '?')}x{t.get('avg_width_m', '?')} m")
    top = sorted(tracks.items(), key=lambda kv: -kv[1]["distance_m"])[:8]
    for tid, m in top:
        print(f"  #{tid:<4d} {m['team']:<7s} {m['duration_sec']:6.1f}s "
              f"{m['distance_m']:7.1f} m  top {m['top_speed_kmh']:4.1f} km/h  "
              f"sprints {m['sprints']}")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()

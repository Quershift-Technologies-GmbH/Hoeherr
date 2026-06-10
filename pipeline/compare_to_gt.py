#!/usr/bin/env python3
"""
CV-Pipeline gegen TeamTrack-GT messen (beides in Pitch-Metern):
- Detection-Recall/Precision (Match: CV-Punkt <2m an GT-Spieler, greedy)
- Track-Fragmentierung (CV-Tracks pro GT-Spieler) + ID-Purity
- Positions-RMSE auf gematchten Paaren
- Distanz pro GT-Spieler: GT vs. Summe zugeordneter CV-Tracks + Coverage

    .venv/bin/python pipeline/compare_to_gt.py --cv positions_cv.parquet \
        --gt positions_gt.parquet --homography homography.npz --out gt_compare.json
"""
import argparse
import json
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd

MATCH_DIST_M = 2.0


def project(df, H):
    pts = df[["x_px", "y_px"]].to_numpy(np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    df = df.copy()
    df["X"], df["Y"] = out[:, 0], out[:, 1]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--homography", default="homography.npz")
    ap.add_argument("--out", default="gt_compare.json")
    ap.add_argument("--relabel-out", default="",
                    help="CV-Positionen mit konsolidierter Spieler-ID speichern "
                         "(Track->Spieler via Majority-Vote; Stand-in für die "
                         "Human-in-the-loop-Review-UI aus Phase 3)")
    args = ap.parse_args()

    hom = np.load(args.homography)
    H = hom["H"]
    L, W = float(hom["pitch_length"]), float(hom["pitch_width"])
    cv_df = project(pd.read_parquet(args.cv), H)
    gt_df = project(pd.read_parquet(args.gt), H)
    cv_df = cv_df[(cv_df.X > -2) & (cv_df.X < L + 2) & (cv_df.Y > -2) & (cv_df.Y < W + 2)]

    frames = sorted(set(cv_df.frame) & set(gt_df.frame))
    print(f"[frames] gemeinsam: {len(frames)}")
    cv_g = dict(tuple(cv_df.groupby("frame")))
    gt_g = dict(tuple(gt_df.groupby("frame")))

    n_gt = n_cv = n_match = 0
    sq_err = []
    pair_votes = defaultdict(Counter)   # cv_tid -> Counter(gt_tid)
    matched_pairs = []                  # (frame, cv_tid, gt_tid, dist)
    for fr in frames:
        a = cv_g[fr]
        b = gt_g[fr]
        n_cv += len(a)
        n_gt += len(b)
        # greedy nearest matching
        dmat = np.linalg.norm(
            a[["X", "Y"]].to_numpy()[:, None, :] - b[["X", "Y"]].to_numpy()[None, :, :],
            axis=2)
        used_a, used_b = set(), set()
        for _ in range(min(len(a), len(b))):
            i, j = np.unravel_index(np.argmin(dmat), dmat.shape)
            if dmat[i, j] > MATCH_DIST_M:
                break
            n_match += 1
            sq_err.append(dmat[i, j] ** 2)
            cv_tid = int(a.iloc[i].tracker_id)
            gt_tid = int(b.iloc[j].tracker_id)
            pair_votes[cv_tid][gt_tid] += 1
            matched_pairs.append((fr, cv_tid, gt_tid))
            dmat[i, :] = np.inf
            dmat[:, j] = np.inf

    recall = n_match / n_gt if n_gt else 0
    precision = n_match / n_cv if n_cv else 0
    rmse = float(np.sqrt(np.mean(sq_err))) if sq_err else None

    # Track -> GT-Spieler (Majority Vote), Purity, Fragmentierung
    cv2gt = {}
    purity_w = []
    for cv_tid, votes in pair_votes.items():
        gt_tid, n_top = votes.most_common(1)[0]
        cv2gt[cv_tid] = gt_tid
        purity_w.append((n_top / sum(votes.values()), sum(votes.values())))
    purity = (sum(p * w for p, w in purity_w) / sum(w for _, w in purity_w)
              if purity_w else 0)
    frags = Counter(cv2gt.values())

    # Distanz-Vergleich pro GT-Spieler (einfach: Summe Streckenlängen, smoothed)
    def track_dist(g):
        g = g.sort_values("t_sec")
        x = g.X.rolling(5, center=True, min_periods=1).median()
        y = g.Y.rolling(5, center=True, min_periods=1).median()
        dt = np.diff(g.t_sec)
        seg = np.hypot(np.diff(x), np.diff(y))
        ok = (dt > 0) & (dt < 1.0) & (seg / np.maximum(dt, 1e-6) * 3.6 < 38)
        return float(seg[ok].sum())

    gt_dist = {int(t): track_dist(g) for t, g in gt_df.groupby("tracker_id")}
    cv_dist_per_gt = defaultdict(float)
    cv_samples_per_gt = defaultdict(int)
    for cv_tid, g in cv_df.groupby("tracker_id"):
        if int(cv_tid) in cv2gt:
            gt_tid = cv2gt[int(cv_tid)]
            cv_dist_per_gt[gt_tid] += track_dist(g)
            cv_samples_per_gt[gt_tid] += len(g)
    gt_samples = gt_df.groupby("tracker_id").size().to_dict()

    per_player = []
    for gt_tid in sorted(gt_dist):
        cov = cv_samples_per_gt.get(gt_tid, 0) / gt_samples[int(gt_tid)]
        d_gt = gt_dist[gt_tid]
        d_cv = cv_dist_per_gt.get(gt_tid, 0.0)
        per_player.append({
            "gt_player": gt_tid, "coverage": round(cov, 3),
            "gt_dist_m": round(d_gt, 1), "cv_dist_m": round(d_cv, 1),
            "dist_err_pct": round((d_cv - d_gt) / d_gt * 100, 1) if d_gt else None,
            "cv_tracks": int(frags.get(gt_tid, 0)),
        })

    covered = [p for p in per_player if p["coverage"] > 0.5]
    summary = {
        "frames": len(frames),
        "detection_recall": round(recall, 4),
        "detection_precision": round(precision, 4),
        "position_rmse_m": round(rmse, 3) if rmse else None,
        "id_purity": round(purity, 4),
        "cv_tracks_total": len(pair_votes),
        "gt_players": len(gt_dist),
        "avg_cv_tracks_per_player": round(np.mean(list(frags.values())), 2) if frags else 0,
        "median_abs_dist_err_pct_cov50": (round(float(np.median(
            [abs(p["dist_err_pct"]) for p in covered if p["dist_err_pct"] is not None])), 1)
            if covered else None),
    }
    json.dump({"summary": summary, "per_player": per_player},
              open(args.out, "w"), indent=1)

    if args.relabel_out:
        lab = cv_df[cv_df.tracker_id.isin(cv2gt)].copy()
        lab["tracker_id"] = lab.tracker_id.map(lambda t: cv2gt[int(t)])
        lab.drop(columns=["X", "Y"]).to_parquet(args.relabel_out, index=False)
        print(f"[relabel] {lab.tracker_id.nunique()} Spieler -> {args.relabel_out}")
    print("\n=== GT-VERGLEICH ===")
    for k, v in summary.items():
        print(f"  {k:32s} {v}")
    print(f"\n  {'GT':>4s} {'cov':>6s} {'gt_m':>8s} {'cv_m':>8s} {'err%':>7s} {'#trk':>5s}")
    for p in per_player:
        print(f"  {p['gt_player']:4d} {p['coverage']:6.0%} {p['gt_dist_m']:8.1f} "
              f"{p['cv_dist_m']:8.1f} {str(p['dist_err_pct']):>7s} {p['cv_tracks']:5d}")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()

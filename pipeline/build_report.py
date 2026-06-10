#!/usr/bin/env python3
"""
metrics.json -> Spieler-Report-JSON im hoeherr-v1-Format (MOCK_PLAYER-Schema).

Tier-A-Felder werden echt befüllt (Distanz, Sprints, Top-Speed, Heatmap,
Ø-Position, Mitspieler-Distanzen). Tier-B-Felder (Passoptionen, Entscheidungs-
qualität, Pressing) bleiben null und sind in meta.not_computed gelistet.
Perzentile = Rang innerhalb aller Feldspieler-Tracks dieses Clips (ehrlich
gelabelt als "vs. Spiel", nicht "vs. Liga").

    .venv/bin/python pipeline/build_report.py --metrics metrics.json \
        --track best --name "Spieler 4" --out report_player.json
"""
import argparse
import json

import numpy as np


def pct_rank(value, population):
    pop = sorted(population)
    if not pop:
        return None
    return int(round(100 * sum(1 for v in pop if v <= value) / len(pop)))


def zone_insight(grid, avg):
    g = np.array(grid)
    thirds = [g[:, :5].sum(), g[:, 5:11].sum(), g[:, 11:].sum()]
    names = ["im eigenen Drittel", "im Mittelfeld", "im letzten Drittel"]
    dom = names[int(np.argmax(thirds))]
    half = "linke" if avg["y"] < 0.45 else ("rechte" if avg["y"] > 0.55 else "zentrale")
    share = max(thirds) / max(g.sum(), 1e-9)
    return (f"Aktivitätsschwerpunkt {dom} ({share:.0%} der Samples), "
            f"überwiegend {half} Spielfeldhälfte.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics.json")
    ap.add_argument("--track", default="best",
                    help="Track-ID oder 'best' (längste Dauer, dann meiste Distanz)")
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="report_player.json")
    args = ap.parse_args()

    M = json.load(open(args.metrics))
    tracks = {int(k): v for k, v in M["tracks"].items()}
    field = {k: v for k, v in tracks.items() if v["team"] in ("team_a", "team_b")}

    if args.track == "best":
        tid = max(field, key=lambda k: (field[k]["duration_sec"], field[k]["distance_m"]))
    else:
        tid = int(args.track)
    me = tracks[tid]
    team = me["team"]
    peers = {k: v for k, v in field.items() if v["team"] == team and k != tid}

    pop_dist = [v["distance_m"] for v in field.values()]
    pop_sprint = [v["sprints"] for v in field.values()]
    pop_speed = [v["top_speed_kmh"] for v in field.values()]

    # Mitspieler-Distanzen: euklid. Abstand der Ø-Positionen (Meter), 4 nächste
    L = M["meta"]["pitch"]["length_m"]
    W = M["meta"]["pitch"]["width_m"]
    mx, my = me["avg_position"]["x"] * L, me["avg_position"]["y"] * W
    dists = sorted(
        (round(float(np.hypot(v["avg_position"]["x"] * L - mx,
                              v["avg_position"]["y"] * W - my)), 1), k)
        for k, v in peers.items())[:4]

    minutes = me["duration_sec"] / 60.0
    report = {
        "schema": "hoeherr-v1 MOCK_PLAYER (Tier-A-Subset)",
        "meta": {
            "source": "supervision-Pipeline v0 (lokal, MPS)",
            "clip_seconds": M["meta"]["clip_seconds"],
            "percentile_basis": "vs. alle Feldspieler dieses Clips (NICHT Liga)",
            "not_computed_tier_b": ["passOptions", "decisionQuality",
                                    "pressingTriggers", "positionDiscipline",
                                    "spaceGain", "decisions.timeline"],
        },
        "jersey": None,
        "displayName": args.name or f"Track #{tid}",
        "position": None,
        "match": {"minutesPlayed": round(minutes, 1),
                  "competition": "TeamTrack-Demo (Drohne, Nadir)"},
        "score": {"overall": None},
        "stats": {
            "distance": {"value": round(me["distance_m"] / 1000, 2), "unit": "km",
                         "percentile": pct_rank(me["distance_m"], pop_dist),
                         "label": "Laufdistanz"},
            "sprints": {"value": me["sprints"], "unit": "x",
                        "percentile": pct_rank(me["sprints"], pop_sprint),
                        "label": "Sprints"},
            "topSpeed": {"value": me["top_speed_kmh"], "unit": "km/h",
                         "percentile": pct_rank(me["top_speed_kmh"], pop_speed),
                         "label": "Top-Speed"},
            "hsrSeconds": {"value": me["hsr_sec"], "unit": "s",
                           "percentile": None, "label": "High-Speed-Running"},
        },
        "heatmap": {
            "grid": me["heatmap_grid"],
            "centroid": me["avg_position"],
            "insight": zone_insight(me["heatmap_grid"], me["avg_position"]),
        },
        "positioning": {
            "avgPosition": me["avg_position"],
            "teammateDistances": [
                {"teammate": f"Track #{k}", "meters": d,
                 "status": "optimal" if d <= 25 else "zu weit"}
                for d, k in dists],
        },
        "team_context": M["teams"][team],
    }
    json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"[done] Report für Track #{tid} ({team}, {minutes:.1f} min, "
          f"{me['distance_m']:.0f} m) -> {args.out}")


if __name__ == "__main__":
    main()

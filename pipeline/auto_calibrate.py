#!/usr/bin/env python3
"""
Auto-Kalibrierung für statisches Nadir-Material:
1. Robuste Linien-Fits auf die 4 Außenlinien (brightest-pixel sampling + Median-Filter)
2. Schnittpunkte = Feldecken (px)
3. Feldmaße aus Fixmaßen ableiten (Strafraum 16,5m tief, 40,32m breit)
4. points.json schreiben -> danach calibrate_pitch.py laufen lassen

    .venv/bin/python pipeline/auto_calibrate.py --frame tt_frame0.png --out points.json
"""
import argparse
import json

import cv2
import numpy as np


def fit_line_in_band(gray, axis, band_lo, band_hi, sample_range, step=100,
                     min_bright=120):
    """Sucht hellsten Pixel quer zur Linie, robuster Polyfit Grad 1.
    axis='h': horizontale Linie, sample über x, y in [band_lo, band_hi].
    axis='v': vertikale Linie, sample über y, x in [band_lo, band_hi]."""
    ss, ps = [], []
    for s in range(*sample_range, step):
        if axis == "h":
            profile = gray[band_lo:band_hi, s - 2:s + 3].mean(axis=1)
        else:
            profile = gray[s - 2:s + 3, band_lo:band_hi].mean(axis=0)
        if profile.max() < min_bright:
            continue
        ss.append(s)
        ps.append(int(np.argmax(profile)) + band_lo)
    ss, ps = np.array(ss), np.array(ps, dtype=float)
    # Outlier raus (Spieler/Text auf der Linie): 2 Runden Median-Abstand
    for _ in range(2):
        if len(ss) < 4:
            break
        coef = np.polyfit(ss, ps, 1)
        resid = np.abs(np.polyval(coef, ss) - ps)
        keep = resid < max(6.0, np.median(resid) * 3)
        ss, ps = ss[keep], ps[keep]
    coef = np.polyfit(ss, ps, 1)
    return coef, len(ss)  # p = coef[0]*s + coef[1]


def intersect(h_coef, v_coef):
    """Schnitt horizontale Linie (y=a*x+b) mit vertikaler (x=c*y+d)."""
    a, b = h_coef
    c, d = v_coef
    y = (a * d + b) / (1 - a * c)
    x = c * y + d
    return float(x), float(y)


def measure_v_line(gray, x_center, y_lo, y_hi, halfwidth=80):
    """Vertikale weiße Linie nahe x_center präzise lokalisieren (Median über Zeilen)."""
    xs = []
    for y in range(y_lo, y_hi, 40):
        row = gray[y - 1:y + 2, x_center - halfwidth:x_center + halfwidth].mean(axis=0)
        if row.max() > 120:
            xs.append(int(np.argmax(row)) + x_center - halfwidth)
    return float(np.median(xs)) if xs else None


def measure_h_line(gray, y_center, x_lo, x_hi, halfwidth=80):
    ys = []
    for x in range(x_lo, x_hi, 40):
        col = gray[y_center - halfwidth:y_center + halfwidth, x - 1:x + 2].mean(axis=1)
        if col.max() > 120:
            ys.append(int(np.argmax(col)) + y_center - halfwidth)
    return float(np.median(ys)) if ys else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", default="points.json")
    args = ap.parse_args()

    gray = cv2.cvtColor(cv2.imread(args.frame), cv2.COLOR_BGR2GRAY)
    Hpx, Wpx = gray.shape

    top, n1 = fit_line_in_band(gray, "h", 60, 280, (500, 3400))
    bot, n2 = fit_line_in_band(gray, "h", 1920, 2110, (500, 3400))
    left, n3 = fit_line_in_band(gray, "v", 180, 460, (300, 1900))
    right, n4 = fit_line_in_band(gray, "v", 3400, 3700, (300, 1900))
    print(f"[lines] top n={n1} slope={top[0]:.4f} | bot n={n2} slope={bot[0]:.4f} | "
          f"left n={n3} | right n={n4}")

    TL = intersect(top, left)
    TR = intersect(top, right)
    BR = intersect(bot, right)
    BL = intersect(bot, left)
    print(f"[corners] TL={TL[0]:.0f},{TL[1]:.0f} TR={TR[0]:.0f},{TR[1]:.0f} "
          f"BR={BR[0]:.0f},{BR[1]:.0f} BL={BL[0]:.0f},{BL[1]:.0f}")

    # Vorläufige Homography mit Norm-Maßen, dann Maße aus Fixmaßen korrigieren
    L0, W0 = 105.0, 68.0
    src = np.array([TL, TR, BR, BL])
    dst = np.array([[0, 0], [L0, 0], [L0, W0], [0, W0]])
    H0, _ = cv2.findHomography(src, dst)

    def to_pitch(x, y):
        p = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), H0)
        return p[0, 0]

    mid_y = int((TL[1] + BL[1]) / 2)
    # Linker Strafraum: vertikale Linie ~16,5m vom linken Tor; Suchfenster über px-Schätzung
    px_per_m_x0 = (TR[0] - TL[0]) / L0
    box_x_guess = int(TL[0] + 16.5 * px_per_m_x0)
    box_left_px = measure_v_line(gray, box_x_guess, mid_y - 500, mid_y + 500)
    box_right_px = measure_v_line(gray, int(TR[0] - 16.5 * px_per_m_x0),
                                  mid_y - 500, mid_y + 500)
    # Strafraum-Querlinien (oben/unten), links gemessen in x-Fenster innerhalb des Strafraums
    px_per_m_y0 = (BL[1] - TL[1]) / W0
    boxtop_y_guess = int(TL[1] + (W0 / 2 - 20.16) * px_per_m_y0)
    boxbot_y_guess = int(TL[1] + (W0 / 2 + 20.16) * px_per_m_y0)
    x_in_box = int(TL[0] + 8 * px_per_m_x0)
    box_top_px = measure_h_line(gray, boxtop_y_guess, x_in_box, x_in_box + 300)
    box_bot_px = measure_h_line(gray, boxbot_y_guess, x_in_box, x_in_box + 300)

    L_real, W_real = L0, W0
    if box_left_px and box_right_px:
        X_meas_l = to_pitch(box_left_px, mid_y)[0]
        X_meas_r = L0 - to_pitch(box_right_px, mid_y)[0]
        X_meas = (X_meas_l + X_meas_r) / 2
        L_real = L0 * 16.5 / X_meas
        print(f"[scale] Strafraumtiefe gemessen {X_meas_l:.2f}/{X_meas_r:.2f} "
              f"(soll 16.5) -> L = {L_real:.1f} m")
    if box_top_px and box_bot_px:
        y_top = to_pitch(x_in_box, box_top_px)[1]
        y_bot = to_pitch(x_in_box, box_bot_px)[1]
        W_real = W0 * 40.32 / (y_bot - y_top)
        print(f"[scale] Strafraumbreite gemessen {y_bot - y_top:.2f} "
              f"(soll 40.32) -> W = {W_real:.1f} m")
    L_real, W_real = round(L_real, 1), round(W_real, 1)

    pts = {
        "pitch_length": L_real,
        "pitch_width": W_real,
        "points": [
            {"px": list(TL), "pitch": [0, 0]},
            {"px": list(TR), "pitch": [L_real, 0]},
            {"px": list(BR), "pitch": [L_real, W_real]},
            {"px": list(BL), "pitch": [0, W_real]},
        ],
    }
    json.dump(pts, open(args.out, "w"), indent=2)
    print(f"[done] {args.out}: Feld {L_real} x {W_real} m")


if __name__ == "__main__":
    main()

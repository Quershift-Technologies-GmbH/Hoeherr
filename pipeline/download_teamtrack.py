#!/usr/bin/env python3
"""TeamTrack soccer_top Segmente von Google Drive ziehen (Name -> ID via Folder-Parse)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import requests

FOLDERS = {  # teamtrack/soccer_top/train/
    "videos": "1rwJUqYIfBHgTFHjgHOmAl6vqpsMptxYW",
    "annotations": "1dPvWYMoOKtdQqyg0mfrwUZzeHwaVEfS5",
}
# Pipeline-Run: Minuten 0-5. Fine-Tune-Train: Minuten 10-15 (disjunkte Spielphasen).
PIPELINE = [f"D_20220220_1_{s:04d}_{s+30:04d}" for s in range(0, 300, 30)]
TRAIN = [f"D_20220220_1_{s:04d}_{s+30:04d}" for s in (600, 660, 720, 780)]
VAL = ["D_20220220_1_0900_0930"]
WANTED = set(PIPELINE + TRAIN + VAL)


def list_folder(fid):
    txt = requests.get(f"https://drive.google.com/drive/folders/{fid}", timeout=60).text
    m = re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'(.+?)';", txt, re.S)
    data = json.loads(m.group(1).encode().decode("unicode_escape"))
    found = {}

    def walk(node):
        if isinstance(node, list):
            if (len(node) > 3 and isinstance(node[0], str) and len(node[0]) > 20
                    and isinstance(node[2], str) and isinstance(node[3], str)
                    and "/" in str(node[3])):
                found[node[2]] = node[0]
            else:
                for ch in node:
                    walk(ch)

    walk(data)
    return found


def main():
    base = Path(__file__).resolve().parent.parent / "teamtrack"
    gdown = Path(sys.executable).parent / "gdown"
    for sub, fid in FOLDERS.items():
        files = list_folder(fid)
        ext = ".mp4" if sub == "videos" else ".csv"
        outdir = base / sub
        outdir.mkdir(parents=True, exist_ok=True)
        todo = [(stem + ext, files.get(stem + ext)) for stem in sorted(WANTED)]
        for name, gid in todo:
            dest = outdir / name
            if gid is None:
                print(f"[MISS] {name} nicht im Drive-Folder!")
                continue
            if dest.exists() and dest.stat().st_size > 1000:
                print(f"[skip] {name}")
                continue
            print(f"[get ] {name} ({gid})", flush=True)
            subprocess.run([str(gdown), "-q", "-O", str(dest), gid], check=True)
    print("[done] alle Downloads")


if __name__ == "__main__":
    main()

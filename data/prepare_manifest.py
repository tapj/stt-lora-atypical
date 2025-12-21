import argparse
import csv
import os
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def find_pairs(pairs_dir: Path):
    audio_files = []
    for p in pairs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            audio_files.append(p)

    rows = []
    missing_txt = 0
    for a in sorted(audio_files):
        txt = a.with_suffix(".txt")
        if not txt.exists():
            missing_txt += 1
            continue
        transcript = txt.read_text(encoding="utf-8").strip()
        if not transcript:
            continue
        rows.append((str(a), transcript, "", ""))  # language, speaker optional
    return rows, missing_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", type=str, required=True, help="Folder containing audio + .txt pairs")
    ap.add_argument("--out_csv", type=str, required=True, help="Output manifest CSV")
    args = ap.parse_args()

    pairs_dir = Path(args.pairs_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows, missing_txt = find_pairs(pairs_dir)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "transcript", "language", "speaker"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    if missing_txt:
        print(f"Warning: {missing_txt} audio files had no matching .txt")


if __name__ == "__main__":
    main()

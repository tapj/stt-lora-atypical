import argparse
import csv
from pathlib import Path

import torch
import torchaudio
from datasets import Audio, load_dataset


def write_manifest(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "transcript", "language", "speaker"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Download a tiny speech dataset for smoke tests.")
    ap.add_argument("--out_dir", type=str, default="data/example_pairs", help="Where to store downloaded WAV files")
    ap.add_argument(
        "--manifest",
        type=str,
        default="data/example_manifest.csv",
        help="Where to write the manifest CSV pointing at the downloaded files",
    )
    ap.add_argument("--num_samples", type=int, default=4, help="How many samples to save from the dummy dataset")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)

    # Tiny HF dataset with very small clips; keeps download/time minimal.
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    rows = []
    for i, row in enumerate(ds.select(range(min(args.num_samples, len(ds))))):
        audio = row["audio"]
        transcript = (row.get("text") or "").strip()
        if not transcript:
            continue

        wav_path = out_dir / f"sample_{i:03d}.wav"
        tensor = torch.tensor(audio["array"]).unsqueeze(0)
        torchaudio.save(str(wav_path), tensor, sample_rate=audio["sampling_rate"])

        language = row.get("language", "")
        rows.append((str(wav_path), transcript, language, ""))  # speaker optional

    write_manifest(rows, manifest_path)

    print(f"Wrote {len(rows)} samples to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

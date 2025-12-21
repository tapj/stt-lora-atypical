import argparse
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from datasets.exceptions import DatasetNotFoundError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mozilla-foundation/common_voice_11_0")
    ap.add_argument("--config", type=str, default="fr")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out_dir", type=str, default="demo_pairs_fr")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(args.dataset, args.config, split=args.split)
    except DatasetNotFoundError as e:
        raise SystemExit(
            "Dataset not found. Check the dataset name/version on the Hugging Face Hub "
            "and ensure you have network access or credentials if required."
        ) from e
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    n = min(args.n, len(ds))
    for i in range(n):
        ex = ds[i]
        wav = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        text = (ex.get("sentence") or ex.get("text") or "").strip()
        if not text:
            continue
        wav_path = out_dir / f"sample_{i:05d}.wav"
        txt_path = out_dir / f"sample_{i:05d}.txt"
        sf.write(wav_path, wav, sr)
        txt_path.write_text(text, encoding="utf-8")

    print(f"Saved pairs to: {out_dir}")


if __name__ == "__main__":
    main()

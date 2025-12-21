import argparse
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from datasets.exceptions import DatasetNotFoundError


TEXT_CANDIDATES = ["sentence", "text", "transcript", "normalized_text"]


def pick_text(example: dict) -> str:
    for k in TEXT_CANDIDATES:
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def try_load(dataset_id: str, config: str, split: str, trust_remote_code: bool):
    return load_dataset(
        dataset_id,
        config,
        split=split,
        trust_remote_code=trust_remote_code,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="fsicoli/common_voice_22_0",
        help="HF dataset id. Examples: fsicoli/common_voice_22_0, fsicoli/common_voice_11_0",
    )
    ap.add_argument("--config", type=str, default="fr", help="Language config, e.g. fr")
    ap.add_argument("--split", type=str, default="train", help="train/validation/test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="data/example_pairs")
    ap.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Needed for some Common Voice mirrors (recommended).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fallback list if the provided dataset id fails.
    candidates = [
        args.dataset,
        "fsicoli/common_voice_22_0",
        "fsicoli/common_voice_11_0",
        "malaysia-ai/common_voice_22_0",
        "malaysia-ai/common_voice_11_0",
    ]

    ds = None
    last_err = None
    for ds_id in candidates:
        try:
            ds = try_load(ds_id, args.config, args.split, trust_remote_code=True if args.trust_remote_code else False)
            print(f"Loaded dataset: {ds_id} ({args.config}, split={args.split})")
            break
        except Exception as e:
            last_err = e
            continue

    if ds is None:
        raise SystemExit(
            "Impossible de charger le dataset.\n"
            f"IDs testés: {candidates}\n"
            f"Dernière erreur: {repr(last_err)}\n\n"
            "Actions:\n"
            "1) Vérifie l'accès réseau vers Hugging Face.\n"
            "2) Essaie avec --trust_remote_code.\n"
            "3) Change --dataset vers un miroir existant."
        )

    # Column name can vary; Common Voice usually uses "audio"
    if "audio" not in ds.column_names:
        raise SystemExit(f"Colonne 'audio' introuvable. Colonnes disponibles: {ds.column_names}")

    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    saved = 0
    target = int(args.n)

    for i in range(len(ds)):
        if saved >= target:
            break

        ex = ds[i]
        text = pick_text(ex)
        if not text:
            continue

        audio = ex["audio"]
        wav = audio["array"]
        sr = int(audio["sampling_rate"])

        wav_path = out_dir / f"sample_{saved:05d}.wav"
        txt_path = out_dir / f"sample_{saved:05d}.txt"

        sf.write(wav_path, wav, sr)
        txt_path.write_text(text, encoding="utf-8")
        saved += 1

    if saved == 0:
        raise SystemExit(
            "Aucun exemple sauvegardé. Probable cause: champ texte vide ou mauvais split/config.\n"
            f"Colonnes disponibles: {ds.column_names}"
        )

    print(f"Saved {saved} pairs to: {out_dir}")


if __name__ == "__main__":
    main()

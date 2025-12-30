import csv
import os
from pathlib import Path


def resolve_manifest_paths(manifest_csv: str) -> str:
    """
    Return a sibling manifest whose audio paths are absolute.

    The output file is named `<stem>.resolved.csv` next to the input manifest.
    Relative paths are resolved against the parent directory of `manifest_csv`.
    """
    src = Path(manifest_csv)
    out = src.with_name(f"{src.stem}.resolved.csv")
    base = src.parent

    with src.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError(f"CSV must contain column 'path'. Found: {reader.fieldnames}")

        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            path_val = (row.get("path") or "").strip()
            if not path_val:
                continue
            if not os.path.isabs(path_val):
                row["path"] = str((base / path_val).resolve())
            writer.writerow(row)

    return str(out)

import sys
import types
from pathlib import Path

import pytest

# Provide a lightweight stub for soundfile to avoid the optional dependency.
if "soundfile" not in sys.modules:
    dummy_sf = types.ModuleType("soundfile")
    dummy_sf.written = []
    dummy_sf.__spec__ = types.SimpleNamespace()  # Needed for datasets import checks
    dummy_sf.__libsndfile_version__ = "0.0.0"

    def _write(path, data, sr):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        dummy_sf.written.append((p, data, sr))

    dummy_sf.write = _write
    sys.modules["soundfile"] = dummy_sf

# Import after stubbing soundfile
from data.save_hf_subset_pairs import main, pick_text


class DummyAudio:
    def __init__(self, array, sampling_rate=16000):
        self.array = array
        self.sampling_rate = sampling_rate


class DummyDataset:
    def __init__(self, examples, column_names=None):
        self.examples = examples
        self.column_names = column_names or ["audio", "text"]

    def cast_column(self, *args, **kwargs):
        return self

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def test_pick_text_prefers_first_non_empty():
    example = {"sentence": "Hello", "text": "World"}
    assert pick_text(example) == "Hello"
    assert pick_text({"sentence": " ", "text": "ok"}) == "ok"
    assert pick_text({"normalized_text": "norm"}) == "norm"
    assert pick_text({"unused": "x"}) == ""


def test_main_saves_subset(tmp_path, monkeypatch):
    audio = DummyAudio(array=[0.1, -0.1], sampling_rate=16000)
    examples = [
        {"audio": {"array": audio.array, "sampling_rate": audio.sampling_rate}, "text": "first"},
        {"audio": {"array": audio.array, "sampling_rate": audio.sampling_rate}, "text": "second"},
    ]
    dummy_ds = DummyDataset(examples)

    def fake_try_load(dataset_id, config, split, trust_remote_code):
        _ = (dataset_id, config, split, trust_remote_code)
        return dummy_ds

    monkeypatch.setattr("data.save_hf_subset_pairs.try_load", fake_try_load)

    # Capture written files via the stubbed soundfile module.
    dummy_sf = sys.modules["soundfile"]
    dummy_sf.written.clear()

    out_dir = tmp_path / "pairs"
    argv = ["prog", "--dataset", "dummy", "--config", "xx", "--split", "train", "--n", "2", "--out_dir", str(out_dir)]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    wav_files = sorted(out_dir.glob("*.wav"))
    txt_files = sorted(out_dir.glob("*.txt"))

    assert len(wav_files) == 2
    assert len(txt_files) == 2
    assert txt_files[0].read_text(encoding="utf-8") == "first"
    assert txt_files[1].read_text(encoding="utf-8") == "second"

    # Ensure stubbed soundfile.write was invoked with expected sampling rate.
    assert dummy_sf.written[0][2] == 16000

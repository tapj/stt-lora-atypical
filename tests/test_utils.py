from src.utils import RunPaths, mixed_precision_flags


def test_mixed_precision_flags_variants():
    assert mixed_precision_flags("fp16") == (True, False)
    assert mixed_precision_flags("bf16") == (False, True)
    assert mixed_precision_flags("no") == (False, False)
    assert mixed_precision_flags("") == (False, False)


def test_runpaths_from_output_dir(tmp_path):
    rp = RunPaths.from_output_dir(str(tmp_path))
    assert rp.output_dir == str(tmp_path)
    assert rp.adapter_dir.endswith("adapter")
    assert rp.best_dir.endswith("best")
    assert rp.logs_jsonl.endswith("metrics.jsonl")

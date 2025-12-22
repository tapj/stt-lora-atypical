from src.metrics import compute_wer_cer


def test_compute_wer_cer_basic():
    metrics = compute_wer_cer(["hello world"], ["hello world"])
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0

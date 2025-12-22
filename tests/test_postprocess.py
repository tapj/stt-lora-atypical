from src.postprocess import apply_corrections, build_initial_prompt


def test_build_initial_prompt_empty_and_trim():
    assert build_initial_prompt([]) is None
    assert build_initial_prompt(["  "]) is None
    prompt = build_initial_prompt(["Alice", " Bob "])
    assert prompt == "Keywords and names: Alice, Bob"


def test_apply_corrections():
    text = "hello wrld"
    corrections = {"wrld": "world", "unused": "x"}
    assert apply_corrections(text, corrections) == "hello world"

from typing import Dict, List, Optional


def build_initial_prompt(phrase_list: List[str]) -> Optional[str]:
    phrase_list = [p.strip() for p in phrase_list if p and p.strip()]
    if not phrase_list:
        return None
    # Whisper responds well to short, relevant prompts.
    return "Keywords and names: " + ", ".join(phrase_list)


def apply_corrections(text: str, correction_dict: Dict[str, str]) -> str:
    # Simple deterministic replacements.
    out = text
    for k, v in (correction_dict or {}).items():
        if k:
            out = out.replace(k, v)
    return out

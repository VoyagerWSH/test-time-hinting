from __future__ import annotations

from typing import Dict


def build_user_prompt(template: str, kv: Dict[str, str]) -> str:
    out = template
    for k, v in kv.items():
        out = out.replace("{" + k + "}", str(v))
    return out

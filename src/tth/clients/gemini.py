from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict

from tth.config import RoleConfig


async def call_gemini(
    role: RoleConfig,
    payload: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    parse_fn: Callable[[str], tuple],
    parse_name: str,
    api_key: str,
) -> Dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {"parsed": None, "raw_text": "", "raw_obj": None, "error": "google-genai not installed"}

    if not api_key:
        return {"parsed": None, "raw_text": "", "raw_obj": None, "error": "missing GEMINI_API_KEY"}

    client = genai.Client(api_key=api_key)
    img_part = None
    if payload.get("ok"):
        img_part = types.Part.from_bytes(data=payload["bytes"], mime_type=payload["mime"])

    full_text = system_prompt.strip() + "\n\n" + user_prompt.strip()
    cfg_kwargs = {
        "max_output_tokens": int(role.max_output_tokens),
        "temperature": float(role.temperature) if role.temperature is not None else 0.0,
    }
    if role.top_p is not None:
        cfg_kwargs["top_p"] = float(role.top_p)
    if role.thinking_budget is not None:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(role.thinking_budget))

    gen_cfg = types.GenerateContentConfig(**cfg_kwargs)

    def _do_call() -> Any:
        contents = [full_text]
        if img_part is not None:
            contents.append(img_part)
        return client.models.generate_content(
            model=role.model,
            contents=contents,
            config=gen_cfg,
        )

    last_err, last_text, last_raw = None, "", None
    for attempt in range(1, role.max_retries + 1):
        try:
            resp = await asyncio.to_thread(_do_call)
            last_raw = str(resp)
            last_text = (getattr(resp, "text", None) or "").strip()
            parsed, perr = parse_fn(last_text)
            if parsed is not None:
                return {"parsed": parsed, "raw_text": last_text, "raw_obj": last_raw, "error": None}
            last_err = perr or f"{parse_name}_parse_failed"
        except Exception as e:
            last_err = str(e)
    return {"parsed": None, "raw_text": last_text or "", "raw_obj": last_raw, "error": last_err}

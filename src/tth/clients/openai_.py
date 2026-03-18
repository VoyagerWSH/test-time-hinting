from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

from tth.config import RoleConfig


def _data_url_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if payload and payload.get("ok") and payload.get("data_url"):
        return payload["data_url"]
    return None


async def call_openai(
    role: RoleConfig,
    image_path: str,
    payload: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    parse_fn: Callable[[str], tuple],
    parse_name: str,
    api_key: str,
) -> Dict[str, Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, timeout=role.timeout)
    data_url = _data_url_from_payload(payload)
    content = [{"type": "input_text", "text": user_prompt}]
    if data_url:
        content.append({"type": "input_image", "image_url": data_url})

    req: Dict[str, Any] = {
        "model": role.model,
        "instructions": system_prompt,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": role.max_output_tokens,
        "temperature": role.temperature,
    }
    if role.top_p is not None:
        req["top_p"] = role.top_p
    if role.reasoning_effort or role.reasoning_summary:
        req["reasoning"] = {}
        if role.reasoning_effort:
            req["reasoning"]["effort"] = role.reasoning_effort
        if role.reasoning_summary:
            req["reasoning"]["summary"] = role.reasoning_summary

    async def _call_with_fallback(rq: Dict[str, Any], attempts: int = 3) -> Any:
        for _ in range(attempts):
            try:
                return await client.responses.create(**rq)
            except Exception as e:
                msg = str(e)
                m = re.search(r"Unsupported parameter:\s*'([^']+)'", msg)
                if not m:
                    raise
                bad = m.group(1)
                if bad.startswith("reasoning.") and "reasoning" in rq and isinstance(rq["reasoning"], dict):
                    rq["reasoning"].pop(bad.split(".", 1)[1], None)
                    if not rq["reasoning"]:
                        rq.pop("reasoning", None)
                else:
                    rq.pop(bad, None)
        return await client.responses.create(**rq)

    last_err, last_text, last_raw = None, "", None
    for attempt in range(1, role.max_retries + 1):
        try:
            r = dict(req)
            if attempt > 1:
                r["instructions"] = system_prompt.strip() + "\n\nReturn ONLY the required JSON object. No extra text."
            resp = await _call_with_fallback(r)
            last_raw = getattr(resp, "model_dump", lambda: str(resp))()
            last_text = getattr(resp, "output_text", "") or ""
            parsed, perr = parse_fn(last_text)
            if parsed is not None:
                return {"parsed": parsed, "raw_text": last_text, "raw_obj": last_raw, "error": None}
            last_err = perr or f"{parse_name}_parse_failed"
        except Exception as e:
            last_err = str(e)
    return {"parsed": None, "raw_text": last_text or "", "raw_obj": last_raw, "error": last_err}

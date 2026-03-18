from __future__ import annotations

from typing import Any, Callable, Dict

from tth.config import RoleConfig


def _data_url_from_payload(payload: dict) -> bool:
    return bool(payload.get("ok") and payload.get("b64") and payload.get("mime"))


async def call_anthropic(
    role: RoleConfig,
    image_path: str,
    payload: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    parse_fn: Callable[[str], tuple],
    parse_name: str,
    api_key: str,
) -> Dict[str, Any]:
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        return {"parsed": None, "raw_text": "", "raw_obj": None, "error": "anthropic not installed"}

    if not api_key:
        return {"parsed": None, "raw_text": "", "raw_obj": None, "error": "missing ANTHROPIC_API_KEY"}

    client = AsyncAnthropic(api_key=api_key, timeout=role.timeout)
    content_blocks = [{"type": "text", "text": user_prompt}]
    if _data_url_from_payload(payload):
        content_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": payload["mime"], "data": payload["b64"]},
        })

    req: Dict[str, Any] = {
        "model": role.model,
        "max_tokens": int(role.max_output_tokens),
        "system": system_prompt,
        "messages": [{"role": "user", "content": content_blocks}],
    }
    if role.temperature is not None:
        req["temperature"] = float(role.temperature)
    if role.top_p is not None:
        req["top_p"] = float(role.top_p)
    if (role.thinking or "").lower() == "enabled":
        req["thinking"] = {"type": "enabled", "budget_tokens": int(role.thinking_budget_tokens or 1024)}

    last_err, last_text, last_raw = None, "", None
    for attempt in range(1, role.max_retries + 1):
        try:
            resp = await client.messages.create(**req)
            last_raw = getattr(resp, "model_dump", lambda: str(resp))()
            text_parts = []
            for blk in (getattr(resp, "content", None) or []):
                if isinstance(blk, dict) and blk.get("type") == "text":
                    text_parts.append(blk.get("text", ""))
                else:
                    t = getattr(blk, "type", None)
                    if t == "text":
                        text_parts.append(getattr(blk, "text", "") or "")
            last_text = "".join(text_parts).strip()
            parsed, perr = parse_fn(last_text)
            if parsed is not None:
                return {"parsed": parsed, "raw_text": last_text, "raw_obj": last_raw, "error": None}
            last_err = perr or f"{parse_name}_parse_failed"
        except Exception as e:
            last_err = str(e)
    return {"parsed": None, "raw_text": last_text or "", "raw_obj": last_raw, "error": last_err}

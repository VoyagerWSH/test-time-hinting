from __future__ import annotations

import os
from typing import Any, Callable, Dict

from tth.config import RoleConfig


def create_client(
    openai_api_key: str | None = None,
    gemini_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    timeout: float = 120.0,
) -> "MultiProviderClient":
    return MultiProviderClient(
        openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", "").strip(),
        gemini_api_key=gemini_api_key or os.environ.get("GEMINI_API_KEY", "").strip(),
        anthropic_api_key=anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        timeout=timeout,
    )


class MultiProviderClient:
    def __init__(
        self,
        openai_api_key: str = "",
        gemini_api_key: str = "",
        anthropic_api_key: str = "",
        timeout: float = 120.0,
    ):
        self._openai_key = openai_api_key
        self._gemini_key = gemini_api_key
        self._anthropic_key = anthropic_api_key
        self._timeout = timeout

    async def call_role(
        self,
        *,
        role: RoleConfig,
        image_path: str,
        payload: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        parse_fn: Callable[[str], tuple],
        parse_name: str,
    ) -> Dict[str, Any]:
        provider = (role.provider or "").lower().strip()
        if provider == "openai":
            from .openai_ import call_openai
            return await call_openai(
                role=role,
                image_path=image_path,
                payload=payload,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_fn=parse_fn,
                parse_name=parse_name,
                api_key=self._openai_key,
            )
        if provider == "gemini":
            from .gemini import call_gemini
            return await call_gemini(
                role=role,
                payload=payload,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_fn=parse_fn,
                parse_name=parse_name,
                api_key=self._gemini_key,
            )
        if provider == "anthropic":
            from .anthropic_ import call_anthropic
            return await call_anthropic(
                role=role,
                image_path=image_path,
                payload=payload,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_fn=parse_fn,
                parse_name=parse_name,
                api_key=self._anthropic_key,
            )
        return {"parsed": None, "raw_text": "", "raw_obj": None, "error": f"unsupported_provider: {provider}"}

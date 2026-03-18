from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def resolve_image_path(image_path: str, base_path: Optional[str] = None) -> Path:
    p = Path(str(image_path))
    if not p.is_absolute() and base_path:
        p = (Path(base_path) / p).resolve()
    elif not p.is_absolute():
        p = p.resolve()
    return p


def guess_mime(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    if mt:
        return mt
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "application/octet-stream"


def validate_image_loadable(
    image_path: str,
    base_path: Optional[str] = None,
) -> Tuple[bool, str]:
    if not image_path or str(image_path).strip() == "" or str(image_path).lower() == "nan":
        return False, "empty image_path"
    p = resolve_image_path(image_path, base_path)
    if not p.exists():
        return False, f"file not found: {p}"
    try:
        with open(p, "rb") as f:
            f.read(16)
        return True, ""
    except Exception as e:
        return False, str(e)


def load_image_bytes_and_mime(
    image_path: str,
    base_path: Optional[str] = None,
) -> Tuple[bytes, str]:
    p = resolve_image_path(image_path, base_path)
    mime = guess_mime(p)
    with open(p, "rb") as f:
        data = f.read()
    return data, mime


def image_to_payload(
    image_path: str,
    base_path: Optional[str] = None,
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    cache = cache or {}
    key = str(image_path)
    if key in cache:
        return cache[key]
    ok, msg = validate_image_loadable(image_path, base_path)
    if not ok:
        out = {"ok": False, "err": msg}
        cache[key] = out
        return out
    b, mime = load_image_bytes_and_mime(image_path, base_path)
    b64 = base64.b64encode(b).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    out = {"ok": True, "mime": mime, "bytes": b, "b64": b64, "data_url": data_url}
    cache[key] = out
    return out

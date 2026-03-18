from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_EXP_ANS_RE = re.compile(r'"answer"\s*:\s*"([^"]+)"', flags=re.IGNORECASE)
_EXP_REASON_KEY_RE = re.compile(r'"reasoning"\s*:\s*"', flags=re.IGNORECASE)


def get_required_columns(cfg_id_col: str, image_col: str, question_col: str,
                        gt_answer_col: str, gt_rationale_col: str,
                        base_answer_col: str, base_rationale_col: str,
                        base_correct_col: str) -> List[str]:
    return [
        cfg_id_col, image_col, question_col,
        gt_answer_col, gt_rationale_col,
        base_answer_col, base_rationale_col, base_correct_col,
    ]


def validate_csv_columns(df: pd.DataFrame, required: List[str]) -> Tuple[bool, str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    return True, ""


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    s = str(text).strip()
    s = re.sub(r"^\s*(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(s)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def validate_hint_schema(obj: Dict[str, Any]) -> Tuple[bool, str]:
    if set(obj.keys()) != {"hint"}:
        return False, "expected exactly key 'hint'"
    h = obj.get("hint")
    if not isinstance(h, list) or not (1 <= len(h) <= 3):
        return False, "hint must be list of length 1..3"
    for item in h:
        if not _is_nonempty_str(item):
            return False, "each hint item must be non-empty string"
    return True, ""


def parse_hint_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    obj = extract_json_object(text or "")
    if not obj:
        return None, "parse_failed"
    ok, msg = validate_hint_schema(obj)
    if not ok:
        return None, msg
    return {"hint": [str(x).strip() for x in obj["hint"]]}, None


def validate_checker_schema(obj: Dict[str, Any]) -> Tuple[bool, str]:
    if set(obj.keys()) != {"verdict", "feedback", "hint"}:
        return False, "expected verdict, feedback, hint"
    v = str(obj.get("verdict", "")).strip().lower()
    if v not in ("pass", "revise"):
        return False, "verdict must be pass or revise"
    if not isinstance(obj.get("feedback", ""), str):
        return False, "feedback must be string"
    h = obj.get("hint")
    if not isinstance(h, list) or not (1 <= len(h) <= 3):
        return False, "hint must be list length 1..3"
    for item in h:
        if not _is_nonempty_str(item):
            return False, "each hint item non-empty string"
    return True, ""


def parse_checker_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    obj = extract_json_object(text or "")
    if not obj:
        return None, "parse_failed"
    ok, msg = validate_checker_schema(obj)
    if not ok:
        return None, msg
    out = {
        "verdict": str(obj["verdict"]).strip().lower(),
        "feedback": str(obj.get("feedback", "") or "").strip(),
        "hint": [str(x).strip() for x in obj["hint"]],
    }
    if out["verdict"] == "pass":
        out["feedback"] = ""
    return out, None


def _exp_strip(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"^\s*(?:json)?\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def _brace_scan_objects(s: str, max_objs: int = 50) -> List[str]:
    s = s or ""
    out = []
    i = s.find("{")
    if i < 0:
        return out
    depth = 0
    in_str = False
    esc = False
    start = None
    for k in range(i, len(s)):
        ch = s[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = k
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                out.append(s[start : k + 1])
                if len(out) >= max_objs:
                    break
                start = None
    return out


def _try_load_dict(s: str) -> Optional[Dict[str, Any]]:
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else None
    except Exception:
        return None


def _extract_answer_reasoning_fallback(s: str) -> Tuple[Optional[str], Optional[str]]:
    if not s:
        return None, None
    m_ans = _EXP_ANS_RE.search(s)
    if not m_ans:
        return None, None
    ans = (m_ans.group(1) or "").strip()
    if not ans:
        return None, None
    m_r = _EXP_REASON_KEY_RE.search(s)
    if not m_r:
        return ans, None
    r_start = m_r.end()
    end = s.rfind('"}')
    if end >= r_start:
        rea = s[r_start:end].strip()
    else:
        endq = s.rfind('"')
        rea = (s[r_start:endq] if endq >= r_start else s[r_start:]).strip()
    return ans, rea if rea else None


def parse_experimenter_json(text: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    s = _exp_strip(text or "")
    obj = _try_load_dict(s)
    if obj and isinstance(obj.get("answer"), str) and isinstance(obj.get("reasoning"), str):
        ans = str(obj["answer"]).strip()
        rea = str(obj["reasoning"]).strip()
        if ans and rea:
            return {"answer": ans, "reasoning": rea}, None
    for cand in reversed(_brace_scan_objects(s)):
        obj = _try_load_dict(cand)
        if obj and "answer" in obj and "reasoning" in obj:
            ans = str(obj.get("answer", "")).strip()
            rea = str(obj.get("reasoning", "")).strip()
            if ans and rea:
                return {"answer": ans, "reasoning": rea}, None
    ans, rea = _extract_answer_reasoning_fallback(s)
    if ans is not None and rea is not None:
        return {"answer": ans, "reasoning": rea}, None
    return None, "parse_failed"


def norm_answer(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and getattr(x, "__str__", None) and "nan" in str(x).lower()):
        return None
    s = str(x).strip().lower()
    return s if s and s != "nan" else None


def is_correct(pred: Any, gt: Any) -> bool:
    return (norm_answer(pred) or "") == (norm_answer(gt) or "")


def parse_base_correct(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value) and not (isinstance(value, float) and value != value)
    s = str(value).strip().lower()
    return s in ("true", "1", "yes")

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _none_if_blank(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return None if s == "" or s.lower() == "none" else s


@dataclass
class RoleConfig:
    provider: str
    model: str
    temperature: float
    top_p: Optional[float]
    max_output_tokens: int
    timeout: float
    max_retries: int
    max_concurrent: int = 4
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    thinking_budget: Optional[int] = None
    thinking: Optional[str] = None
    thinking_budget_tokens: Optional[int] = None


@dataclass
class RunConfig:
    input_csv: str
    output_dir: str
    image_base_path: str
    id_col: str
    image_col: str
    question_col: str
    gt_answer_col: str
    gt_rationale_col: str
    base_answer_col: str
    base_rationale_col: str
    base_correct_col: str
    caption_col: str
    process_repair: bool
    process_reinforcement: bool
    max_rounds: int
    skip_checker: bool
    seed: int
    checkpoint_every_n: int
    num_workers: int
    proposer: RoleConfig
    checker: RoleConfig
    experimenter: RoleConfig
    proposer_system_prompt_repair: str
    proposer_user_prompt_template_repair: str
    proposer_system_prompt_reinforce: str
    proposer_user_prompt_template_reinforce: str
    checker_system_prompt_repair: str
    checker_user_prompt_template_repair: str
    checker_system_prompt_reinforce: str
    checker_user_prompt_template_reinforce: str
    experimenter_system_prompt: str
    experimenter_user_prompt_template: str


def _role_from_dict(d: Dict[str, Any]) -> RoleConfig:
    return RoleConfig(
        provider=str(d.get("provider", "openai")).lower().strip(),
        model=str(d["model"]),
        temperature=float(d.get("temperature", 0.0)),
        top_p=None if d.get("top_p") in ("", None) else float(d["top_p"]),
        max_output_tokens=int(d.get("max_output_tokens", 2048)),
        timeout=float(d.get("timeout", 120.0)),
        max_retries=int(d.get("max_retries", 3)),
        max_concurrent=int(d.get("max_concurrent", 4)),
        reasoning_effort=_none_if_blank(d.get("reasoning_effort")),
        reasoning_summary=_none_if_blank(d.get("reasoning_summary")),
        thinking_budget=None if d.get("thinking_budget") in ("", None) else int(d["thinking_budget"]),
        thinking=_none_if_blank(d.get("thinking")),
        thinking_budget_tokens=(
            None if d.get("thinking_budget_tokens") in ("", None) else int(d["thinking_budget_tokens"])
            if str(d.get("provider", "")).lower() == "anthropic"
            else None
        ),
    )


_REQUIRED_KEYS = [
    "input_csv", "output_dir", "id_col", "image_col", "question_col",
    "gt_answer_col", "gt_rationale_col", "base_answer_col", "base_rationale_col", "base_correct_col",
    "process_repair", "process_reinforcement", "max_rounds", "skip_checker",
    "seed", "checkpoint_every_n", "num_workers",
    "proposer", "checker", "experimenter",
    "proposer_system_prompt_repair", "proposer_user_prompt_template_repair",
    "proposer_system_prompt_reinforce", "proposer_user_prompt_template_reinforce",
    "checker_system_prompt_repair", "checker_user_prompt_template_repair",
    "checker_system_prompt_reinforce", "checker_user_prompt_template_reinforce",
    "experimenter_system_prompt", "experimenter_user_prompt_template",
]


def load_config(path: str | Path) -> RunConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    return RunConfig(
        input_csv=str(raw["input_csv"]),
        output_dir=str(raw["output_dir"]),
        image_base_path=str(raw.get("image_base_path") or ""),
        id_col=str(raw["id_col"]),
        image_col=str(raw["image_col"]),
        question_col=str(raw["question_col"]),
        gt_answer_col=str(raw["gt_answer_col"]),
        gt_rationale_col=str(raw["gt_rationale_col"]),
        base_answer_col=str(raw["base_answer_col"]),
        base_rationale_col=str(raw["base_rationale_col"]),
        base_correct_col=str(raw["base_correct_col"]),
        caption_col=str(raw.get("caption_col") or raw.get("gt_rationale_col") or ""),
        process_repair=bool(raw["process_repair"]),
        process_reinforcement=bool(raw["process_reinforcement"]),
        max_rounds=int(raw["max_rounds"]),
        skip_checker=bool(raw["skip_checker"]),
        seed=int(raw["seed"]),
        checkpoint_every_n=int(raw["checkpoint_every_n"]),
        num_workers=int(raw["num_workers"]),
        proposer=_role_from_dict(raw["proposer"]),
        checker=_role_from_dict(raw["checker"]),
        experimenter=_role_from_dict(raw["experimenter"]),
        proposer_system_prompt_repair=str(raw["proposer_system_prompt_repair"]),
        proposer_user_prompt_template_repair=str(raw["proposer_user_prompt_template_repair"]),
        proposer_system_prompt_reinforce=str(raw["proposer_system_prompt_reinforce"]),
        proposer_user_prompt_template_reinforce=str(raw["proposer_user_prompt_template_reinforce"]),
        checker_system_prompt_repair=str(raw["checker_system_prompt_repair"]),
        checker_user_prompt_template_repair=str(raw["checker_user_prompt_template_repair"]),
        checker_system_prompt_reinforce=str(raw["checker_system_prompt_reinforce"]),
        checker_user_prompt_template_reinforce=str(raw["checker_user_prompt_template_reinforce"]),
        experimenter_system_prompt=str(raw["experimenter_system_prompt"]),
        experimenter_user_prompt_template=str(raw["experimenter_user_prompt_template"]),
    )

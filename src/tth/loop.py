from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional, Tuple

from tth.config import RunConfig
from tth.prompts.builder import build_user_prompt
from tth.schema import (
    is_correct,
    parse_checker_json,
    parse_experimenter_json,
    parse_hint_json,
)

PROPOSER_PARSE_RETRIES = 2
PROPOSER_RETRY_SLEEP = 0.25


def _hint_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _row_get(row: Dict[str, Any], col: str, default: str = "") -> str:
    v = row.get(col, default)
    if v is None or (isinstance(v, float) and str(v) == "nan"):
        return default
    return str(v).strip()


async def run_agentic_loop(
    row: Dict[str, Any],
    cfg: RunConfig,
    client: Any,
    payload_getter: Callable[[str], Dict[str, Any]],
    sem_proposer: Optional[asyncio.Semaphore] = None,
    sem_checker: Optional[asyncio.Semaphore] = None,
    sem_experimenter: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    image_path = _row_get(row, cfg.image_col)
    q = _row_get(row, cfg.question_col)
    cap = _row_get(row, cfg.caption_col or cfg.gt_rationale_col)
    gt = _row_get(row, cfg.gt_answer_col)
    base_answer = _row_get(row, cfg.base_answer_col)
    base_reasoning = _row_get(row, cfg.base_rationale_col)
    base_correct_val = row.get(cfg.base_correct_col)
    from tth.schema import parse_base_correct
    is_repair = not parse_base_correct(base_correct_val)

    payload = payload_getter(image_path)
    if not payload.get("ok"):
        return {
            "hint_json": None,
            "selected_round": None,
            "checker_verdict": None,
            "checker_feedback": None,
            "exp_answer": None,
            "exp_reasoning": None,
            "exp_correct": False,
            "fatal_error": payload.get("err", "image_load_failed"),
            "trajectory": json.dumps({"rounds": [], "fatal_error": payload.get("err")}),
            "outcome": "fatal",
            "discarded": False,
        }

    if is_repair:
        sys_proposer = cfg.proposer_system_prompt_repair
        tpl_proposer = cfg.proposer_user_prompt_template_repair
        sys_checker = cfg.checker_system_prompt_repair
        tpl_checker = cfg.checker_user_prompt_template_repair
        success_goal = "flip_to_correct"
    else:
        sys_proposer = cfg.proposer_system_prompt_reinforce
        tpl_proposer = cfg.proposer_user_prompt_template_reinforce
        sys_checker = cfg.checker_system_prompt_reinforce
        tpl_checker = cfg.checker_user_prompt_template_reinforce
        success_goal = "keep_correct"

    @asynccontextmanager
    async def _acq(sem: Optional[asyncio.Semaphore]):
        if sem is None:
            yield
        else:
            async with sem:
                yield

    latest_feedback = ""
    last_exp_answer = ""
    last_exp_reasoning = ""
    candidates = []
    first_checker_pass_idx = None
    trajectory = {"rounds": []}

    for round_idx in range(1, cfg.max_rounds + 1):
        intro = f"Round {round_idx}\n\n" if round_idx > 1 else ""
        proposer_user = build_user_prompt(tpl_proposer, {
            "INTRO": intro,
            "QUESTION": q,
            "CAPTION": cap,
            "GROUND_TRUTH_ANSWER": gt,
            "BASE_ANSWER": base_answer,
            "BASE_REASONING": base_reasoning,
            "OPTIONAL_CHECKER_FEEDBACK": latest_feedback or "",
            "OPTIONAL_EXPERIMENTER_LAST_ANSWER": last_exp_answer or "",
            "OPTIONAL_EXPERIMENTER_LAST_REASONING": last_exp_reasoning or "",
        })

        proposer_res = None
        for _ in range(PROPOSER_PARSE_RETRIES + 1):
            async with _acq(sem_proposer):
                proposer_res = await client.call_role(
                    role=cfg.proposer,
                    image_path=image_path,
                    payload=payload,
                    system_prompt=sys_proposer,
                    user_prompt=proposer_user,
                    parse_fn=parse_hint_json,
                    parse_name="hint",
                )
            if proposer_res.get("parsed") is not None:
                break
            if PROPOSER_RETRY_SLEEP > 0:
                await asyncio.sleep(PROPOSER_RETRY_SLEEP)

        if proposer_res is None or proposer_res.get("parsed") is None:
            trajectory["rounds"].append({"round": round_idx, "stage": "proposer", "error": proposer_res.get("error") if proposer_res else "no_res"})
            continue

        hint_obj = proposer_res["parsed"]
        used_hint = hint_obj
        checker_verdict = None
        checker_feedback = ""

        if not cfg.skip_checker:
            checker_user = build_user_prompt(tpl_checker, {
                "QUESTION": q,
                "CAPTION": cap,
                "GROUND_TRUTH_ANSWER": gt,
                "BASE_ANSWER": base_answer,
                "BASE_REASONING": base_reasoning,
                "HINT_JSON": _hint_dumps(hint_obj),
            })
            async with _acq(sem_checker):
                chk_res = await client.call_role(
                    role=cfg.checker,
                    image_path=image_path,
                    payload=payload,
                    system_prompt=sys_checker,
                    user_prompt=checker_user,
                    parse_fn=parse_checker_json,
                    parse_name="checker",
                )
            if chk_res.get("parsed") is None:
                checker_verdict = "revise"
                checker_feedback = chk_res.get("error") or "checker_parse_failed"
            else:
                checker_verdict = chk_res["parsed"]["verdict"]
                checker_feedback = chk_res["parsed"]["feedback"]
                used_hint = {"hint": chk_res["parsed"]["hint"]}
            if checker_verdict == "pass" and first_checker_pass_idx is None:
                first_checker_pass_idx = len(candidates)

        exp_user = build_user_prompt(cfg.experimenter_user_prompt_template, {
            "QUESTION": q,
            "HINT_JSON": _hint_dumps(used_hint),
        })
        async with _acq(sem_experimenter):
            exp_res = await client.call_role(
                role=cfg.experimenter,
                image_path=image_path,
                payload=payload,
                system_prompt=cfg.experimenter_system_prompt,
                user_prompt=exp_user,
                parse_fn=parse_experimenter_json,
                parse_name="experimenter",
            )

        if exp_res.get("parsed") is None:
            exp_answer = ""
            exp_reasoning = ""
            exp_correct = False
        else:
            exp_answer = exp_res["parsed"]["answer"]
            exp_reasoning = exp_res["parsed"]["reasoning"]
            exp_correct = is_correct(exp_answer, gt)

        latest_feedback = checker_feedback
        last_exp_answer = exp_answer
        last_exp_reasoning = exp_reasoning

        candidates.append({
            "round": round_idx,
            "hint_obj": used_hint,
            "checker_verdict": checker_verdict,
            "checker_feedback": checker_feedback,
            "exp_answer": exp_answer,
            "exp_reasoning": exp_reasoning,
            "exp_correct": exp_correct,
        })
        trajectory["rounds"].append({
            "round": round_idx,
            "hint_json": used_hint,
            "checker_verdict": checker_verdict,
            "checker_feedback": checker_feedback,
            "exp_answer": exp_answer,
            "exp_correct": exp_correct,
        })

        if success_goal == "flip_to_correct" and exp_correct:
            return {
                "hint_json": _hint_dumps(used_hint),
                "selected_round": round_idx,
                "checker_verdict": checker_verdict if not cfg.skip_checker else None,
                "checker_feedback": checker_feedback if not cfg.skip_checker else None,
                "exp_answer": exp_answer,
                "exp_reasoning": exp_reasoning,
                "exp_correct": True,
                "fatal_error": None,
                "trajectory": json.dumps(trajectory, ensure_ascii=False),
                "outcome": "success",
                "discarded": False,
            }
        if success_goal == "keep_correct" and exp_correct:
            return {
                "hint_json": _hint_dumps(used_hint),
                "selected_round": round_idx,
                "checker_verdict": checker_verdict if not cfg.skip_checker else None,
                "checker_feedback": checker_feedback if not cfg.skip_checker else None,
                "exp_answer": exp_answer,
                "exp_reasoning": exp_reasoning,
                "exp_correct": True,
                "fatal_error": None,
                "trajectory": json.dumps(trajectory, ensure_ascii=False),
                "outcome": "success",
                "discarded": False,
            }

    if not candidates:
        return {
            "hint_json": None,
            "selected_round": None,
            "checker_verdict": None,
            "checker_feedback": None,
            "exp_answer": None,
            "exp_reasoning": None,
            "exp_correct": False,
            "fatal_error": "no_candidates",
            "trajectory": json.dumps(trajectory, ensure_ascii=False),
            "outcome": "no_candidates",
            "discarded": not is_repair,
        }

    if is_repair:
        if not cfg.skip_checker and first_checker_pass_idx is not None:
            chosen = candidates[first_checker_pass_idx]
        else:
            chosen = candidates[-1]
        return {
            "hint_json": _hint_dumps(chosen["hint_obj"]),
            "selected_round": chosen["round"],
            "checker_verdict": chosen["checker_verdict"] if not cfg.skip_checker else None,
            "checker_feedback": chosen["checker_feedback"] if not cfg.skip_checker else None,
            "exp_answer": chosen["exp_answer"],
            "exp_reasoning": chosen["exp_reasoning"],
            "exp_correct": chosen["exp_correct"],
            "fatal_error": None,
            "trajectory": json.dumps(trajectory, ensure_ascii=False),
            "outcome": "partial_repair",
            "discarded": False,
        }
    else:
        return {
            "hint_json": None,
            "selected_round": None,
            "checker_verdict": None,
            "checker_feedback": None,
            "exp_answer": None,
            "exp_reasoning": None,
            "exp_correct": False,
            "fatal_error": None,
            "trajectory": json.dumps(trajectory, ensure_ascii=False),
            "outcome": "discard",
            "discarded": True,
        }

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from tth.config import RunConfig
from tth.schema import (
    get_required_columns,
    parse_base_correct,
    validate_csv_columns,
)
from tth.images import image_to_payload
from tth.loop import run_agentic_loop
from tth.clients import create_client


def set_seed(seed: int) -> None:
    random.seed(seed)


def build_rows(df: pd.DataFrame, cfg: RunConfig) -> List[Dict[str, Any]]:
    required = get_required_columns(
        cfg.id_col,
        cfg.image_col,
        cfg.question_col,
        cfg.gt_answer_col,
        cfg.gt_rationale_col,
        cfg.base_answer_col,
        cfg.base_rationale_col,
        cfg.base_correct_col,
    )
    ok, msg = validate_csv_columns(df, required)
    if not ok:
        raise ValueError(msg)

    rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        base_correct = parse_base_correct(row.get(cfg.base_correct_col))
        if base_correct and not cfg.process_reinforcement:
            continue
        if not base_correct and not cfg.process_repair:
            continue
        rows.append(row)
    return rows


def row_done(row: Dict[str, Any], hint_col: str = "hint_json", error_col: str = "fatal_error", outcome_col: str = "outcome") -> bool:
    h = row.get(hint_col)
    e = row.get(error_col)
    o = row.get(outcome_col)
    if pd.notna(h) and h is not None and str(h).strip() != "":
        return True
    if pd.notna(e) and e is not None and str(e).strip() != "":
        return True
    if pd.notna(o) and o is not None and str(o).strip() != "":
        return True
    return False


async def run_all(
    cfg: RunConfig,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
) -> pd.DataFrame:
    in_path = input_path or Path(cfg.input_csv)
    out_path = output_path or Path(cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    rows = build_rows(df, cfg)
    if not rows:
        return df

    out_cols = [
        "hint_json", "selected_round", "checker_verdict", "checker_feedback",
        "exp_answer", "exp_reasoning", "exp_correct", "fatal_error", "trajectory",
        "outcome", "discarded",
    ]
    id_col = cfg.id_col
    id_to_idx = {}
    for i, r in enumerate(rows):
        id_to_idx[str(r.get(id_col, i))] = i

    result_rows = [dict(r) for r in rows]
    for r in result_rows:
        for c in out_cols:
            if c not in r:
                r[c] = None

    ckpt_path = checkpoint_path or (out_path / "checkpoint.csv")
    if resume and ckpt_path.exists():
        ckpt = pd.read_csv(ckpt_path)
        for _, cr in ckpt.iterrows():
            rid = str(cr.get(id_col, ""))
            if rid not in id_to_idx:
                continue
            idx = id_to_idx[rid]
            for c in out_cols:
                if c in cr and pd.notna(cr[c]):
                    result_rows[idx][c] = cr[c]

    pending = [i for i in range(len(result_rows)) if not row_done(result_rows[i])]
    if not pending:
        out_df = pd.DataFrame(result_rows)
        out_df.to_csv(out_path / "output.csv", index=False)
        return out_df

    timeout = max(cfg.proposer.timeout, cfg.checker.timeout, cfg.experimenter.timeout)
    pbar = tqdm(total=len(pending), desc="hint optimization", unit="row")
    client = create_client(timeout=timeout)
    cache = {}
    base = cfg.image_base_path or ""

    def payload_getter(path: str):
        return image_to_payload(path, base_path=base if base else None, cache=cache)

    sem_p = asyncio.Semaphore(max(1, cfg.proposer.max_concurrent))
    sem_c = asyncio.Semaphore(max(1, cfg.checker.max_concurrent))
    sem_e = asyncio.Semaphore(max(1, cfg.experimenter.max_concurrent))

    batch_size = max(1, cfg.checkpoint_every_n)
    for start in range(0, len(pending), batch_size):
        batch_idx = pending[start : start + batch_size]
        tasks = [
            run_agentic_loop(
                result_rows[i],
                cfg,
                client,
                payload_getter,
                sem_proposer=sem_p,
                sem_checker=sem_c,
                sem_experimenter=sem_e,
            )
            for i in batch_idx
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in zip(batch_idx, results):
            if isinstance(res, Exception):
                result_rows[i]["fatal_error"] = str(res)
                result_rows[i]["outcome"] = "fatal"
                result_rows[i]["discarded"] = False
            else:
                for k, v in res.items():
                    result_rows[i][k] = v

        ckpt_df = pd.DataFrame(result_rows)
        ckpt_df.to_csv(ckpt_path, index=False)
        pbar.update(len(batch_idx))
    pbar.close()

    out_df = pd.DataFrame(result_rows)
    out_df.to_csv(out_path / "output.csv", index=False)
    return out_df


def run_sync(
    cfg: RunConfig,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
) -> pd.DataFrame:
    set_seed(cfg.seed)
    return asyncio.run(run_all(cfg, input_path, output_path, checkpoint_path, resume))

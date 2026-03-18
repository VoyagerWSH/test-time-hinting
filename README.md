# Test-Time Hinting

**Test-time scaling (TTS)** has proven effective for LLMs; its application to vision–language models (VLMs) remains less explored. Existing VLM TTS methods often require open-weight access or expensive repeated sampling. **Test-Time Hinting (TTH)** improves VLM performance via a single VLM call and black-box API access: a lightweight hint generator predicts, for a given test input, which “hint” to prepend to the prompt, providing targeted guidance that steers the VLM away from characteristic failure modes. Hints improve accuracy on natural-image VQA and generalize to unseen benchmarks and VLMs without retraining.

![Test-Time Hinting](assets/tth.gif)

This repository provides the **hint optimization** pipeline: given a dataset of image–question pairs and base model responses, it runs a three-role agentic loop (Proposer, Checker, Experimenter) to produce high-quality hints. The Proposer suggests hints; the Checker enforces safety and quality (no answer leakage, contrastive guidance); the Experimenter is the target VLM, which re-answers under the hint so the loop can terminate when the model answers correctly (repair) or stays correct (reinforcement).

## Setup

```bash
pip install -r requirements.txt
```

Set API keys (e.g. in the environment):

- `OPENAI_API_KEY` — for Proposer and Checker
- `GEMINI_API_KEY` — for Experimenter (target VLM), if using Gemini

## Input

A CSV with one row per example. Column names are configurable; typical names:

| Column         | Description                          |
|----------------|--------------------------------------|
| `id`           | Row identifier                        |
| `image_path`   | Path to image file                    |
| `question`     | Full question (include options if MCQ)|
| `gt_answer`    | Ground-truth answer                   |
| `gt_rationale` | Ground-truth rationale (optional)     |
| `base_answer`  | Target model’s answer without hint    |
| `base_rationale`| Target model’s reasoning              |
| `base_correct` | Whether base answer matches gt        |

## Usage

From the repo root, install in development mode then run:

```bash
cd test-time-hinting
pip install -e .
python -m tth.main --config configs/default.yaml --input data.csv --output out/
```

Or without installing: `PYTHONPATH=src python -m tth.main ...`

Resume from a checkpoint:

```bash
python -m tth.main --config configs/default.yaml --input data.csv --output out/ --resume
```

## Config

Edit `configs/default.yaml` to set input/output paths, column names, and roles. Example setup: Gemini as Experimenter (target VLM), OpenAI as Proposer and Checker. All column names and provider/model IDs are configurable.

## Output

The output CSV adds columns: `hint_json`, `selected_round`, `checker_verdict`, `checker_feedback`, `exp_answer`, `exp_reasoning`, `exp_correct`, and optionally `fatal_error`, `trajectory`. Rows that are reinforcement trials and never succeed under any hint are marked discarded (no hint retained).

## Citation

If you use this code, please cite the paper (Test-Time Hinting for Black-Box Vision-Language Models, ECCV 2026).

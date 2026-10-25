# Paper 9: can a model be trained to read its own computation?

**Question.** In paper 8, Qwen3-1.7B's self-report reconstructed the answer from the input and ignored a manipulated internal state. Paper 9 asks whether training can change that. The test pits training on interventional examples (an answer-fixed activation swap changes an internal step while the input and the answer stay the same) against training on ordinary examples. It then checks whether the trained reports track the model's natural computation better than any predictor that sees only the input.

**Attribution.** The core idea is the author's (Sam Ramdan, 2026-10-21): train self-report on rows where an answer-fixed intervention changed an internal step, test transfer to natural computation against an input-only predictor, and use decoded neurofeedback as the human analogy. The novelty check, the design choices (task, ground truth, interventions, arms, tests, decision rule) and all code are Claude's (Opus 5.5), recorded as such in PREDICTIONS.md.

## Files

| File | What |
|---|---|
| `NOVELTY.md` | Novelty verdict and narrowed claim; full evidence in `novelty_work/NOVELTY_REPORT.md` |
| `PREDICTIONS.md` | Preregistration (frozen; hashes in `PREDICTIONS.sha256`) |
| `DEVIATIONS.md` | Every change after the freeze |
| `RESULTS.md` | Results, opening with a plain-language summary |
| `code/` | `cities.py` (gazetteer), `p9lib.py` (engine), `build.py` and `items.py` (units and examples), `trainlib.py`, stage scripts, `analyze.py` |
| `res/` | Analysis data (written only after the freeze, except `res/screen.json`, the pre-freeze item screen) |
| `res_pilot/` | Pilot on pilot cities (never analysed) |
| `res_selftest/` | End-to-end self-test on pilot cities (never analysed) |
| `TASK_SEARCH.md` | The overnight search for a replacement task (v3) |
| `PREDICTIONS_V2.md`, `PREDICTIONS_V3.md`, `PREDICTIONS_V4.md` | The frozen addenda (hashes in the matching `.sha256` files): v2 premise test, v3 multiplication-confidence task, v4 learnable two-bin version with a budget ladder |
| `code_v2/`, `code_v3/`, `code_v4/` | Code of each addendum. v4: `v4lib.py` (two-bin layer and fp32-letter engine), `v4stats.py` (input-only predictors), `trainlib_v4.py`, `v4_run.py` (ladder, gen, train, eval), `v4_analyze.py`, `job_v4.py`, `run_v4.sh` |
| `res_v2/`, `res_v3/`, `res_pilot_v3/` | v2 and v3 data |
| `res_v4_design/` | v4 design numbers from the base-model item screen (pre-freeze) |
| `res_pilot_v4/`, `res_v4/` | v4 learnability ladder and main data (written only after the v4 freeze) |
| `*_smoke/` | Code tests on pilot items with tiny budgets (never analysed) |
| `stage0/` | An exploratory linear-probe check on pilot products, made by a separate session (not part of any preregistration) |
| `audit/` | Independent audits (`AUDIT_v1v2.md`, `AUDIT_v3.md`, then v4) |
| `logs/` | Job logs |

## How it runs

Every GPU job goes through `tools/gpu_lock.py` with labels `p9-*` (v4: `p9v4-*`, with `--max-min 90`; training checkpoints every 200 steps and exits 3 near the lock's deadline, so the job is re-queued and resumes). Each chunk first waits for mains power if the laptop is on battery below 30%. The driver is `code/run_main.sh`, and creating `STOP` in this folder stops it after the current chunk. All stages resume: a finished training run or evaluation is skipped.

## Planned follow-ups (written down now, NOT run in this study)

1. **Replication on other small models, on the laptop:** SmolLM2-1.7B-Instruct and Qwen3-4B (both cached) and OLMo-2-0425-1B (the cached checkpoint is the base model, so it needs a chat-format check first). Rerun the frozen code with the model key and layers mapped by relative depth, after a fresh item screen. Each model gets its own short preregistration.
2. **8-14B models on a rented GPU,** only with the author's explicit approval of the cost, and only if the Qwen3-1.7B result is SUPPORTED or the natural-variation gate fails because of model size.
3. **Chain-of-thought faithfulness extension.** Train on intervened reasoning steps: with thinking on, swap the computation at one intermediate step of a multi-step problem for a donor's step that leads to the same final answer. Then test whether the model reports the route it actually took on natural (unpatched) chains, against an input-only predictor and a predictor that reads the written chain of thought.

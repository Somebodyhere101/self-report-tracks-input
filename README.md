# Self-Report Positions in Qwen3-1.7B Track the Input, Not the Swapped-In Computation

This repository holds the code, preregistrations, results and audits for **"Self-Report Positions in Qwen3-1.7B Track the Input, Not the Swapped-In Computation: Answer-Fixed Swaps, Probes and Null Training Results"** (Sam Ramdan, October 2026). It is paper 9 in a series on whether hidden-state directions and self-reports track the processes they are named after.

- **Paper:** [`paper/self-report-tracks-input.pdf`](paper/self-report-tracks-input.pdf). The LaTeX source, `make_numbers.py` (which writes every number and figure from the result files) and the citation check are in `research_next/p9_trained_introspection/paper/`.
- **Zenodo:** `<ZENODO-DOI>` (added when the record is published). The record also holds `code-and-data.zip`, which adds the per-item result files that are too large for git.
- **Previous paper in the series:** [Where a Logit-Lens Settling Profile Is Carried, and Whether Qwen3-1.7B Reports It (paper 8)](https://doi.org/10.5281/zenodo.23532061).

## What the study does

Paper 8 found that Qwen3-1.7B's self-report about a multiplication follows what the question implies, not a swapped-in internal state. Paper 9 asks whether training can change that: can a self-report be trained to read the model's computation rather than reconstruct it from the input? The test trains reporters on examples where an answer-fixed swap changed an internal step while the input and the answer stayed the same, and compares them with reporters trained on ordinary, injected or text-only examples. It then asks where the information that a report would need can and cannot be read.

| Stage (folder) | Question | Result |
|---|---|---|
| v1 (`PREDICTIONS.md`, `code/`) | Two-hop task ("Which continent is the country containing {city} in?"): train reporters on swapped bridge countries | Stopped at its frozen bed check; no main data |
| v2 (`PREDICTIONS_V2.md`, `code_v2/`, `res_v2/`) | Do the report and a readout recover the country from the same window? | Preregistered verdict SAME WINDOW: reading the computation and re-reading the input cannot be told apart in this task |
| Task search (`TASK_SEARCH.md`, `diag/`) | A replacement task with an intermediate computed after the input | Confidence in a product's second digit passed; six candidates did not |
| v3 (`PREDICTIONS_V3.md`, `code_v3/`, `res_v3/`) | Do reporters trained on swapped confidence report natural confidence better than an input-only predictor? | Preregistered verdict NOT SUPPORTED: all reporters stayed at chance |
| Stage 0a, 0b (`stage0/`, `stage0b/`) | Where can the swapped-in confidence be read? Does short training open a route? | Exploratory. Readable at the answer digits, not at the report positions; short training opened no route |
| v4 (`PREDICTIONS_V4.md`, `code_v4/`, `res_v4_design/`) | A two-bin version with a budget ladder | Drafted, never frozen; stopped by its own pre-freeze rule after Stage 0b |

`RESULTS.md` gives the full results, opening with a plain-language summary. `DEVIATIONS.md` logs every change after each freeze. `audit/` holds the independent audits, which recomputed the v1 to v3 results and every number in the paper.

**Wording in the frozen files.**
- "Claude" and "the assistant" are Claude Opus 5.5. "The coordinator" or "the coordinating session" is the Claude session that passed on the author's requests.
- **Correction:** `PREDICTIONS.md`, `PREDICTIONS_V2.md`, `PREDICTIONS_V3.md`, the never-frozen draft `PREDICTIONS_V4.md` and the study's `README.md` say that the core idea is the author's, and `PREDICTIONS.md` includes the method in it. The question, and the idea of answering it by training, were the author's; the method and the neurofeedback analogy were proposed by Claude in reply. These files are not edited; `RESULTS.md` has been corrected. The dated correction is in `DEVIATIONS.md` (2026-10-25), and the paper's statement below gives the corrected split.

## Layout

The study's code finds paper 8's and the S2 study's modules by relative paths from the workspace root, and the frozen hash files name them by those paths. The repository therefore keeps the workspace layout:

| Path | Contents |
|---|---|
| `research_next/p9_trained_introspection/` | The study: preregistrations, code, results, audits and the paper source |
| `paper8/selfreport/`, `paper8/patch2/`, `paper8/patch/`, `paper8/code/` | Paper 8's interchange engine (`srlib.py`, `patchlib2.py`, `patchlib.py`, `p8lib.py`, `p6lib.py`), unchanged |
| `research_next/s2_monitors/code/` | The S2 study's LoRA layer (`s2lib.py`) and its scorers (`scorers.py`), unchanged |
| `paper/` | The paper as a PDF |
| `DATA.md` | What is withheld and how to rebuild it |
| `strip_report.json` | What the release filter removed or changed |

## Verifying the freezes

From `research_next/p9_trained_introspection/`, run:

```
python code/freeze.py check
python code_v2/freeze_v2.py check
python code_v3/freeze_v3.py check
```

They compare every line of `PREDICTIONS.sha256`, `PREDICTIONS_V2.sha256` and `PREDICTIONS_V3.sha256` with the files. `PREDICTIONS_V3.sha256.amend` records a logged pre-data fix to v3's code-test mode, which the check accepts. Lines that start with `WS:` are paths from the repository root. Every hashed file is in this repository byte for byte.

**One line fails:** the study's `README.md` in `PREDICTIONS.sha256`. That README is a file index. It was extended after the v1 freeze as later stages were added, and the change was not logged at the time; `DEVIATIONS.md` records it (2026-10-25). It holds no decision rule and no analysis reads it. Every other hashed file matches, in all three freezes.

## Reproducing

**Environment.** Python 3.12 with `requirements.txt` (CUDA build of torch) and one GPU with 8 GB or more. The runs used an 8 GB laptop GPU, with the model in bf16 and thinking off.

**Notes.**
- The original runs went through a shared GPU-lock wrapper (`tools/gpu_lock.py`), which is not part of this release. The stage scripts run without it: call the `python` lines in `code/run_chunks.sh`, `code_v2/run_v2.sh` and `code_v3/run_v3.sh` directly. The hashed shell scripts name the original Python interpreter path; replace it with yours.
- Trained adapters and the injection vectors are withheld (see `DATA.md`); the training stages rebuild them.
- Stage 0a and 0b stored their hidden-state features outside the study folder. `s0a_extract.py` and `s0b_extract.py` rebuild them; the paths are set by the `S0_FEAT` and `S0B_FEATDIR` environment variables.

**Order.** Each driver script lists its stages; the docstrings give the arguments.
- v1: `code/job.py <stage>` for each stage in `code/run_chunks.sh` (the frozen bed check stops after the pilot), then `code/analyze.py`.
- v2: `code_v2/job_v2.py`, then `code_v2/analyze_v2.py`.
- v3: `code_v3/job_v3.py <stage>` for each stage in `code_v3/run_v3.sh`, then `code_v3/v3_analyze.py`.
- Stage 0a: `stage0/s0a_extract.py` (GPU), then `stage0/s0a_probe.py` (CPU). Stage 0b: `stage0b/s0b_extract.py`, `s0b_train.py` and `s0b_tailpos.py`.

**Numbers.** `python research_next/p9_trained_introspection/paper/make_numbers.py` rewrites `numbers.json` and the figures from the result files (CPU only). The audits' recomputation is `audit/paper_audit/recompute_paper.py`; it imports no study code.

## AI use and author contribution

This study was carried out by an AI system: Claude (Anthropic), Claude Opus 5.5 (`claude-opus-5-5`), in every session, including the audit and reviewer sessions. It ran as Claude Code agent sessions on the author's computer, and it:

- checked the novelty of the idea;
- chose the design of every stage and wrote each preregistration;
- wrote all the code;
- ran every experiment on the author's hardware;
- analysed the results;
- searched for and checked the cited literature;
- wrote the manuscript and this documentation.

One Claude session coordinated the others and passed on the author's requests. On the author's behalf, after the author approved a fourth study and asked for a quick small-scale check of whether it could work, it made that study's design decisions, ran Stage 0a, and set the pre-freeze rules that stopped it.

The study's question, whether a model's self-report can be trained to read its own computation rather than reconstruct it from the input, and the suggestion to answer it by training, were the author's. A related item, a trained self-report on a looped model's depth, was already on a shortlist that the AI system had produced and shown to the author the day before. The AI system proposed the method in reply: training the self-report on examples where an answer-fixed intervention changed an internal step, and testing it against the model's natural computation and an input-only predictor. It also proposed decoded neurofeedback as the human analogue, in reply to the author's question about whether people can do this. The replacement task, the probe stages and the stopping rules were the AI system's. The author started the study, asked for it to be finished overnight, approved the fourth study and its feasibility stage, and approved drafting the paper; the second study ran under the first study's frozen rules without a separate approval. The author provided the computing hardware and decided to release the study. The author did not independently check the code, analyses or technical claims.

**How the work was checked.** Independent Claude sessions acting as auditors recomputed the results from the raw files. One audit covered v1 and v2, one covered v3, and one covered every number in the manuscript. Their reports are in `research_next/p9_trained_introspection/audit/`.

## Citation

See `CITATION.cff`. Code: MIT (`LICENSE`). Paper and data on Zenodo: CC BY 4.0.

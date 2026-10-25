# Paper 9 audit brief (independent auditor, after the results exist)

Recompute every number from the raw files, check every verdict against its frozen rule, and list every overclaim. Do not trust `analysis*.json` or `RESULTS.md`: recompute from the per-row files with your own code (put it in `audit/`). Do not import the study's analysis modules.

## Record (read first)

- **Preregistrations.**
  - `PREDICTIONS.md`: v1, frozen 17:28. Its bed check failed at 22:17.
  - `PREDICTIONS_V2.md`: v2, frozen 22:40, a confirmation of the diagnosis.
  - `PREDICTIONS_V3.md`: v3, frozen 23:34, the replacement task.
- **Freeze files.** `PREDICTIONS*.sha256` and `PREDICTIONS_V3.sha256.amend`. Run `code/freeze.py check`, `code_v2/freeze_v2.py check` and `code_v3/freeze_v3.py check`.
- **Logs.** `DEVIATIONS.md` holds every change and failure. `TASK_SEARCH.md` holds the exploratory screens (logs in `diag/`).

## 1. v1 bed failure (`res_pilot/pilot.json`, `res_pilot/pilot_gates.json`)

Confirm PG1–PG5 from `pilot.json`. Confirm that no v1 main data exist: `res/` holds only `screen.json`.

## 2. v2 (`res_v2/premise.json`, rule in PREDICTIONS_V2.md section 3)

Recompute:
- the Part A calibrations, the two differences, and their bootstrap CIs (10,000 resamples over items);
- the Part B donor and recipient rates per boundary, the transition boundaries, and the item-level report/readout agreement with its CI;
- the Part C rates;
- H1–H3 and the verdict.

Gazetteer countries come from `res/screen.json`.

## 3. v3 (`res_v3/`, rule in PREDICTIONS_V3.md section 3; v1's thresholds in `code/analyze.py` GATE)

**Raw files.**
- `screen.json`: per item, correctness, lp of each digit, lp2, ent2, para_lp2, split.
- `gen.json`:
  - th (quartile cut points from train items);
  - the tests T1–T5, each intervened row with cb_int, lp2_int, answer_kept and kl;
  - the training lists;
  - the items table (bin, para_bin, ent_bin).
- `eval_N.json` and `eval_{O,I,J,X}_s{0,1}.json`: per-row picks, rotation-averaged scores and per-rotation letter log-probs, in the same order as the rows of `gen.json["tests"]`.

**Recompute:**
- **Bins.** Each item's bin from lp2 and th: bin 0 if lp2 ≥ th[2], 1 if ≥ th[1], 2 if ≥ th[0], else 3.
- **Gates G1–G5.** G3 is the agreement between the probability bin and the entropy bin.
- **T3.** The four input-only predictors (para bin; majority train bin; a learned conditional logit fit on train items only; X with seed-averaged scores). Then P\*, Δ_arm with bootstrap CIs (P\* re-chosen in each resample), and the divergent-item breakdown.
- **T1, T2, T4 and T5,** as defined in the preregistrations.
- **The verdict and its reasons.**
- **Spot checks.** For five rows, recompute rot_scores from the stored per-rotation log-probs.

## 4. Leakage and design checks

- **v3 items.** No test product appears among training items, donors or concept vectors. The bin cut points come from train items only.
- **v3 options.** Every option set contains its label.
- **Deviations.** Every deviation is logged, and none changes a frozen rule after data were seen.

## 5. RESULTS.md

Check that:
- every number matches your recomputation;
- the plain-language summary does not overclaim. In particular:
  - one model and one task per version;
  - v3's intermediate is a confidence (metacognitive) quantity, not a bridge value;
  - J and X are descriptive;
  - v2 rests on a placeholder readout;
  - no general claim about "introspection";
- the author's chat messages are not quoted.

## Output

Write `audit/AUDIT.md`:
- a verdict line for each of v1, v2 and v3: CONFIRMED, or the corrected verdict;
- a table of recomputed against reported numbers, with mismatches marked;
- a numbered list of errors and overclaims, each with a suggested fix.

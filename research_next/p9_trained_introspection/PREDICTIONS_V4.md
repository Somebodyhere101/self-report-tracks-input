# Paper 9, v4 preregistration: an immediate, learnable version of the confidence self-report test

## Plain-language summary

v3 asked whether a small model can be trained to say how sure it was of a digit it wrote by reading its own internal state, rather than by guessing from the question. It gave no answer, because none of the trained reporters learned the task.

A quick check made today (Stage 0) showed why. Where v3 asked for the report, in a new turn after the answer, the swapped-in confidence was no longer there, and the model worked its confidence out again from the question.

So v4 asks for the report immediately, with a short question added right after the answer's digits, while still keeping the old after-the-answer report for comparison. It also uses two choices ("Very sure" or "Less sure") instead of four, and tries larger training budgets step by step.

Before any test data, both main reporters must show on practice problems that they can learn:
- the swap-trained reporter must name the swapped-in confidence on new swaps;
- the ordinary reporter must do as well as the best predictor that sees only the question.

If no budget achieves both, the result is UNINFORMATIVE. Otherwise v3's comparison decides: does swap training give a self-report that tracks the model's real confidence better than the question alone, and does it carry over to kinds of internal change it never saw?

{{S0B_SUMMARY}}

## 1. Status and attribution

- **History.**
  - v1 (`PREDICTIONS.md`) failed its bed check.
  - v2 confirmed why.
  - v3 (`PREDICTIONS_V3.md`) ran on multiplication confidence. It came out NOT SUPPORTED, and the audit confirmed this. But no arm learned at all: training losses stayed at ln 4 and test accuracies at chance.
- **Attribution.**
  - The core idea is the author's (Sam Ramdan).
  - After the author approved a v4, the coordinating session made the v4 decisions on his behalf, paraphrased in section 9:
    - a learnability gate;
    - a budget ladder;
    - two bins;
    - unchanged arms, tests and rule;
    - a transfer secondary;
    - freeze and audit;
    - operations;
    - reporting;
    - after Stage 0a, an immediate report as the primary format, with the delayed report as a secondary arm.
  - The refinements, thresholds, code and this document are Claude's (Opus 5.5), and every refinement is listed in section 9.
- **Stage 0, exploratory and on pilot products only. It was run before this freeze and shaped v4:**
  - **Stage 0a** (`stage0/STAGE0.md`) was run by the coordinating session. In the untrained model, the swapped-in two-bin confidence is not linearly readable at the turn-2 report token: a probe trained on natural rows agrees with the recipient's own bin (0.60–0.64), not the donor's. At the answer positions from layer 16 on, the probe follows the donor (0.61–0.66).
  - **Stage 0b** (`stage0b/STAGE0B.md`, mine) repeated 0a's probe at both new report positions and ran tiny training runs on pilot half A only. {{S0B_FACTS}}
- **Other pre-freeze work, all on pilot items or the base-model item screen:**
  - `res_v4_design/design.json` (two-bin design numbers);
  - two code tests: a smoke run of the whole chain, and a deadline-and-resume test of the training loop, which matched an uninterrupted run bit for bit;
  - a third smoke run of the final code (section 8).
- **What did not exist at freeze time:** any v4 ladder data, any v4 main data, and any use of pilot half B or of the train or test products by a trained reporter.

## 2. Task, label, region and report formats (`code_v4/v4lib.py`, `code_v4/v4stats.py`)

- **Task, items, splits and lp2.** All are v3's, unchanged:
  - the prefilled multiplication "What is a * b?" / "a * b = " plus the four digits;
  - lp2 is the model's own log-probability of its emitted second digit at the first answer position.
- **Turn 1 now ends with the four digits.**
  - Region R is the four digit positions, and every intervention acts on them only.
  - lp2 and every turn-1 label are unchanged, because they depend only on positions up to the first digit.
- **Label (changed: two bins).**
  - Bin 0, "Very sure": lp2 ≥ med, the median lp2 of the correct train items (−0.001326, i.e. p2 ≈ 0.99867).
  - Bin 1, "Less sure": below med.
  - The train bins are 556/555 and the test bins 585/563.
  - **In plain words, "Less sure" is still confident.** Among the 555 train items in that bin, the median probability of the second digit is 0.982, and 76% are above 0.90. The two labels separate "near-certain" from "very likely".
  - v3's four-bin result stands as the four-bin answer and is not re-run.
- **Report formats.** Both are run on the same turn-1 computation. The adapter is active on the report tail only, and the report position is never inside R. If it were inside R, a swapped run's report would be the donor's own.
  - **Immediate (primary).** The answer continues, right after the digits: "How sure am I of the second digit of this answer? / A) Very sure / B) Less sure / The answer is (" (27 tokens).
  - **Delayed (secondary).** The answer is closed (`<|im_end|>`), and v4's turn-2 question follows: "How sure were you of the second digit of your answer?" with the same options and "Answer with the letter." (45 tokens).
- **Scoring.**
  - Both option orders are used, and an option's score is its mean log-probability over the two orders, with a softmax over the letters A and B.
  - The letter logits are computed in fp32. Exact ties are broken by a seeded coin; in v3, bf16 ties covered 49–63% of rows and all went to the first option.
- **Why two bins.**
  - **Learnability is the reason for the change.** One cut at the median replaces three cuts at p2 ≈ 0.982, 0.9987 and 0.99993. In Stage 0a, linear readability at the answer position is 0.74 with two bins and 0.44 with four.
  - **Power is the cost, checked on v3's item screen.** The best input-only predictor (section 4) scores 0.867 on the test items, against 0.597 for v3's four-bin bar. So 153 of 1,148 items diverge from it, instead of 463. Power stays at 1.00, at n = 1,148 and at n = 600, under v1's strong reader (0.97 / 0.70).

## 3. Arms and training (`code_v4/trainlib_v4.py`)

- **Primary arms (immediate report).**
  - **O:** natural labels.
  - **I:** same-product swaps of R from layer 16, alternating different-bin donors (whose bin also differs from the recipient's paraphrase bin) and same-bin donors. The label is the donor's bin.
  - **J:** alternates the other bin's vector injected at R and the recipient's own bin.
  - **X:** reads the transcript as text.
  - **N:** untrained.
  - With two bins, a "different" donor or injection exists only for recipients whose bin equals their paraphrase bin.
- **Secondary arms (delayed report).**
  - **Id:** arm I's examples, trained and tested with the delayed report.
  - **Nd:** untrained, delayed.
- **Training.**
  - Rank r LoRA on all 28 blocks, alpha 2r, adapter on the report tail only.
  - Adam, a 10-step linear warm-up then a constant lr, gradient clip 1.0, batch 8, one random option order per example, full-vocabulary cross-entropy, seeds 0 and 1.
  - The rank, lr and number of steps S come from the ladder (section 5), the same for every arm. There are S × 8 examples per arm.
  - **Order of examples.**
    - In the ladder, which uses one seed, the examples are a fixed stream read in order, with option orders seeded by step. A run of S steps is then an exact prefix of a longer one.
    - In the main phase, each seed reads the arm's examples in its own seeded order, as v3 did.
    - Id reads exactly I's examples, in I's order and with I's option orders. I and Id differ only in the report format.
- **Operations.**
  - Checkpoints every 200 steps and at every ladder stop. A checkpoint records rank and lr, and resuming checks both.
  - Training (in the ladder and the main phase) saves and exits 3 when the gpu_lock deadline is near, and the job is re-queued (`--max-min 90`).
  - The job driver starts any other stage only with at least 8 minutes left.
  - 4 CPU threads, and a wait for mains power below 30% battery.
  - The `pilot` and `main` jobs refuse to run unless `PREDICTIONS_V4.sha256` verifies.

## 4. Input-only predictors (P\*), re-chosen in every bootstrap resample

- **para:** the bin of the paraphrase prompt's lp2 at med (test 0.812).
- **recal_para** (new, from the v3 audit): the paraphrase lp2 cut at its own train median, −0.00465 (test 0.867).
- **learned** (refined features): v1's conditional logit on the train items. Its features are the paraphrase bin, plus the bin-1 indicator alone and times u, the standardised log(−paraphrase lp2) (test 0.846).
- **majority:** test 0.510.
- **X:** the trained external observer.

## 5. Learnability gate: the budget ladder on pilot half B (`code_v4/v4_run.py ladder`)

**Data.**
- Pilot products are split in half by product (seeded): half A (314 items) is for training and half B (276 items) is held out.
- Half B has not been used by any trained reporter before this freeze.
- I and O are trained on half A with the immediate report, seed 0.

**Ladder.** Four (rank, lr) settings are each trained once and tested at each stop they reach. In order of cost:

| Rung | Rank | lr |
|---|---|---|
| 300 steps | 8 | 1e-4 (v3's budget) |
| 300 steps | 8 | 1e-3 |
| 300 steps | 32 | 3e-4 |
| 300 steps | 32 | 1e-3 |
| 1,200 steps | 32 | 3e-4 |
| 1,200 steps | 32 | 1e-3 |
| 3,000 steps | 32 | 3e-4 |
| 3,000 steps | 32 | 1e-3 |

**Gate for a rung.** Both conditions must hold:
- **I:** names the donor's bin on at least 2/3 of half B's different-bin swaps (70 rows) AND at least 2/3 of its same-bin swaps (116 rows).
  - 2/3 is the equal-difficulty translation of 0.50 with four bins. Both cover one third of the distance from chance to perfect (κ = 1/3): (0.50 − 0.25)/(1 − 0.25) = (0.667 − 0.5)/(1 − 0.5).
  - Requiring both kinds of row excludes a reporter that only flips when it detects tampering, and one that follows the input.
- **O:** natural-item accuracy on half B ≥ P\*_pilot. This is the best of para, recal_para, majority and learned on half B, with the last two fitted on half A: 0.888, from recal_para.

**Rules.**
- The cheapest rung that passes is used for every arm, and the ladder stops there.
- **If no rung passes, the verdict is UNINFORMATIVE**, and the ladder table is reported.
- **Exploratory fallback** (it never changes the verdict). If no rung passes, but at some rung I passes its condition and O reaches at least 2/3 on half B's natural items, the main phase still runs at the cheapest such rung. Its results are reported as exploratory only.

**Two-bin bed checks on half B, before training.** If either fails, the result is UNINFORMATIVE:
- the probability bin and the entropy bin agree on ≥ 0.70;
- swaps carry the donor's bin on ≥ 0.90 of different-bin rows.

**Tamper-control scale.** The ladder setup recalibrates T4's KL-matched scale s_kl for the four-digit R, by v3's rule on 120 half-A different-bin pairs.

## 6. Main phase, tests and decision rule (`code_v4/v4_run.py gen/train/eval`, `code_v4/v4_analyze.py`)

**Fixed from v3's frozen pilot (`config_v3.json`).** l1 = 16, ls = 22, la = 18 and s_add = 2; s_kl comes from section 5. The bin vectors and the swap norm are recomputed for two bins from the train items over the four-digit R.

**Tests** (primary arms, immediate report; the secondary arms get the same tests):
- **T1:** held-out same-product swaps from layer 16, on different-bin and same-bin rows.
- **T2a:** a single-layer swap at layer 22.
- **T2c:** the other bin's vector at layer 18, scale 2. It is applied to those of the first 400 test items whose bin equals their paraphrase bin: 342 items.
- **T2 changed rows:** the intervened bin differs from both the natural bin and the paraphrase bin. A family is evaluable with ≥ 20 such rows.
- **T2 bar: 1/2**, chance with two options. v3's bar of 1/3 was a four-option tamper detector's level. With two options, a tamper detector would score 1.0, so T2 is credited only if I passes the KL-matched tamper control (T4-kl, defined next).
- **T3:** the 1,148 natural test items. Δ = the arm's accuracy minus P\*'s, with a 95% bootstrap CI (P\* re-chosen per resample).
- **T4:** KL-matched and norm-matched random directions at R. Pass if the upper CI bound of the increase in switching is ≤ 0.15.
- **T5:** 48 general multiple-choice questions in a new turn, and multiplication with the adapter on everywhere.

**Gates G1–G5 (v3's).**
- G1: test accuracy ≥ 0.70, with ≥ 150 items.
- G2: digits kept in ≥ 0.95 of T1 swaps.
- G3: the probability and entropy bins agree on ≥ 0.70 (a weak check).
- G4: different-bin swaps carry the donor's bin in ≥ 0.90.
- G5: at least 25 divergent items, with power ≥ 0.80.

**Verdict.** v3's rule with the two-bin T2 bar, on the primary (immediate) arms:
- **SUPPORTED** requires all of:
  - Δ_I's lower bound > 0;
  - Δ_O's lower bound not > 0;
  - the T2 pooled lower bound (clustered by item) > 1/2, with every evaluable family above 1/2;
  - I passes T4-kl.
- **NOT SUPPORTED** when the gates pass but any of those fails. The reasons are listed.
- **UNINFORMATIVE** in any of these cases: the learnability gate, a bed check or a gate fails; neither T2 family is evaluable; or the main phase ran only as the exploratory fallback.

**Key secondary: TRANSFER**, per family (T2a and T2c), for I. It is also reported for O, J and Id.
- **YES:** the lower bound of the clustered 95% CI on changed rows is > 1/2, the point estimate is ≥ 2/3, and the arm passes T4-kl.
- **PARTIAL:** only the lower bound is > 1/2.
- **NO:** otherwise, or if T4-kl fails.
- **NOT EVALUABLE:** fewer than 20 changed rows.

**Secondary: immediate against delayed.** I − Id on:
- T3;
- T1 different-bin rows;
- T1 same-bin rows;
- the T2 changed rows.

Each comes with a 95% CI paired by item. TRANSFER and the T3 Δ are reported for both. This comparison is descriptive and does not enter the verdict.

**Other secondaries (descriptive):** the ladder table, the divergent-item breakdown, and each arm's share of first-option picks.

## 7. My predictions (recorded before any v4 ladder or main data)

{{PRIORS}}

## 8. Compute and deviations

- **Estimate.**
  - The ladder takes about 0.3 GPU-h if the first rung passes, and at most about 1.6 GPU-h for all eight rungs.
  - The main phase has 10 training runs and 12 evaluations: about 0.5, 1.4 or 3.1 GPU-h at 300, 1,200 or 3,000 steps.
  - Total ≤ 4.7 GPU-h.
- **Driver.** `code_v4/run_v4.sh` runs the pilot (ladder and gate), then the main phase, then the CPU analysis.
- **Pre-freeze tests.** A smoke run of the final code is in `res_pilot_v4_smoke/` and `res_v4_smoke/`, on pilot items with tiny budgets. Its numbers are not results.
- **Deviations** go in `DEVIATIONS.md`. An independent audit follows the run.

## 9. Decisions and refinements (the coordinating session's v4 decisions, paraphrased, and what was made of them)

1. **Learnability gate before any main data, on pilot products.** The decision sets 0.50 for I with four bins and asks that O match P\*'s pilot accuracy.
   - The I bar becomes 2/3 with two bins, on both different-bin and same-bin swaps.
   - O's bar is used as given: P\*_pilot, including the recalibrated paraphrase predictor.
   - **Interpretation.** The decision's wording names UNINFORMATIVE for the case where the conditions are not met, and says no main test should run on a reporter that cannot learn. I read this as requiring both conditions, because v3's rule compares I with O.
   - Added: the exploratory fallback, which never changes the verdict.
2. **Budget ladder:** steps 300 → 1,200 → 3,000, rank 8 → 32, lr in {1e-4, 3e-4, 1e-3}, with the cheapest passing rung used for every arm. This became eight rungs, as nested stops of four (rank, lr) settings.
3. **Two bins primary, four secondary.** Two bins, at the train median. The four-bin answer is v3's result. The example count scales with the rung.
4. **Unchanged.** Arms; T3 against P\* re-chosen per resample; T1, T2 and T4; the rule; swaps from layer 16.
   - **Adapted:** the T2 bar is 1/2; the "different" cases are restricted to items whose bin equals their paraphrase bin; R is the four digits (decision 9); s_kl is recalibrated for that R.
   - **Strengthened:** P\* gains recal_para, and the learned predictor's features changed.
   - **Changed:**
     - letter scoring is fp32 with seeded tie-breaks;
     - examples are read as a stream in the ladder, and in a seeded per-seed order in the main phase (section 3);
     - the KL medians of the T1 swaps and the T4 directions are reported, to show how well "KL-matched" holds.
5. **Transfer secondary.** TRANSFER, with its own thresholds (section 6).
6. **Freeze and audit.** `PREDICTIONS_V4.sha256` is written before any v4 ladder or main data. The priors are in section 7, and an independent audit follows.
7. **Operations.** `--max-min 90`, checkpoints, exit 3 near the deadline, 4 threads and the battery wait.
8. **Reporting.** The verdict is reported in plain words, with the transfer result.
9. **After Stage 0a: the report must be elicited where the information lives.**
   - The primary report is an immediate confidence question appended right after the four digits, with the adapter active only on it. The delayed report is kept as a secondary arm.
   - Before the freeze, the Stage 0a probe method was to show the swapped-in bin readable at the new report position. The bar is a natural-trained probe agreeing with the donor > 0.60 with two bins.
   - **Implementation.** The decision describes the tag as inside R, in the swapped layers. Read literally, a swapped run's report would then be the donor's own report, I's labels would become natural labels on donor items, and T1 would stop testing reading. So R is the four digits and both report tails follow it (section 2).
   - **The probe check's outcome:** {{PROBE_OUTCOME}}

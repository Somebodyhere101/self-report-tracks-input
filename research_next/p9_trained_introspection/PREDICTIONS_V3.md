# Paper 9, v3 preregistration: can a model be trained to read its own confidence, computed after the input?

## Plain-language summary

The first task (which country a "which continent" question goes through) could not be used. In this model the country is held in the same place as the city name itself, so "reading the computation" and "reading the input" are the same thing (v2 confirmed this on about 300 cities). Tonight's task search therefore looked for a step the model computes after it has read the question.

The task that passed is multiplication. The model writes the product of two two-digit numbers, and how sure it is of the second digit is computed at the answer positions, after the question. We can swap that internal state for the one from a different multiplication with the same product: the question and the written answer stay the same, but the model's confidence becomes the donor's.

We then train the model to say how sure it was, either on ordinary examples or on swapped ones. The decisive test is on ordinary, untouched problems: does the trained answer track the model's real confidence better than the best predictor that sees only the question? Everything else is the same as the original plan: the arms, the tests and the decision rule.

## 1. Status and attribution

- **History:**
  - v1 (`PREDICTIONS.md`) failed its bed check.
  - v2 (`PREDICTIONS_V2.md`) confirmed why: SAME WINDOW on 306 known and 200 paired test cities.
  - The coordinating session, relaying the author's wish that everything be done tonight, asked for a replacement task. The task should have an intermediate computed after the input. It should be screened with v1's gates and preregistered as a frozen addendum, using the same arms, tests and decision rule.
- **Attribution.** The core idea is the author's (Sam Ramdan). The task choice, adaptations and code are Claude's (Opus 5.5).
- **Task search.** `TASK_SEARCH.md` records it; every screen was exploratory, on CPU, with ad-hoc items. The candidates tried:
  - sum compared with c: yes-bias, 21/40 by forced choice;
  - first or last digit, or tens, of a sum: 2–11/40;
  - above or below 50: 29/40;
  - carry route: the sum was right 24/24, but the carry question failed 12/24 even with the operands visible;
  - least-sure digit of a product: degenerate, always the second digit;
  - **second-digit confidence of a product: passed.** Accuracy 41/48; probability and entropy labels agree 47/48; a same-product swap moves the label 10/10; the untrained report says "sure" 24/24.
- **What existed at freeze time:** those screen logs (`diag/`) and the v1/v2 files. No v3 item screen, pilot, training or evaluation existed.

## 2. Task and ground truth (`code_v3/v3lib.py`)

- **Items.** Every a·b with 11 ≤ a ≤ b ≤ 99 and a four-digit product. Splits are by product: a seeded cycle of train, test, pilot, train, test.
- **Turn 1.** User: "What is a * b?". Assistant: "a * b = " followed by the product's four digits, teacher-forced. Only items whose four digits the model gets right (argmax) are used.
- **Intermediate.** The model's own log-probability of the emitted second digit, lp2. It is computed at the first answer-digit position, after the input. This is the computation's actual output distribution, not a lens or a probe.
- **Label.** The quartile bin of lp2, with cut points from the train items: "Completely sure", "Very sure", "Fairly sure", "Not very sure".
- **Region R.** The answer positions: the four digits and the closing `<|im_end|>\n`. The operands are never touched.
- **Ground truth for any run.** The bin of its own measured lp2, so no readout is needed. Swapping R at layers ≥ l from a same-product donor makes the swapped run's lp2 equal the donor's.

## 3. Same arms, tests and rule as v1, adapted (all adaptations listed)

**Arms O, I, J, X, N, with v1's budget** (rank-8 LoRA, all blocks, adapter in turn 2 only, Adam, 300 steps × 8, 2,400 examples, seeds 0 and 1):
- **Labels:** O and X get the natural bin. I gets the donor's bin. J gets the injected bin.
- **Report** (turn 2): "How sure were you of the second digit of your answer?", with the four bins as options in four rotations.
- **I examples** alternate:
  - different-bin donors, whose bin also differs from the recipient's paraphrase bin;
  - same-bin donors.

  Both kinds use the same product.
- **J examples** alternate an injected bin vector that differs from both the recipient's bin and its paraphrase bin, and the recipient's own bin. A bin vector is the mean R state at layer la of the train items in that bin, minus the overall mean.
- **X** reads the transcript as text and is asked "How sure do you think the assistant was of the second digit of its answer?".

**Tests:** T1 (F1 swaps on held-out products), T2 (F2a single-layer swap and F2c bin vectors; chance 1/3; ≥ 20 changed rows), T3, T4 (KL-matched and norm-matched random directions; pass if the upper CI ≤ 0.15) and T5. The changes from v1:
- **T5 ungated:** multiplication accuracy with the adapter on everywhere.
- **T3 input-only predictors** (P\* is the most accurate, re-chosen in every bootstrap resample):
  - **para:** the bin of lp2 under a paraphrased prompt ("Multiply a by b." / "a x b = "), the analogue of v1's 1-hop answer;
  - **majority:** the most common train bin, the model-free analogue of v1's gazetteer;
  - **learned:** a conditional logit (v1's `cond_logit_fit`) over the bins, fit on train items, using the paraphrase bin, |bin − paraphrase bin|, and the bin indicators times the standardised paraphrase lp2;
  - **X.**
- **T2 changed rows:** the intervened bin differs from both the recipient's natural bin and its paraphrase bin. This is the analogue of v1's "≠ ref and ≠ gazetteer".

**Gates, with v1's thresholds (`code/analyze.py` GATE):**
- **G1:** test accuracy ≥ 0.70, and ≥ 150 usable test items.
- **G2:** digits kept in ≥ 0.95 of T1 swaps.
- **G3** (adapted; there is no readout, so no calibration): the probability-bin and entropy-bin labels agree on ≥ 0.70 of T3 items.
- **G4:** the different-bin T1 swaps carry the donor's bin in ≥ 0.90.
- **G5:** natural divergence from P\* on at least 25 items, with power ≥ 0.80, as in v1.

**Verdict, as v1:**
- **SUPPORTED** requires all of: Δ_I's lower CI bound > 0; Δ_O's lower bound not > 0; T2 pooled lower bound > 1/3 with every evaluable family above 1/3; and I passes T4-kl.
- **NOT SUPPORTED** when the gates pass but any of those fails. The reasons are listed.
- **UNINFORMATIVE** when a gate fails.

## 4. Pilot rules (`code_v3/v3_run.py pilot`; pilot products only)

v1's rules on the same grids:
- **l1:** the largest B ≤ 16 with swap transfer ≥ 0.90.
- **ls:** the single layer with the largest transfer.
- **la, s:** the largest change-to-injected-bin rate with digits kept ≥ 0.90.
- **s_kl:** matched to F1's median R-KL.
- **lr:** the better of 1e-4 and 3e-4 for arm I trained on pilot half A, tested on half B.

## 5. Bed checks (`code_v3/make_config_v3.py`)

If any check fails, the chain stops before main data:
- **PG1:** accuracy over all items ≥ 0.70, and ≥ 150 usable test items.
- **PG2:** the probability-bin and entropy-bin labels agree on ≥ 0.70 of pilot items.
- **PG3:** F1 transfer at l1 ≥ 0.90.
- **PG5:** the smoke pass (gen, training, evaluation and analysis on pilot items with tiny budgets, in `res_v3_smoke/`) completes.

## 6. My predictions (recorded before any v3 data)

| Outcome | Prior |
|---|---|
| UNINFORMATIVE through G5 (the paraphrase predictor tracks lp2 too closely) | 0.35 |
| NOT SUPPORTED: I fails T3 | 0.30 |
| NOT SUPPORTED: O also beats the bar, or T2 at chance | 0.15 |
| SUPPORTED | 0.20 |

- **T1:** I learns the trained family, while O and N stay with the recipient's own bin.

## 7. Run, budget, deviations

- **Driver:** `code_v3/run_v3.sh` runs pilot, then main, then the CPU analysis. Every GPU job goes through `tools/gpu_lock.py` as `p9v3-*`, yields after 45 minutes, and waits for mains power if the battery is below 30%. Each job is limited to 4 CPU threads.
- **Freeze check:** `code_v3/job_v3.py` refuses to run unless `PREDICTIONS_V3.sha256` verifies.
- **Estimate:** about 2 GPU-h; 3.7 h of the budget remains.
- **Deadline:** if the bed fails, or nothing passes by 02:00, the result is UNINFORMATIVE with the search record.
- **Deviations** go in `DEVIATIONS.md`, and an independent audit follows.

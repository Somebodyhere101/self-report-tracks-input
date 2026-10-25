# Paper 9 results: can a model be trained to read its own computation? (Qwen3-1.7B, 2026-10-21/22)

## Plain-language summary

We asked whether a small language model (Qwen3-1.7B) can be trained so that, when asked about its own processing, it reports what actually happened inside it rather than re-deriving an answer from the question.

The first task we planned could not be used. Its checks, frozen in advance, failed: a placeholder readout of the model's states after the city name could not recover which country a "which continent" question goes through. A diagnosis and then a preregistered follow-up on 306 held-out cities found that the country later questions name is taken from the city name's own internal states. That was true both for the model's self-report and for our placeholder readout. So, with these instruments, "reading the computation" and "reading the question" could not be told apart.

A same-night search for a replacement found one task that passed v1's checks as adapted to it: how sure the model is of the second digit of a multiplication answer. That confidence is computed after the question, and it can be swapped internally without changing the question or the answer. Unlike the planned bridge country, it is the model's confidence in its own answer (a metacognitive quantity), not an intermediate value of the task. So this test is of trained confidence reporting.

On that task the preregistered verdict is NOT SUPPORTED. The version trained on internally swapped examples did not report its natural confidence better than a predictor that only sees the question, and it did not carry over to new kinds of internal change. No version learned to report the confidence of untouched problems under the frozen training budget: not the swapped-example version, the ordinary-example version, the injection-trained version, nor an outside observer. So the comparison between label sources could not be made: none of them taught the report at this budget and in this format. An exploratory check found only a small sensitivity of the swapped-example version to swapped problems.

A planned fourth study (v4) would have used two confidence levels, larger training budgets and a question asked right after the answer. It was stopped before its freeze by a frozen-in-advance check (Stage 0). Probes showed that the swapped-in confidence is present at the answer digits, but not where the model computes its report. There, the model's internal state follows the confidence that the question implies, whether the report is asked right after the answer or in a new turn. Short training (up to 900 steps) did not change this (section 9).

This covers one model, two tasks and the training budgets listed. It does not show that trained self-report is impossible. All decision rules were frozen before the data existed, every change is logged in DEVIATIONS.md, and independent audits recomputed the numbers.

## 1. v1: the planned task failed its frozen bed check (no main data)

**The plan.** `PREDICTIONS.md` (v1, frozen 17:28) used the two-hop question "Which continent is the country containing {city} in?". Swaps act on the computation after the city name, region R. The question was whether a self-report trained on such swaps tracks the model's natural computation better than predictors that see only the input. Before any main data, frozen checks had to show that a placeholder readout of R can tell which country the computation carried.

**The checks** (`res_pilot/pilot_gates.json`). PG1 uses the full item screen; PG2 and PG3 use pilot cities.

| Check | Value | Threshold | Result |
|---|---|---|---|
| PG1: 2-hop accuracy over 1,008 usable cities | 0.954 (962/1,008) | ≥ 0.70 | pass |
| PG1: usable test cities | 363 | ≥ 150 | pass |
| PG2: readout calibration on 164 known-like pilot cities | 0.116 (19/164)ᵃ | ≥ 0.80 | **fail** |
| PG2: readout reliability (country and capital readouts agree) | 0.309 (60/194 usable pilot cities; 0.299 on the 164 known-like) | ≥ 0.70 | **fail** |
| PG3: swap transfer at l1 = 0 (reliable rows) | 0.754 (49/65) | ≥ 0.90 | **fail** |
| PG5: end-to-end self-test on pilot cities | completed | — | pass |

ᵃ The frozen code counts unreliable readouts as misses. Without the reliability filter (v2's definition), the same data give 0.354 (58/164).

**Outcome.** The chain stopped before generating main data. No reporter was trained or evaluated on test cities; the learning-rate check and the self-test trained reporters on pilot cities only.

**What the swap-transfer figures show.** "Transfer" was flat at 0.73–0.78 whether R was swapped from layer 0, from layer 20, or at a single layer. So the readout showed no detectable dependence on R.
- A likely explanation, not tested: the 19 known donors were selected because the readout already named their own country, and 7 of them are in India or Mexico, among the readout's most frequent picks. A readout that ignores R would then "transfer" to them at about this rate.
- The untrained self-report on these swaps named the recipient's (visible) city's country on 86% of rows and the donor's on 0%.

## 2. Diagnosis (exploratory; 12 pilot cities; CPU; `diag/`)

**Readout variants** (`diag/diag_readout.log`):

| Placeholder readout fed... | Letters | Country names |
|---|---|---|
| nothing (prior) | 3/12 | 4/12 |
| R at every layer (the v1 readout) | 3/12 | 4/12 |
| R and the city tokens at every layer | 12/12 | 11/12 |
| (city visible, no placeholder) | 12/12 | 11/12 |

The answer format is not the problem. Scoring country names directly gives the same pattern. On one city checked, the letters carried essentially all of the next-token probability.

**Swap boundary** (`diag/diag_city.log`; 12 known cities, 12 pairs):

| From layer l | 0 | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| Readout fed the city tokens and R only at layers ≥ l (= gazetteer) | 12/12 | 12/12 | 12/12 | 4/12 | 3/12 | 3/12 |
| Untrained report under a city+R swap from layer l: names the donor | — | 12/12 | 11/12 | 0/12 | 0/12 | 0/12 |
| The same readout of the swapped run (layers ≥ l only): names the donor | — | 12/12 | 11/12 | 4/12 | 3/12 | 2/12 |

On 12 pairs, both the report and the readout switched between layers 8 and 12. This was exploratory; v2 tested it.

## 3. v2: preregistered test of the diagnosis on held-out cities (`PREDICTIONS_V2.md`, frozen 22:40)

**Verdict under the frozen v2 rule: SAME WINDOW.** Under this rule, the bed cannot separate reading the computation from reconstructing from the input in Qwen3-1.7B. So v1's training contrast cannot be built in this task. All three hypotheses hold (`res_v2/analysis_v2.json`), and the independent audit confirmed every number (`audit/AUDIT_v1v2.md`).

**The instrument.** v2 rests on a placeholder readout. The city is replaced by a placeholder, the states of a run are copied in, and the model is asked "Which country was the question about?". This shows which country a second question can recover from the copied states. It does not show which country the two-hop computation itself used, because donor and recipient share the continent answer by design.

**Part A** (306 known-like test cities; 95% bootstrap CIs):

| Readout fed... | Calibration [95% CI] |
|---|---|
| nothing (prior) | 0.314 [0.261, 0.366] |
| R at every layer | 0.353 [0.301, 0.408] |
| R and the city tokens at every layer | 0.961 [0.938, 0.980] |

- City+R minus R is +0.608 [0.549, 0.660].
- R minus prior is +0.039 [−0.007, 0.085]. R gives the readout no detectable country information, though an effect of up to about 8 points is not excluded.
- Copying R changes the readout's pick on 31% of items without raising calibration significantly (exact McNemar p = 0.13).

**Part B** (200 pairs; the city tokens and R swapped from a different-country, same-continent donor at layers ≥ B). The share of rows naming the donor:

| B | 4 | 6 | 8 | 9 | 10 | 11 | 12 | 14 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| untrained self-report | 0.965 | 0.960 | 0.935 | 0.935 | 0.905 | 0.840 | 0.025 | 0.015 | 0.015 | 0.015 |
| placeholder readout of the swapped run (all layers copied) | 0.965 | 0.955 | 0.910 | 0.900 | 0.890 | 0.795 | 0.025 | 0.020 | 0.020 | 0.020 |

- **Transition.** Both switch at B = 12.
- **Agreement.** Item-level agreement between the two is 0.9645. Pair-resampled CI [0.951, 0.977].
- **Chance level.** Two readers following these curves independently would agree on 0.863, so the chance-corrected κ is 0.74 (0.68 at B = 11).
- **Asymmetry.** The report names the donor while the readout does not on 40 rows at B = 4–11, against 15 the other way (exact McNemar p = 0.001). The report follows the swap slightly more often than the readout.

**Part C** (R alone swapped at every layer, v1's training family). The untrained report names the donor on 0.02 of 200 rows and the recipient's own country on 0.945.

**Reading.** In this task and model, the country that later questions name is taken from the city-token states at layers up to about 11. A swap of the city tokens and R from layer 11 upward moves both the report and the readout to the donor (0.840 and 0.795). A swap from layer 12 upward moves neither (0.025). A swap of R alone barely moves the report.

## 4. Task search (exploratory screens; `TASK_SEARCH.md`, `diag/`)

At the coordinating session's request (the author wanted everything done tonight), I searched for a replacement task whose intermediate is computed after the input. Each candidate faced v1's gates, and the search was to stop at the first pass.

| # | Candidate | Result | Gate failed |
|---|---|---|---|
| 1 | Single-digit sum, answered even or odd | answered "Even" to all 16; the sum was barely readable after the operands (5/16; prior 1/16) | task, readout |
| 2 | Sum compared with c ("Answer yes or no") | always "Yes" (21/40) | task |
| 3 | First digit, last digit or tens of a sum (forced choice) | 10/40, 2/40 and 11/40 | task |
| 4 | Sum above or below 50 | 29/40; not pursued | borderline |
| 5 | Carry route (the model writes the sum) | sums 24/24, but "did this addition require carrying?" got 12/24 (always "No") even with the operands visible | readout |
| 6 | Product, least-sure digit | always the second digit (20/20); no swap can move it | label degenerate |
| 7 | **Product, confidence in the second digit (quartile bins)** | **accuracy 41/48; probability and entropy labels agree 47/48; same-product swaps move the label 10/10; untrained report "sure" 24/24** | **passed (23:26)** |

Each screen used 6–48 items and was exploratory. "Failed" means failed this screen; it does not mean the task could not work in another format.

**Logs.** Candidates 1 and 2 have their original logs. Candidates 3–7 (`screen_formats*.py`, `screen_carry.py`, `screen_mult.py`, `screen_mult2.py`) printed to the console only. Their logs in `diag/` are deterministic CPU reruns made on 2026-10-22, each labelled as a rerun. The reruns reproduce every figure in the table above.

## 5. v3: the training study on the replacement task (`PREDICTIONS_V3.md`, frozen 23:34)

**Pilot and bed checks** (`res_pilot_v3/`; PG2 and PG3 on pilot products, PG1 on the full item screen). All passed:
- **PG1:** accuracy 0.853 over all items, with 1,148 usable test items.
- **PG2:** the probability bin and the entropy bin agree on 0.983.
- **PG3:** a swap of the answer positions from layer 16 carries the donor's bin 1.00, with all digits kept.
- **PG5:** the smoke pass completed.
- **Parameters set by the frozen rules:** training swap from layer 16; layer 22 for the unseen single-layer family (its transfer was 0.58; the lower layers were lower); bin vectors injected at layer 18 at twice the swap's size; learning rate 1e-4.
- **Learning-rate check (information only):** arm I reached different-bin accuracy 0.34 at lr 1e-4 and 0.28 at lr 3e-4.

**Bins.** The bin cut points (train items) are at log p2 -0.0181, -0.00133 and -0.0000749, i.e. p2 of about 0.982, 0.9987 and 0.99993.

**Run.** The main job was suspended overnight: lid closed, laptop unplugged, from 00:12 to 10:37. It finished after wake, and the data are unaffected (see `DEVIATIONS.md`).

**Gates** (`res_v3/analysis.json`). All pass:

| Gate | Value | Threshold |
|---|---|---|
| G1 test accuracy / usable test items | 0.864 / 1,148 | >= 0.70 / >= 150 |
| G2 digits kept in T1 swaps | 1.00 | >= 0.95 |
| G3 probability-bin vs entropy-bin agreement (T3 items) | 0.996 | >= 0.70 |
| G4 different-bin T1 swaps carry the donor's bin | 1.00 (412) | >= 0.90 |
| G5 items where the best input-only predictor errs / power | 463 of 1,148 (0.403) / 1.00 | >= 25 / >= 0.80 |

G3 (like PG2) is a weak check. The probability bin and the entropy bin come from the same output distribution at the same position, so their agreement is close to guaranteed. It does not validate the ground truth the way v1's readout calibration was meant to.

**Verdict under the frozen rule: NOT SUPPORTED.** The rule lists two reasons:
- I does not beat the best input-only predictor on T3: Delta_I -0.358 [-0.393, -0.322].
- I is not above 1/3 on the unseen intervention families: T2 pooled 0.280 [0.254, 0.306].

O did not beat the bar either. I passed the tamper control trivially, since its reports barely depend on the turn-1 state.

**T3, natural transfer** (1,148 unswapped test items; truth = the item's own confidence bin). Chance is 0.25. The test bins are 273/312/275/288, so the best constant answer ("Very sure") scores 0.272:

| Reporter | Accuracy (seed 0, seed 1) | Delta vs best input-only [95% CI] |
|---|---|---|
| N, untrained | 0.272 | -0.325 [-0.368, -0.281] |
| O, ordinary labels | 0.233, 0.234 | -0.363 [-0.397, -0.328] |
| I, swapped labels | 0.249, 0.229 | -0.358 [-0.393, -0.322] |
| J, injection labels (descriptive; not part of the verdict) | 0.235, 0.251 | -0.354 [-0.389, -0.318] |

The input-only predictors (accuracy against the truth):
- **Paraphrase bin:** 0.597. This is the model's confidence when the same multiplication is asked in another format. It is tied with the learned predictor, which made identical predictions on every item.
- **Learned conditional logit:** 0.597.
- **External observer X**, which reads the transcript as text: 0.245.
- **Majority bin:** 0.238.

**Divergent items (preregistered secondary).** On the 463 items where P\* errs, each arm's rate of naming the truth against naming P\*'s pick:
- N 0.406 vs 0.272. N always answers "Very sure", which happens to match the truth on many of these items.
- O 0.234 vs 0.162.
- I 0.279 vs 0.226.
- J 0.270 vs 0.123.

**T1, the trained intervention family on held-out products.** Chance is 0.25, and a reporter that only detects tampering would reach 1/3. The donor bins on different-bin rows are 89/114/122/87, so the best constant answer ("Fairly sure") scores 0.296:

| Reporter | Different-bin rows: names the donor's bin [95% CI] | Same-bin rows | Different-bin rows: names its own natural bin |
|---|---|---|---|
| N | 0.277 [0.233, 0.318] | 0.217 | 0.252 |
| O | 0.242 [0.210, 0.274] | 0.226 | 0.204 |
| I | 0.301 [0.267, 0.335] | 0.229 | 0.226 |
| J (descriptive) | 0.212 [0.181, 0.245] | 0.217 | 0.244 |

**T2, unseen families.** The "changed" rows are those where the intervened bin differs from both the natural bin and the paraphrase bin: 335 single-layer swaps at layer 22 and 312 bin-vector injections.

| Reporter | Single-layer swap | Bin vector | Pooled [95% CI] |
|---|---|---|---|
| N | 0.257 | 0.215 | 0.236 [0.202, 0.272] |
| O | 0.242 | 0.285 | 0.263 [0.237, 0.289] |
| I | 0.240 | 0.322 | 0.280 [0.254, 0.306] |
| J (descriptive) | 0.267 | 0.298 | 0.282 [0.255, 0.309] |

**T4, tamper control.** The change in error rate under KL-matched random directions was within +/-0.02 for every reporter. For I it was -0.011 [-0.053, 0.030], which passes the <= 0.15 rule. The norm-matched control was also close to zero. These passes are trivial, because no reporter's answer depends on the state.

**T5, capability drift.** None. General multiple-choice accuracy with the adapter in turn 2 was 0.875-0.885, against 0.896 for N. Multiplication accuracy with the adapter on everywhere was 0.995-0.998 for the trained arms (N 1.000), on 400 test items the base model answers correctly; this is a retention measure.

**Descriptive: training did not take for any arm** (`res_v3/trainlog_*.json`).
- **Losses.** In all 8 runs the training loss fell from 5-11 (the model's initial confident wrong letters) to about 1.39 within 25 steps, and stayed there to step 300. 1.39 is ln 4, the loss of guessing uniformly among the four options. The reporters learned the answer format but not the label.
- **Behaviour.** Test accuracies sit at 0.21-0.32. The untrained model always answers "Very sure".
- **Ties.** The trained reporters' letter distributions are nearly uniform (median top-letter probability 0.26-0.31). bf16 rounding makes their scores exactly tied on 49-63% of T3 rows, and ties default to "Completely sure", so their pick frequencies mostly reflect tie-breaking. Breaking ties at random changes T3 accuracies by at most 0.013.
- **Exploratory signal (audit, not preregistered).** In a label-permutation test, arm I is slightly above the level expected from its own pick frequencies on intervened rows whose bin moved away from the input-implied bins:
  - T1 different-bin rows: 0.301 against 0.253, p = 0.0006;
  - T2c changed rows: 0.322 against 0.270, p = 0.0012.

  Both survive a Bonferroni correction over the 20 pooled arm x row-set tests. The audit also ran 36 per-seed tests; over all 56, only the T1 result survives. Both seeds are above their own independence level on both row sets, but each is clearly significant on only one: on T1 rows, seed 1 has p = 0.0006 and seed 0 has p = 0.11; on bin-vector rows, seed 0 has p = 0.0018 and seed 1 has p = 0.047. The effect is small and below the 1/3 level that tampering detection alone could reach. It does not appear on same-bin swaps or on untouched items. It indicates weak sensitivity to the intervention, not reading of natural confidence.
- **What this means for the comparison.** O was trained on exactly the natural labels and X on the same examples as text, and both stayed at chance. So this test cannot tell "interventional labels do not teach reading" apart from "this label cannot be learned at this budget and in this format".

**Caveats.**
- The bins separate very small probability differences.
- **The intermediate.** v3's intermediate is a confidence (metacognitive) quantity, not an intermediate value of the task like v1's bridge country.
- **The confidence is mostly determined by the input.** On T3 items:
  - the rank correlation between lp2 and the paraphrase lp2 is 0.91;
  - the paraphrase bin is within one bin of the truth on 96.5% of items, though it runs on average 0.38 bins less sure.
- **The frozen bar was not the strongest one available.** A paraphrase predictor recalibrated at its own train quartiles (exploratory, from the audit) scores 0.718. Part of G5's divergence therefore reflects a calibration offset between prompt formats. This makes the NOT SUPPORTED verdict safer, not weaker.

## 6. What this means for the question

- **What the approach needed.** A self-report trained to read the computation needs, first, an internal step that changes while the input and the answer stay fixed, and second, a training signal the report can actually learn.
- **The two-hop task failed the first need.** In Qwen3-1.7B, the country that the self-report and the placeholder readout name is taken from the city's own token states up to about layer 11. No readable trace was found after the city: v2 bounds it at about +8 points with 95% confidence. So this bed offered no route these instruments could separate from reconstruction.
- **The multiplication-confidence task met the first need but failed the second.** Its intermediate is a confidence (metacognitive) quantity rather than a task value. Swaps changed the model's confidence while the question and answer stayed fixed, exactly as designed (G2 and G4 at 1.00). But with a rank-8 adapter, 300 steps and lr 1e-4, no reporter learned the four confidence bins from any label source. So the frozen verdict, NOT SUPPORTED, means only that interventional training produced no advantage here. It does not mean observational training worked better, since neither worked.
- **What a fair test now needs:**
  - a bed screened for a readable intermediate after the input (a placeholder readout of the post-input region must beat the no-copy prior);
  - a learnability control, run before any comparison: O must learn natural labels above chance (or coarser labels, such as top against bottom quartile, and a larger budget);
  - the strongest input-only bar, including a recalibrated paraphrase predictor;
  - then the same O-against-I comparison.
- **Planned follow-ups** (README.md; not run): replication on other cached models; 8-14B models only with the author's approval; and the chain-of-thought extension, where intermediate steps are written out and so do leave a trace after the input.

## 7. Novelty and attribution

`NOVELTY.md`: the core comparison is still open, but narrowed. Observational and interventional training have been compared for low-level signals (OpenReview 5P7dVpGNPO). In-place self-reading with natural labels has been done (arXiv 2610.17495). Neither has been done for a task intermediate under answer-fixed interventions.

The question, whether a self-report can be trained to read the computation rather than the input, and the idea of answering it by training, are the author's (Sam Ramdan). The method (training on answer-fixed interventions and testing against an input-only predictor), the neurofeedback analogy, the design, the task search, the code, the runs and the analysis are Claude's (Opus 5.5). See `DEVIATIONS.md`, 2026-10-25. Independent audits: `audit/AUDIT_v1v2.md` and `audit/AUDIT_v3.md`.

## 8. Compute and files

**GPU time:**
- v1 pilot job: 15.5 min.
- v2: 3.6 min.
- v3: pilot 7.5 min (23:59-00:06); main 38 min awake (10:48-11:26); the first main job held the lock 10.4 h, nearly all of it with the laptop suspended.

The diagnoses and the task search ran on the CPU, with at most 4 threads after 22:50.

| File | What |
|---|---|
| `PREDICTIONS.md`, `PREDICTIONS_V2.md`, `PREDICTIONS_V3.md` | Preregistrations |
| `PREDICTIONS*.sha256`, `PREDICTIONS_V3.sha256.amend` | Freeze hashes and the logged amendment |
| `DEVIATIONS.md` | Changes, failures and corrections |
| `TASK_SEARCH.md` | Task search |
| `NOVELTY.md` | Novelty verdict |
| `res/screen.json`, `res_pilot/`, `res_selftest/` | v1 item screen, pilot and self-test |
| `res_v2/` | v2 data and analysis |
| `res_v3/`, `res_pilot_v3/`, `res_v3_smoke/` | v3 data, pilot and smoke pass |
| `diag/` | Diagnosis and search scripts and logs |
| `code/`, `code_v2/`, `code_v3/` | Code for v1, v2 and v3 |
| `audit/` | Audits |

## 9. v4 (stopped before its freeze): Stage 0 (`PREDICTIONS_V4.md` draft, `stage0/`, `stage0b/`)

**What v4 would have been.** After v3, the author approved a v4, and the coordinating session set its decisions on the author's behalf:
- a learnability gate before any main data;
- a budget ladder up to rank 32 and 3,000 steps;
- two confidence bins;
- a transfer secondary.

After Stage 0a, it added that the primary report must be immediate: a question appended right after the answer's digits. The code and a draft preregistration exist in `code_v4/` and `PREDICTIONS_V4.md`. **They were never frozen, and no v4 ladder or main data exist.**

**Stage 0a** (`stage0/STAGE0.md`; run by the coordinating session; exploratory, pilot products). The task is v3's, with two bins at the train median of lp2.
- A probe trained on natural rows was tested on held-out swaps whose bin changes. At v3's turn-2 report token it agrees with the donor on 0.36–0.40 and with the recipient on 0.60–0.64, at every layer.
- At the first answer digit from layer 16, it agrees with the donor on 0.61–0.66.
- With two bins, a probe trained on swapped rows alone looks like signal (0.68–0.74). But a probe on the recipient's unswapped features matches it (0.69–0.77), because a changed two-bin label is always the inverse of the recipient's.

**Stage 0b** (`stage0b/STAGE0B.md`; mine; exploratory).
- **Probe check.** With R = the four digits, the immediate report position (27 tokens after the digits) gives donor agreement 0.35–0.40, and the delayed one (45 tokens after) 0.36–0.41, at every layer from 8 to 27. The coordinator's pre-freeze bar was > 0.60, so the immediate format failed.
- **Tiny training runs** (pilot half A only; 190 training and 124 held-out items):
  - arm I: both formats, rank 8 at lr 1e-4 and 1e-3, up to 300 steps;
  - arm O: immediate format, rank 8 at lr 1e-3, up to 300 steps;
  - arms I and O: immediate format, rank 32 at lr 3e-4, up to 900 steps.

  No run learned. 12 of 16 checkpoints answered almost constantly (the same answer on at least 87% of natural items), and the rest mixed the two answers at about chance. Training losses settled near ln 2, and balanced swap accuracy was 0.46–0.52 (0.47–0.52 for arm I in the immediate format). O reached at most 0.573 on natural items (the majority level), against 0.895 for question-only predictors.
- **Tail-position check** (`stage0b/tailpos.json`). The swapped-in confidence is no longer linearly readable from the first token after the digits. Donor agreement is 0.63–0.66 at the last digit, but 0.33–0.36 one token later, and 0.30–0.42 across the whole immediate report, at layers 16, 20, 24 and 27.

**Decision** (the coordinating session; the author had approved the plan that included it, without being shown its threshold; outcome: stop). The trained-report line stops:
- **If the swapped-in confidence survives at the first 1–3 positions after the digits** (donor agreement > 0.60), a one-token-tag variant is proposed, but not run.
- **Otherwise there is no v4 freeze and no ladder.** Paper 9 is written up from this evidence (`paper/`).

**GPU time:** Stage 0a 0.6 min (the coordinating session's job); Stage 0b 21.7 min; code tests 14.4 min (three smoke runs and a deadline test; lock labels `p9v4-smoke`, `p9v4-test2`, `p9v4-smoke3`).

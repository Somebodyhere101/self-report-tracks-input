# Paper 9 audit: v3 (independent recomputation)

**Who and when.** Independent auditor subagent (Claude, Opus 5.5), 2026-10-22, after the v3 chain had finished (`MAIN_DONE` 11:26:12, `analysis.json` 11:26:24).

**Scope.**
- AUDIT_BRIEF.md sections 3, 4 and 5 for v3.
- The v3 freeze amendment, and the DEVIATIONS.md entries at 23:41, 23:56 and 10:55.
- v1 and v2 were audited in `audit/AUDIT_v1v2.md`. Their sentences in RESULTS.md are reviewed here only where the plain-language summary and section 6 restate them.

**Method.**
- **Code.** My own code is in `audit/`. It uses numpy and the standard library only, and imports no study module. I read `code/analyze.py` and `code_v3/v3_analyze.py` for their definitions only. Everything ran on the CPU with 4 threads; the inputs total about 20 MB.
- **Inputs.**
  - `res_v3/screen.json`, `gen.json`, the nine `eval_*.json` files and the eight `trainlog_*.json` files;
  - `res_pilot_v3/`;
  - `logs/` and `tools/gpu_queue.log`, both read-only;
  - `res_v3/analysis.json`, read for comparison only.
- **Bootstrap.** 10,000 percentile resamples with my own seed (4242; the study used 9).
  - P\* is re-chosen in every resample.
  - The T2 bootstrap is clustered by recipient (530 recipients).
- **Learned predictor.** My own damped-Newton fit of the conditional logit (max |gradient| < 1e-14).
- **G5 power.** My own simulation with 4,000 runs.
- **Picks.** Every pick was re-derived from the stored per-rotation letter log-probabilities, for all 27,370 report rows.

## Verdict

**v3 NOT SUPPORTED: CONFIRMED.**

- G1–G5 all pass, and T2 is evaluable: both families have at least 20 changed rows.
- Both reported reasons hold:
  - Δ_I = −0.358, 95% CI [−0.392, −0.322]. The lower bound is not above 0.
  - I's pooled T2 accuracy is 0.280 [0.254, 0.306]. The lower bound is not above 1/3, and both families are below 1/3 (0.240 and 0.322).
- O does not beat the bar either (Δ_O lower bound −0.397), and I passes T4-kl (upper bound 0.030 ≤ 0.15).
- I compared 117 quantities with `analysis.json` and found 0 mismatches.
  - All point estimates are identical.
  - The 40 CI bounds differ by at most 0.0025, which is bootstrap noise.
- Every number in RESULTS.md section 5 matches, with one exception: the range "0.21-0.30" should be 0.21–0.32.
- The remaining problems are wording, missing caveats and one omitted secondary statistic (items 1–16 below). None of them changes the verdict.

## 1. Freeze, amendment and deviation log

| Check | Result |
|---|---|
| `python code_v3\freeze_v3.py check`, run from the folder root | All 20 files OK ("all match", exit 0). |
| Which hash each file matches (my own hashing, `v3_freeze_audit.py`) | 18 files match their 23:34 freeze hash. `code_v3/v3_run.py` and `code_v3/freeze_v3.py` match only their amended hashes. The amend file has exactly two lines, for these two files. `PREDICTIONS_V3.md` matches its frozen hash (ce2914bd…), and the checker never accepts an amended hash for it. |
| The `v3_run.py` change | Reverting `(v["split"] in ("train", "test") or SMOKE)` to `v["split"] in ("train", "test")` reproduces the frozen hash 05f9ae1f… byte for byte (17,571 bytes, the size recorded in the pre-edit bytecode cache of 23:34:53). So the only change is the smoke-mode items table, as logged at 23:41. In main mode the table holds exactly the 2,259 correct train and test items, with no pilot items, so main-mode behaviour is unchanged. |
| The `freeze_v3.py` change | Three edits reverted together reproduce the frozen hash b79cb7cb… exactly (1,836 bytes): (1) remove the `AMEND` constant; (2) remove the six-line amend reader; (3) restore `ok = os.path.exists(resolve(p)) and sha(p) == h`, with LF line endings. The pre-edit bytecode confirms that the FILES list (the freeze coverage) is unchanged. The edit also converted the file from LF to CRLF line endings, with no functional effect. |
| Timing | Freeze at 23:34:54. `p9v3-pilot-1` was queued at 23:35:05. Both files were edited at 23:36:19. The job started at 23:59:05 and printed "all match" with the amended files. The first v3 data (`res_v3/screen.json`) were written at 00:00:01. No v3 data existed when the files were amended. |
| The 23:56 corrections | Item 5 (the 23:41 label was an estimate, and the job was still queued at 23:54) agrees with `tools/gpu_queue.log`: WAIT at 23:35:05, START at 23:59:05. |

**The overnight-suspension entry (2026-10-22 10:55).**

| Claim | Evidence | Status |
|---|---|---|
| `p9v3-main-1` started at 00:12, and gen began at 00:12:03 | Job log; gpu_queue START at 00:12:03. | confirmed |
| `jvec.pt` was written at 02:43:58 and `gen.json` at 10:38:05 | File times. | confirmed |
| gen was logged at 626.1 min | Job log: "END v3_run.py gen rc 0 626.1 min". | confirmed |
| After wake the job yielded with exit 3, and main-2 started at 10:48 | The job log shows "time budget reached; yielding" at 10:38:07. gpu_queue shows END rc=3 after 37,564 s, then main-2 WAIT at 10:38:28 and START at 10:48:13. Another job (`si-tier3`) held the lock from 10:38:13 to 10:47:59. | confirmed |
| In main-2, gen took 0.3 min because it skipped | "exists …gen.json", then END 0.3 min. | confirmed |
| Training began at 10:48 | `train O 0` started at 10:48:32. | confirmed |
| Lid closed and laptop unplugged; Modern Standby at 00:12, hibernation at 05:24, wake at 10:37 | Relayed from the system log; I cannot check this from the study files. It is consistent with them: the first log line after the gap is at 10:37:28, and gpu_queue logs "KEEP-AWAKE off" at 10:37:58. | consistent |
| The data are unaffected | Within gen, only the train-item capture for the bin vectors and swap norm (`jvec.pt`) finished during the gap. The T1–T4 measurements are logged at 10:37:28–10:38:05, after wake. Every stored `cb_int` equals the bin of its stored `lp2_int`. Every label equals the bin recomputed from `screen.json`, which was written before the gap. G2 and G4 are both 1.00. Determinism itself cannot be checked without a rerun. | plausible; internally consistent |
| The lock change does not affect this chain | `tools/gpu_lock.py` was last modified at 10:52:09, after main-2 began waiting at 10:38:28. | confirmed |
| Logged before results existed | DEVIATIONS.md was last modified at 10:54:01. The first evaluation file was written at 11:06:54. So the entry was written while training ran, before any evaluation data existed. | confirmed |

There are two cosmetic slips in the entry (item 16). No frozen rule was changed after data were seen.

## 2. Recomputed against reported

The "Recomputed" column is my value. "analysis.json" and "RESULTS.md" are the reported values. CIs are 95% bootstrap intervals.

| Quantity | Recomputed | analysis.json | RESULTS.md | Status |
|---|---|---|---|---|
| Items screened / correct train / correct test | 3,339 / 1,111 / 1,148 | n_test 1,148 | 1,148 | match |
| th (quartile cut points of lp2 over the 1,111 correct train items) | −0.018150, −0.0013264, −0.00007486 (p2 0.9820, 0.99867, 0.999925) | identical to `gen.json` th | −0.0181, −0.00133, −0.0000749 (p2 0.982, 0.9987, 0.99993) | match |
| Bins in the train / test items | 278/278/280/275 and 273/312/275/288 | – | – | – |
| Items table: bin, para bin and entropy bin recomputed | 0 mismatches in 2,259 | – | – | match |
| G1: test accuracy / usable test items | 0.8638 (1,148/1,329) / 1,148 | 0.8638 / 1,148 | 0.864 / 1,148 | match; pass |
| G2: digits kept in T1 swaps | 1.000 (757/757) | 1.0 | 1.00 | match; pass |
| G3: probability bin = entropy bin on T3 items | 0.9956 (1,143/1,148) | 0.9956 | 0.996 | match; pass |
| G4: different-bin T1 swaps carry the donor's bin | 1.000 (412/412); same-bin rows also 1.000 (345/345) | 1.0 (n 412) | 1.00 (412) | match; pass |
| G5: P\*, divergent items, power | para (tied with learned); 463 (0.403); power 1.00 | para; 463; 1.0 | 463 of 1,148 (0.403) / 1.00 | match; pass |
| T3 predictors: para / majority / learned / X | 0.5967 / 0.2378 / 0.5967 / 0.2448 | same | 0.597 / 0.238 / 0.597 / 0.245 | match |
| T3 N: accuracy; Δ [CI] | 0.2718; −0.3249 [−0.3676, −0.2831] | 0.2718; −0.3249 [−0.3676, −0.2814] | 0.272; −0.325 [−0.368, −0.281] | match |
| T3 O: accuracy (seeds 0, 1); Δ [CI] | 0.2334, 0.2343; −0.3628 [−0.3972, −0.3288] | same; [−0.3968, −0.3284] | 0.233, 0.234; −0.363 [−0.397, −0.328] | match |
| T3 I: accuracy (seeds 0, 1); Δ [CI] | 0.2491, 0.2291; −0.3576 [−0.3924, −0.3223] | same; [−0.3933, −0.3219] | 0.249, 0.229; −0.358 [−0.393, −0.322] | match |
| T3 J: accuracy (seeds 0, 1); Δ [CI] | 0.2352, 0.2509; −0.3537 [−0.3885, −0.3166] | same; [−0.3885, −0.3184] | 0.235, 0.251; −0.354 [−0.389, −0.318] | match |
| T3 divergent items (463): names truth / names P\*'s pick | N 0.406/0.272; O 0.234/0.162; I 0.279/0.226; J 0.270/0.123 | identical | **not reported** | omitted (item 6) |
| T1 N: different-bin [CI]; same-bin; names own bin | 0.2767 [0.2330, 0.3204]; 0.2174; 0.2524 | same; [0.2330, 0.3180] | 0.277 [0.233, 0.318]; 0.217; 0.252 | match |
| T1 O | 0.2415 [0.2100, 0.2743]; 0.2261; 0.2039 | same | 0.242 [0.210, 0.274]; 0.226; 0.204 | match |
| T1 I | 0.3010 [0.2670, 0.3362]; 0.2290; 0.2257 | same; [0.2670, 0.3350] | 0.301 [0.267, 0.335]; 0.229; 0.226 | match |
| T1 J | 0.2124 [0.1820, 0.2451]; 0.2174; 0.2439 | same; [0.1808, 0.2451] | 0.212 [0.181, 0.245]; 0.217; 0.244 | match |
| T2 changed rows: T2a / T2c | 335 of 477 / 312 of 400 | 335 / 312 | 335 / 312 | match |
| T2 N: T2a / T2c / pooled [cluster CI] | 0.2567 / 0.2147 / 0.2365 [0.2015, 0.2716] | same; [0.2018, 0.2718] | 0.257 / 0.215 / 0.236 [0.202, 0.272] | match |
| T2 O | 0.2418 / 0.2853 / 0.2628 [0.2370, 0.2893] | same; [0.2365, 0.2887] | 0.242 / 0.285 / 0.263 [0.237, 0.289] | match |
| T2 I | 0.2403 / 0.3221 / 0.2798 [0.2535, 0.3059] | same; [0.2543, 0.3063] | 0.240 / 0.322 / 0.280 [0.254, 0.306] | match |
| T2 J | 0.2672 / 0.2981 / 0.2821 [0.2546, 0.3101] | same; [0.2547, 0.3095] | 0.267 / 0.298 / 0.282 [0.255, 0.309] | match |
| T2 rule for I (≥ 20 rows; pooled lower bound > 1/3; every family > 1/3) | both families evaluable; fails | fails | "not above 1/3" | match |
| T4-kl switch increase [CI]: N / O / I / J | −0.0150 [−0.0550, 0.0250] / −0.0138 [−0.0525, 0.0238] / −0.0113 [−0.0525, 0.0300] / 0.0000 [−0.0350, 0.0363] | same means; CIs within 0.0025 | "within ±0.02"; I −0.011 [−0.053, 0.030] | match; I passes |
| T4-norm switch increase: N / O / I / J | 0.0075 / 0.0013 / −0.0238 / −0.0113 | same | "close to zero" | match |
| T4 bin changed by the perturbation: kl / norm | 0.300 / 0.385 | same | – | match |
| T5 multiple-choice accuracy with the adapter in turn 2: N / O / I / J | 0.8958 / 0.8750 / 0.8854 / 0.8854 | same | 0.875–0.885 against 0.896 | match |
| T5 multiplication accuracy with the adapter everywhere: N / O / I / J | 1.000 / 0.995 / 0.9975 / 0.995 | same | "0.995–1.000" | match (wording, item 12) |
| T5 drift flags | none | none | none | match |
| Verdict and reasons | NOT SUPPORTED; (1) I fails T3, (2) I fails T2 | same | same | match |
| Training loss: step 0 / step 25 / mean of steps 50–299 (8 runs) | 5.21–11.01 / 1.27–1.58 / 1.382–1.418 (all runs: 1.399; ln 4 = 1.386) | – | "from 5-11 … to about 1.39 within 25 steps" | match |
| "Test accuracies sit at 0.21-0.30" | Across arms on T1–T3: 0.212–0.322 (I on T2c: 0.322) | – | 0.21–0.30 | **MISMATCH** (minor; item 3) |
| "The untrained model always answers 'Very sure'" | N picks bin 1 on all 3,582 report rows (T1–T4) | – | yes | match |

**Definitional checks.**
- **Learned predictor.** Its predictions equal the para bin on all 1,148 T3 items in all four fits I tried:
  - fit on the 991 O-list items or on all 1,111 train items;
  - with the code's eight features, or with only the five the prereg text lists (without the bin intercepts).
- **Majority bin.** It is bin 0 over the O-list items, as the study used, but bin 2 over all train items. Either gives about 0.24 on T3.

## 3. Rotation scoring spot check (`eval_I_s0.json`, T3)

Rotation r shows option (j + r) mod 4 at letter j, so the score of option k is the mean over r of lp[r][(k − r) mod 4].

| Row | Item | Truth bin | Recomputed scores (bins 0–3) | Equal to stored | Pick: recomputed / stored | Tied top |
|---|---|---|---|---|---|---|
| 0 | 11\*92 | 0 | −1.447606, −1.416356, −1.385106, −1.416356 | yes (bit-identical) | 2 / 2 | no |
| 287 | 27\*66 | 2 | −1.410887, −1.410887, −1.410887, −1.442137 | yes | 0 / 0 | bins 0, 1, 2 |
| 574 | 39\*67 | 1 | −1.403457, −1.434707, −1.403457, −1.403457 | yes | 0 / 0 | bins 0, 2, 3 |
| 861 | 55\*63 | 1 | −1.419594, −1.419594, −1.357094, −1.450844 | yes | 2 / 2 | no |
| 1147 | 99\*99 | 0 | −1.430207, −1.430207, −1.367707, −1.430207 | yes | 2 / 2 | no |

- **All rows.** Over all 27,370 report rows in the nine evaluation files, there are 0 score mismatches and 0 pick mismatches. The opposite rotation convention gives 5,770 pick mismatches on T3 rows, so the check is not vacuous.
- **Row order.** The evaluation rows follow the row order of `gen.json`:
  - N's report log-probs on T4, T2c and T1 rows are much closer to the T3 report for the same recipient than to a shuffled one. For example, on T4 the mean absolute difference is 0.106 against 0.749 when shuffled.
  - The T5u keys equal the first 400 T3 items.
- **Ties.** Two of these five rows have a tied top score, and the frozen argmax picks the lowest index (bin 0). This is common for the trained arms. See "Other observations" and item 4.

## 4. Leakage and design

- **Splits.** The train, test and pilot product sets are disjoint, and every product falls in a single split.
- **Training lists.** O, X, I and J have 2,400 rows each.
  - They use 1,107 distinct keys, all correct train-split items.
  - No test product appears among training items or donors.
  - None of these keys is among the 1,148 test keys used in T1–T5.
- **Donors and constraints.** All hold:
  - **Donor pairs.** I, T1 and T2a donors share the recipient's product and are a different item.
  - **I list.** On I's different-bin rows, the donor's bin differs from both the recipient's bin and its paraphrase bin. On same-bin rows, the bins are equal. The rows alternate exactly.
  - **J list.** On J's different rows, c is neither the recipient's bin nor its paraphrase bin. On same rows, c is the recipient's bin. J alternates exactly.
  - **T1–T4 rows.** T1 rows satisfy the same constraints as I's, swap layers 16–27, and give each recipient at most one row of each kind. T2a uses layer 22 only. T2c has c outside {bin, paraphrase bin} at scale 2. T4 uses scale 1.332 (kl) and 2 (norm).
  - **X list.** X's list equals O's.
- **Cut points and bin vectors come from train items only.**
  - th and the entropy cut points equal the frozen quartile rule on the 1,111 correct train items. Cut points computed from test or pilot items differ.
  - In `gen()` (`code_v3/v3_run.py` lines 201–223), th, the entropy cut points, the bin vectors (jvec) and the swap norm (base_norm) are all computed from `pools(scr, "train")`. `V3_SMOKE` is unset in main mode.
  - The pilot used pilot products only.
- **Option sets.** Every report, in training and in evaluation, shows the fixed four-bin list in one of four cyclic rotations. That covers the in-place self-report (`v3lib.add_report`) and X's external prompt (`v3lib.external_prompt`, used by `trainlib_v3.train` and `eval_external`).
  - All 27,370 stored report rows have 4 rotations × 4 normalised letter log-probs.
  - Every label is a bin index from 0 to 3, so every option set contains its label.
- **Deviations.** Every change is logged, and none changed a frozen rule after data were seen (section 1).

## 5. Training logs (`res_v3/trainlog_*.json`; one 8-example batch loss every 25 steps)

| Run | Step 0 | Step 25 | Mean, steps 50–299 | Range, steps 25–299 |
|---|---|---|---|---|
| O seed 0 | 11.01 | 1.55 | 1.400 | 1.34–1.55 |
| O seed 1 | 5.21 | 1.42 | 1.418 | 1.36–1.54 |
| I seed 0 | 9.43 | 1.27 | 1.404 | 1.27–1.64 |
| I seed 1 | 6.82 | 1.36 | 1.394 | 1.36–1.44 |
| J seed 0 | 8.89 | 1.58 | 1.394 | 1.31–1.58 |
| J seed 1 | 6.29 | 1.50 | 1.386 | 1.32–1.50 |
| X seed 0 | 9.09 | 1.38 | 1.414 | 1.34–1.48 |
| X seed 1 | 10.74 | 1.31 | 1.382 | 1.31–1.42 |

- **The ln 4 claim is confirmed.** Every run settles near ln 4 = 1.386; the mean over all runs (steps 50–299) is 1.399.
- **The label-blind floor.** A reporter that knows only the label frequencies of its training list would score 1.3853 (O and X), 1.3845 (I) or 1.3846 (J). The losses sit at that floor, so no arm learned anything about individual labels. The trained reporters' outputs agree with this: their median top-letter probability is 0.26–0.31, against N's 0.95.
- **Precision of the wording.** Each logged value is a single 8-example batch, so "about 1.39 within 25 steps" is loose: step-25 values run from 1.27 to 1.58. The claim is still fair.

## 6. Errors and overclaims (most important first; none changes the verdict)

1. **A missing caveat: v3's intermediate is a confidence (metacognitive) quantity, not a bridge-like value.**
   - TASK_SEARCH.md states this caveat. RESULTS.md never does: not in the plain-language summary, not in section 5, not in section 6.
   - The summary presents v3 as the replacement test of "reporting what actually happened inside it". Section 6 says v3 "met the first need" (an internal step).
   - What v3 actually tests is whether a model can be trained to report its own output confidence.
   - *Fix* (summary, after "…without changing the question or the answer"): "Unlike the planned bridge country, this is the model's confidence in its own answer (a metacognitive quantity), not an intermediate value of the task, so v3 tests trained confidence reporting." Add the same qualifier to section 6, bullet 3.

2. **The plain-language summary (paragraph 2) restates v1/v2 more strongly than the data allow.**
   - "Checks frozen in advance showed that … the needed hidden step … sits in the same place as the city name itself" is wrong on two counts:
     - The frozen v1 checks showed only that the readout of R failed.
     - The localisation came from the exploratory diagnosis and then from v2. v2 measured which country later questions name from copied states, not where the two-hop computation holds the bridge (AUDIT_v1v2 items 1, 3 and 4).
   - "Our measuring tool" is the placeholder readout, and the summary does not say so.
   - Section 6, bullet 2 ("the bridge country is read … So there was no reconstruction-free route to train") has the same problem.
   - *Fix:* "Its frozen bed checks failed: a placeholder readout of the states after the city name could not recover the country. An exploratory diagnosis, then a preregistered follow-up on 306 held-out cities, found that the country later questions name (both the model's self-report and the placeholder readout) is taken from the city-name states. So with these instruments 'reading the computation' and 'reading the question' could not be told apart." In section 6, write "the country that the self-report and the placeholder readout name", and end "so this bed offered no route these instruments could separate from reconstruction".

3. **"No version learned the task at all", "training did not take for any arm" and "Test accuracies sit at 0.21-0.30" need qualifying.**
   - The range is wrong: I scores 0.322 on T2c, so it should read 0.21–0.32.
   - My exploratory analysis (not preregistered) adds a small qualification. Arm I is above the level expected from its own pick frequencies (label-permutation test, 5,000 permutations) only on intervened rows whose bin moved away from the input-implied bins:
     - T1 different-bin rows: 0.301 against 0.253, p = 0.0006;
     - T2c changed rows: 0.322 against 0.270, p = 0.0012.
   - Both seeds point the same way, and both p-values survive a Bonferroni correction over the 20 arm × row-set tests.
   - I is not above that level on same-bin swaps (0.229 against 0.245) or on untouched items (0.239 against 0.243).
   - The effect is small and below the 1/3 level a pure tamper detector could reach. It indicates weak sensitivity to the intervention, not reading of natural confidence.
   - On T3, every arm sits at its chance level (0.229–0.272).
   - *Fix* (summary): "No version learned to report the confidence of untouched problems." In section 5, give the corrected range and one sentence on I's small effect on intervened rows, labelled exploratory.

4. **"The trained arms spread their picks without tracking the label" misdescribes what the picks are.**
   - The trained reporters' letter distributions are nearly uniform. Their letter log-prob differences fall on the 0.125 grid of bf16 logits.
   - Their rotation-averaged scores are exactly tied at the top on many T3 rows: 49–63% for the O, I and J runs, and 31–63% for X (29% for X's seed-averaged scores).
   - The frozen argmax resolves a tie to the lowest index, "Completely sure". As a result, the share of picks naming bin 0 is 0.29–0.74 for O, I and J, and 0.96 for X seed 0.
   - Breaking ties at random instead changes T3 accuracies by at most 0.013, so nothing else is affected.
   - *Fix:* "The trained reporters' letter distributions are nearly uniform (median top-letter probability 0.26–0.31). Their scores are often exactly tied, and ties default to 'Completely sure', so their pick frequencies mostly reflect tie-breaking."

5. **"I passed the tamper control" is true but uninformative.**
   - I's reports barely depend on the turn-1 state, so a random direction cannot raise its error rate. A switch increase near 0 is guaranteed.
   - *Fix:* "I passed T4 trivially, since its reports do not track the state."

6. **A preregistered secondary statistic is omitted.** RESULTS omits the divergent-item breakdown, although PREDICTIONS.md section 7 lists it and v3 adopted it.
   - It covers the 463 items where P\* errs: how often each arm names the truth, against how often it names P\*'s pick.
   - The values are N 0.406/0.272, O 0.234/0.162, I 0.279/0.226 and J 0.270/0.123.
   - *Fix:* Add a line. Note that N's 0.406 comes from its constant "Very sure", which matches the truth on many divergent items.

7. **The "chance 0.25" captions hide uneven base rates.**
   - The test bins are 273/312/275/288, so the best constant answer ("Very sure", N's only pick) scores 0.272 on T3.
   - On T1 different-bin rows the donor bins are 89/114/122/87, so the best constant answer ("Fairly sure") scores 0.296. I's 0.301 is close to that.
   - *Fix:* Give these base rates next to "chance 0.25".

8. **J is not labelled descriptive.**
   - Section 5 gives J's Δ and CI in the same table as N, O and I.
   - Under PREDICTIONS.md section 8, which v3 adopts, J's results and the X-versus-O comparison are descriptive. X's T3 accuracy, by contrast, is part of the input-only bar.
   - *Fix:* Mark J's row "descriptive; not part of the verdict".

9. **"Passed the same checks" overstates the bed validation.**
   - In v3, G3 and PG2 compare the probability bin and the entropy bin of the same output distribution at the same position. Both are monotone in the off-top mass when one digit dominates.
   - So the agreement (0.996 and 0.983) is close to guaranteed, and these checks do not validate the ground truth the way v1's readout calibration was meant to.
   - *Fix:* In the summary, write "passed v1's checks as adapted to this task". In section 5, note that G3 is a weak check.

10. **The paraphrase caveat and the input-only bar.**
    - "Mostly determined by the input" is supported but not quantified. On T3 items:
      - the rank correlation between lp2 and the paraphrase lp2 is 0.91;
      - the paraphrase bin is within one bin of the truth on 96.5% of items;
      - it runs on average 0.38 bins less sure than the truth.
    - An exploratory input-only predictor fit on train items only does much better than the frozen one: binning the paraphrase lp2 at its own train quartiles gives 0.718, against 0.597.
    - So the frozen bar was not the strongest input-only predictor, and G5's 463 "divergent" items partly reflect a calibration offset between prompt formats. This makes the NOT SUPPORTED verdict safer, not weaker.
    - "It is the best of the four" also hides a tie: the learned predictor made the same prediction on every item.
    - *Fix:* Quantify the caveat, write "tied with the learned predictor, which made identical predictions", and in section 6 add a recalibrated paraphrase predictor to what a fair test needs.

11. **The summary's learnability sentence reads as a finding about difficulty.** "So the negative result says more about how hard this signal is to report at this budget…" interprets a confound.
    - *Fix:* "So the comparison between label sources could not be made: none of them taught the report at this budget and in this format." Section 5 already words the confound correctly.

12. **T5 wording.**
    - "Multiplication accuracy with the adapter on everywhere was 0.995-1.000": the 1.000 is N, which has no adapter.
    - The 400 items are test items the base model already answers correctly, so this is a retention measure.
    - *Fix:* "Trained arms 0.995–0.998 (N 1.000), on 400 test items the base model answers correctly."

13. **T2 wording.** "Rows where the intervention changed the bin" understates the rule. A changed row is one where the intervened bin differs from both the natural bin and the paraphrase bin. *Fix:* Say that.

14. **Section 5 caption.** "(`res_pilot_v3/`; pilot products only)" covers PG1, which is computed over all items and counts test items. *Fix:* "PG2–PG3 on pilot products; PG1 on the full item screen."

15. **The task-search record is incomplete.**
    - TASK_SEARCH.md, PREDICTIONS_V3.md section 1 and RESULTS section 4 say the screen logs are in `diag/`.
    - Only candidates 1 and 2 have log files (`diag_arith.log`, `screen_sumcmp.log`). Candidates 3–7 printed to the console only, including the passing candidate 7 (`screen_mult2.py`). So their figures (for example 41/48, 47/48, 10/10, 27/32 and 24/24) cannot be checked from files.
    - *Fix:* Say that candidates 3–7 have scripts but no saved logs, or rerun those CPU screens with logging and label them as reruns.

16. **Cosmetic slips in the DEVIATIONS.md 10:55 entry.**
    - The heading says 10:55, but the file was last written at 10:54:01. This is the same estimate pattern as the 23:41 label.
    - "Pilot (00:02–00:06)": the pilot job ran 23:59:05–00:06:32. The pilot stage itself ran 00:00:03–00:02:47, and the smoke pass ran until 00:06:32. RESULTS gives 23:59–00:06.
    - *Fix:* Append a correction note. Nothing material.

**Checked and found adequate.**
- The summary limits itself to one model, two tasks and one budget.
- The learnability confound is stated in sections 5 and 6.
- Neither RESULTS nor the v3 documents quote the author's chat messages. Section 4 paraphrases the relayed request ("the author wanted everything done tonight") without quoting it.
- There is no general claim about "introspection".
- Attribution is correct.
- The compute times in section 8 match `tools/gpu_queue.log`: 448 s, 2,279 s and 37,564 s.

## Other observations (no change to the verdict)

- **bf16 quantisation.** The letter logits and the lp2 ground truth are both computed from bf16 `lm_head` outputs, which are cast to float32 only afterwards. For the near-uniform trained reporters, every preference is one bf16 step (0.125 in logit), so many exact ties follow (item 4). The frozen prereg does not specify how ties are broken; the frozen code takes the first maximum.
- **Answer retention under T2.** All four argmax digits were kept on 0.962 of T2a rows and 0.875 of T2c rows (T4: 0.985 kl, 0.970 norm). As frozen, the T2 analysis does not condition on retention; the written answer is teacher-forced in every case.
- **T4 matching in the main data.** The median R-KL is 0.097 for T4-kl and 0.070 for the T1 swaps (0.088 on different-bin rows). So the KL matching set in the pilot roughly holds in the main data.
- **Pilot learning-rate rule.** v1's rule scores acc_diff + acc_same, but v3's frozen pilot code scores acc_diff only, on 80 rows. The v3 text says "v1's rules". Both were frozen before any data, and the choice (1e-4; 0.3375 against 0.275) is reported only for information.
- **Pilot parameters.** All of them follow from `pilot.json` under the frozen rules: l1 16; ls 22 (transfer 0.583); la 18 and s_add 2 (0.683 with retention 0.933); s_kl 1.3316 (log-log interpolation of 0.1194 between 0.0537 and 0.3716); lr 1e-4.

## Files

- `audit/recompute_v3.py`: gates, bins, the T1–T5 statistics, the verdict, and the comparison with `analysis.json`. It writes `audit/recomputed_v3.json`.
- `audit/v3_checks.py`: the five-row spot check, the leakage and design checks, the row-order check, independence and permutation tests, tie and spread statistics, training-label entropies, and the exploratory recalibrated paraphrase predictor. It writes `audit/v3_checks.json`.
- `audit/v3_arm_I_signal.py`: per-seed permutation tests behind item 3. It writes `audit/v3_arm_I_signal.json`.
- `audit/v3_freeze_audit.py`, `audit/v3_freeze_reconstruct.py` and `audit/v3_pyc_freeze_dump.py`: per-file hash status, and the exact reconstruction of both frozen files from their pre-edit bytecode caches.
- `audit/v3_kl_check.py`, `audit/v3_ci_diff.py` and `audit/v3_explore.py`: the KL medians, the CI-difference summary and a structure dump.

# Paper 9 deviations from PREDICTIONS.md

Each entry gives the time (+04:00), what changed and why, and whether any analysis data had been seen. `PREDICTIONS.md` and the hashed files are never edited after the freeze. A changed code file gets a new hash line here.

## 2026-10-21 17:28 (+04:00): freeze (not a deviation)

- **Freeze.** `PREDICTIONS.sha256` was written at 17:28 (PREDICTIONS.md sha256 4057de36…). It covers 30 files: the documents, all code, and the six dependency files from paper 8 and S2.
- **State at freeze.** No Qwen3-1.7B data of any kind existed. Only the CPU dry-run outputs on SmolLM2-135M existed (`res_tiny/`, `res_pilot_tiny/`, `res_selftest_tiny/`).
- **Pre-freeze review.** An independent code review found no blockers. Its two should-fix and five minor findings were fixed before the freeze:
  - the learned-predictor fit;
  - hash coverage and import-path order;
  - G4 restricted to different-country rows;
  - the donor and injected country must differ from the gazetteer country, and option sets include it;
  - training-list length assertions;
  - the drift flag includes 1-hop;
  - the self-test uses pilot-split cities.
- **Queue entry.** The chained GPU job (`code/run_chunks.sh`) was put in the GPU queue at 17:04, before the freeze, to hold its place. `code/job.py` refuses to run unless `freeze.py check` passes, and `tools/gpu_hold.txt` holds every GPU job until 22:00. `run_chunks.sh` was not modified after it started.

## 2026-10-21 22:17 (+04:00): v1 bed check FAILED; the chain stopped before any analysis data

**What ran.** The chained job ran from 22:02 to 22:17:
- the engine self-test on Qwen3-1.7B (all checks pass);
- the item screen;
- the pilot;
- the end-to-end self-test on pilot cities.

`make_config.py` then applied the frozen bed checks.

**Results.**
- **PG1 passed:** 2-hop accuracy 0.954 over 1,008 cities, with 363 usable test items.
- **PG2 failed:** on 164 known-like pilot cities, calibration of the placeholder readout of R was 0.116 (required ≥ 0.80), and reliability was 0.309 (required ≥ 0.70).
- **PG3 failed:** swap transfer at l1 = 0 was 0.754 (required ≥ 0.90). Transfer was the same, 0.73–0.78, for every swap condition, from-0 to from-20 and single layers 6 to 22. So the readout was not reading R at all.
- **For information:** the untrained report named the recipient's country on 86% of swaps and the donor's on 0%.

As frozen, the chain stopped before `stage_gen.py`. No train/test readouts, no reporter training and no evaluation exist. `res/` holds only `screen.json`.

**Diagnosis afterwards (exploratory; pilot-split cities only; CPU; `diag/diag_readout.py`, log `diag/diag_readout.log`).**
- **Letter format.** Letters carry essentially all of the next-token mass at the answer position (≈ 1.0), so the format is not the problem. The model is near-deterministic in its letter choice, and the 21–25-nat first-step training losses come from that.
- **Readout calibration on 12 known pilot cities:**

| Readout variant | Letters | Names |
|---|---|---|
| City visible | 12/12 | 11/12 |
| Placeholder, nothing copied (prior) | 3/12 | 4/12 |
| Placeholder plus R copied at every layer (the v1 readout) | 3/12 | 4/12 |
| Placeholder plus R and the city tokens copied at every layer | 12/12 | 11/12 |

**Conclusion.** In Qwen3-1.7B the computation after the city name carries no country that the model's own downstream question-answering can read. The bridge is held at the city tokens. So the v1 intervention (swapping R) did not change the bridge, and the v1 ground truth cannot work.

**Next step.** Under PREDICTIONS.md section 10, a new design gets its own preregistration (v2). v1 stays frozen as the record of this failed bed.

## 2026-10-21 22:40 (+04:00): v2 preregistration frozen (a new design, as v1 section 10 requires)

- **What v2 is.** `PREDICTIONS_V2.md` (sha256 544f4758…) and `PREDICTIONS_V2.sha256` cover the v2 code (`code_v2/`), the v1 engine files it imports, and the paper 8 and S2 dependencies. v2 tests the diagnosis on held-out test cities. It trains nothing.
- **Refused early start.** At 22:33 a v2 queue entry was registered before the freeze. It got the GPU at 22:38, and `job_v2.py` refused to run because v2 was not yet frozen, as designed. No data were produced.
- **Re-queued** at 22:40, after the freeze.
- **Data seen before the v2 freeze:** the v1 pilot and the diagnoses on 12 pilot cities only. No test city had been read out, swapped or reported on.

## 2026-10-21 23:31 (+04:00): v2 result

`res_v2/analysis_v2.json`, run under the frozen v2 rule: **SAME WINDOW**. All of H1, H2 and H3 hold. Details are in RESULTS.md.

## 2026-10-21 23:35 (+04:00): task search and the v3 freeze (instruction from the coordinating session)

- **Instruction.** At about 22:50 the coordinating session relayed the author's wish that everything be done tonight. It asked for a replacement task with an intermediate computed after the input, screened with v1's gates and preregistered as a frozen addendum. The arms, tests and decision rule were to stay the same, adapted to the task. The deadline for a candidate to pass was 02:00. Jobs were limited to 4 CPU threads and 6 GB of RAM.
- **Search.** Seven candidates were screened on CPU with ad-hoc items (`TASK_SEARCH.md`, `diag/`). Candidate 7 passed at 23:26: confidence in the second digit of a two-digit product.
- **Freeze.** v3 was frozen at 23:34: `PREDICTIONS_V3.md` (sha256 ce2914bd…) and `PREDICTIONS_V3.sha256`, which covers the v3 code and the v1 files it imports.
- **Data at freeze.** Before the freeze, only the CPU screens and a CPU check of the v3 library on four items existed. No v3 item screen, pilot or main data existed.
- **Adaptations from v1.** All are listed in PREDICTIONS_V3.md section 3. The main ones:
  - the ground truth is the run's own output probability, so there is no readout;
  - G3 becomes agreement between the probability and entropy labels;
  - the input-only predictors are adapted (paraphrase, majority, learned, X);
  - T5 ungated measures multiplication accuracy;
  - the self-test becomes a smoke pass in `res_v3_smoke/`.
- **Queued** at 23:35 (`code_v3/run_v3.sh`).

## 2026-10-21 23:41 (+04:00): v3 code fix before any v3 data (smoke-mode item table)

- **Bug.** In smoke mode `gen` uses pilot items for both train and test. But its item table (`G["items"]`) kept only the train and test splits, so the smoke analysis would have failed with a missing key. That failure would have stopped the chain at bed check PG5.
- **Fix.** The table now includes pilot items in smoke mode only. Main-mode behaviour is unchanged.
- **Freeze check.** `freeze_v3.py` gained v1's amendment mechanism: a changed file's new hash is accepted only if it is listed in `PREDICTIONS_V3.sha256.amend`, and `PREDICTIONS_V3.md` itself can never be amended. The amended files are `code_v3/v3_run.py` and `code_v3/freeze_v3.py`.
- **Data seen:** none. The v3 job was still waiting in the GPU queue.

## 2026-10-21 23:56 (+04:00): corrections to earlier entries, from the independent v1/v2 audit (`audit/AUDIT_v1v2.md`)

Earlier entries are not edited. These notes correct them.

1. **22:17 entry, reliability figure.** The PG2 reliability of 0.309 is over all 194 usable pilot cities (60/194). On the 164 known-like cities it is 0.299 (49/164). PREDICTIONS.md section 10 says "on pilot known-like items", but the frozen `stage_pilot.py` computes reliability over all usable pilot items. This text/code discrepancy predates any data, and PG2 fails under either definition.
2. **22:17 entry, "No train/test readouts, no reporter training and no evaluation exist".** This should read: no reporter was trained or evaluated on test cities. Reporters were trained and evaluated on pilot cities only, in the learning-rate check and the self-test.
3. **22:17 entry, its "Conclusion".** It rests on an exploratory 12-city diagnosis. "The bridge" itself was not measured, only which country a placeholder readout and a report name. v2 later tested the pattern, and the confirmed claim is v2's frozen verdict wording, not the 22:17 conclusion.
4. **Calibration definitions.** v1's 0.116 counts unreliable known-like items as misses (frozen code). On v2's definition (pick = gazetteer, no reliability filter), the same pilot data give 0.354 (58/164).
5. **Entry timestamps.** The entry labelled "23:41" was written at about 23:36–23:40; I estimated the time when writing it. The amend-file label "2026-10-21T23:41" carries the same estimate. Both were written before the v3 job started; the job was still waiting in the queue at 23:54.

## 2026-10-22 10:55 (+04:00): v3 pilot passed its bed checks; the main run was suspended overnight (no design change)

**Pilot (00:02–00:06).** All frozen bed checks passed (`res_pilot_v3/pilot_gates.json`):
- **PG1:** accuracy 0.853 over all items, 1,148 usable test items.
- **PG2:** probability and entropy bins agree on 0.983.
- **PG3:** transfer 1.00 at l1 = 16, digits kept 1.00.
- **PG5:** the smoke pass completed.
- **PG4 (information only):** arm I's pilot learning-rate check reached different-bin accuracy 0.34 at lr 1e-4 and 0.28 at 3e-4. The frozen rule chose 1e-4.

`config_v3.json`: l1 16, ls 22, la 18, s_add 2, s_kl 1.332, lr 1e-4.

**Main job `p9v3-main-1`** started at 00:12, and its first step (`gen`) began at 00:12:03.

**Suspension.** As relayed by the coordinating session from the system log, the laptop's lid was closed and the laptop was unplugged at that moment. It went into Modern Standby at 00:12, hibernated at 05:24 and woke at 10:37. File times show the process made only intermittent progress during standby: `jvec.pt` was written at 02:43:58 and `gen.json` at 10:38:05, after wake. So the logged 626.1 minutes for `gen` is suspension time, not compute time.

**After wake.** The job then reached its 45-minute yield (exit 3), and `p9v3-main-2` started at 10:48. In it, `gen` took 0.3 min only because the stage found `gen.json` already present and skipped. Training began at 10:48.

**Data.** Unaffected. The computation is deterministic, `gen` finished with rc 0, and suspension does not change numerics.

**Lock change.** `tools/gpu_lock.py` has since gained a hard time limit (60 awake minutes by default). `p9v3-main-2` was launched under the old lock, so this chain is unaffected.

## 2026-10-22 11:56 (+04:00): v3 audit corrections (`audit/AUDIT_v3.md`, verdict CONFIRMED)

- **10:55 entry.** The file was last written at 10:54:01, so the "10:55" label is an estimate, as before. "Pilot (00:02–00:06)" should read: the pilot job ran 23:59:05–00:06:32, the pilot stage 00:00:03–00:02:47, and the smoke pass until 00:06:32.
- **Task-search logs.** Candidates 3–7 printed to the console only. Their logs in `diag/` are deterministic CPU reruns made on 2026-10-22, each labelled as a rerun at the top of its log.
- **RESULTS.md.** All 16 audit items were applied (wording, caveats, base rates, the divergent-item breakdown and the exploratory arm-I signal). No number changed.

## 2026-10-22 12:33 (+04:00): v4 Stage 0 and the coordinator's report-format decision (before the v4 freeze)

- **Stage 0 requested.** The coordinating session, acting for the author, asked for a fast feasibility check before any long v4 training. It was recorded before freezing `PREDICTIONS_V4.md` and allowed to shape it. It has two parts:
  - Stage 0a: is the swapped-in confidence linearly readable where the reporter answers?
  - Stage 0b: does a reporter learn it in short runs?
- **Stage 0a was run by the coordinating session** (`stage0/STAGE0.md`; pilot products only, exploratory). In the untrained model, the swapped-in two-bin confidence is not linearly readable at the turn-2 report token:
  - a probe trained on natural rows agrees with the recipient on 0.60–0.64 of swapped rows, and with the donor on 0.36–0.40;
  - at the answer positions from layer 16 on, it follows the donor (0.61–0.66);
  - the two-bin "swapped-only 0.71" is an artifact of the label always inverting the recipient's bin.
- **Coordinator decision (attribution: the coordinating session, for the author).**
  - The primary v4 report is IMMEDIATE: a confidence question appended to the answer right after its four digits, with the adapter active only on those appended positions.
  - The turn-2 DELAYED report is kept as a secondary arm, for an immediate-against-delayed contrast.
  - Before the freeze, the Stage 0a probe method must show the swapped-in bin readable at the new report position (natural-trained probe agreeing with the donor > 0.60 with two bins).
  - Optionally, the report token of a trained arm-I adapter is probed.
  - Everything else is as in the earlier v4 decisions.
- **My implementation.** The decision describes the tag as inside region R, so that the report position lies in the swapped layers. Read literally, that cannot work: if the report positions were swapped from layer 16 on, a swapped run's report would be the donor's own report, I's labels would become natural labels on donor items, and T1 would stop testing reading. So:
  - turn 1 now ends with the four digits, and **R = the four digit positions** (interventions act on them only);
  - the immediate tail (`v4lib.tail_imm`) continues the answer right after the digits;
  - the delayed tail (`v4lib.tail_del`) closes the answer with `<|im_end|>\n` and asks v4's turn-2 question;
  - both read the same intervened turn-1 run, and the adapter is active on the whole tail;
  - lp2 and every turn-1 label are unchanged: they depend only on positions up to the first digit.
- **Stage 0b (mine; exploratory, pilot products only).**
  - **Probe check:** Stage 0a's extraction repeated for both tails on the same 590 pilot items and 474 pairs, analysed with Stage 0a's own probe script, unchanged (`stage0b/probe_imm/`, `stage0b/probe_del/`).
  - **Tiny training runs:** arm I in both formats at lr 1e-4 and 1e-3, rank 8, up to 300 steps; arm O, immediate; one longer rank-32 run of each of I and O, immediate (`stage0b/s0b_train.py`).
  - The runs use **pilot half A only**, split again into 60% training and 40% held-out products, so that **half B stays untouched for v4's frozen learnability gate**.
  - The cached turn-1 trainer the coordinator suggested was not built. v4's trainer runs a 300-step run in about 2 minutes, and its checkpoint and resume path was verified bit-exact today. A second, unverified training path before the freeze was not worth the speed-up.

## 2026-10-22 13:17 (+04:00): Stage 0b results; v4 ladder not started (`stage0b/STAGE0B.md`)

- **Probe check** (Stage 0a's method and script; 256 swapped rows whose two-bin label differs). A probe trained on natural rows agrees with the donor:
  - **0.35–0.40** at the immediate report position;
  - **0.36–0.41** at the delayed one, at every layer from 8 to 27;
  - **0.61–0.66** at the first digit, layers 16–27.

  The coordinator's condition for the immediate format (donor > 0.60) is **not met**.
- **Tiny runs.** Pilot half A only, split 190/124; half B untouched. No reporter learned:
  - arm I, both formats, rank 8 at lr 1e-4 and 1e-3, up to 300 steps;
  - arm O, immediate, rank 8 at lr 1e-3, up to 300 steps;
  - arms I and O, immediate, rank 32 at lr 3e-4, up to 900 steps.

  Balanced swap accuracy stayed at 0.47–0.52 (chance 0.5), answers were constant, and losses stayed at ln 2. O's natural accuracy was at most 0.573, the majority level, against 0.895 for the question-only predictors.
- **Consequence, under the coordinator's Stage 0 rule.** The v4 budget ladder is not started, and v4 is not frozen. One further exploratory check is queued: where along the immediate tail the information fades (`stage0b/s0b_tailpos.py`). The coordinator decides what follows.
- **GPU used by Stage 0b:** 21.7 min (lock label `p9v4-s0b`).

## 2026-10-22 13:24 (+04:00): RESULTS.md correction (v2 Part B table)

- While collecting numbers for the paper draft, I found that RESULTS.md section 3's Part B table did not match `res_v2/analysis_v2.json`. When the v1/v2 audit's three-decimal fix (item 13) was applied, two-decimal values had been padded with a zero, instead of the file's values being used.
- The wrong cells were:
  - **untrained self-report:** 0.940, 0.910 and 0.010 (×3), against 0.935, 0.905 and 0.015 in the file, at B = 8, 10 and 14–20;
  - **readout:** 0.960 and 0.950, against 0.965 and 0.955, at B = 4 and 6.
- The table now prints the file's values, and the agreement is given as 0.9645.
- No conclusion changes: the transition at B = 12 and SAME WINDOW are unaffected.

## 2026-10-22 13:46 (+04:00): tail-position check; the trained-report line stops; paper 9 drafted

- **Tail-position check** (`stage0b/s0b_tailpos.py`; exploratory). Donor agreement at the first token after the digits is 0.33–0.35, and at most 0.39 over the first three positions. The coordinator's rule (approved by the author) needed > 0.60 there to propose a one-token-tag variant.
- **Decision.** The rule's outcome is to stop the trained-report line: v4 is not frozen, and no ladder runs.
- **Paper 9.** Drafted in `paper/` from this evidence, as the coordinator asked. `paper/make_numbers.py` generates every number and figure from the result files. Next comes an independent number audit; nothing is published.

## 2026-10-22 14:45 (+04:00): manuscript audit (`audit/AUDIT_paper.md`); corrections to records; concurrent-work re-check

- **Manuscript audit.** 338 numbers were checked: 336 match, 1 is a fair rounding, and 1 was wrong (a Table 2 cell, 0.32 for 0.31). The audit's 4 major and 27 further issues concern framing, scope and wording; none changes a preregistered verdict. The manuscript was revised for all of them: title, the two routes left open by the swap depth, the unexplained failure of arm O, probe coverage, v1's metrics, Stage 0b run settings, captions, the Bonferroni count, κ, and ten added verified references.
- **Corrections to earlier entries and records** (earlier entries are not edited):
  - The 13:46 entry, RESULTS.md section 9 and STAGE0B.md section 3 gave the first-token donor agreement as 0.33–0.35. The raw maximum is 0.3555, so the range is **0.33–0.36**. RESULTS.md and STAGE0B.md are fixed. The error came from rounding the 3-decimal `numbers.json` value 0.355 a second time.
  - The 13:17 entry gave balanced swap accuracy as 0.47–0.52. Over all Stage 0b runs it is **0.46–0.52** (minimum 0.462, arm O at 150 steps). For arm I in the immediate format it is 0.47–0.52.
  - RESULTS.md's sentence that both seeds "point the same way" in the exploratory arm-I signal now gives the per-seed p-values and the 56-test Bonferroni count.
- **Timing of the second Stage 0 rule.** The coordinator's rule was: propose a one-token tag only if donor agreement exceeds 0.60 at one of the first three positions after the digits; otherwise stop. It arrived as a message before the tail-position job ran (gpu_queue.log: 13:41:05–13:41:35). The earliest file containing it is `%TEMP%\p9\results_v4.py`, saved at 13:34:23, which wrote RESULTS.md section 9. It was not logged here until 13:46. That entry should have been made on receipt.
- **Concurrent-work re-check** (NOVELTY.md asked for one before any write-up):
  - OpenReview 5P7dVpGNPO: an exact-title web search found no arXiv version.
  - The SPAR Fall 2026 project: its page, "Introspection Training for Verbalization Activations", exists. It trains verbal reports against probe, feature, ablation and patching ground truth. No results are posted, and the search did not confirm the mentor.
  - Neither changes the manuscript's claims. The manuscript no longer claims novelty for trained interventional self-report; it reports a localisation and null training results.

## 2026-10-22 14:56 (+04:00): manuscript re-audit; final wording fixes

- **Re-audit.** The auditor re-checked the revision (`audit/AUDIT_paper.md`, "Re-check of the revision"). Issues 1–31: 23 were resolved and 8 partly resolved. Of the new or changed numbers, 44 of 46 were right.
- **Fixed after the re-audit:**
  - "at least 88%" is now "at least 87%" (one checkpoint is 109/124);
  - the tie rate is now given for arms O, I and J only (49–63%; X 31% and 63%);
  - "lost" and "do not carry" are now "no longer linearly readable";
  - the coverage of "after the digits" is stated;
  - the Discussion now separates the two tasks and no longer contradicts the two-routes caveat;
  - the human-introspection bullet now says that the swapped-in change, not confidence as such, is undetectable at the report positions;
  - the title is now "Self-Report Positions in Qwen3-1.7B Track the Input, Not the Swapped-In Computation";
  - Figure 2's legend is now below the panels, and the PDF has been rebuilt;
  - RESULTS.md and STAGE0B.md wording on constant answers, ln 2 and the "Very sure" share is corrected.


## 2026-10-25 (+04:00): release preparation; correction to the attribution in frozen files

- **Attribution correction.** `PREDICTIONS.md` (section 1), `PREDICTIONS_V2.md`, `PREDICTIONS_V3.md` and `README.md` say that the core idea is the author's, and `PREDICTIONS.md` lists the method (training on rows where an answer-fixed intervention changed an internal step, testing against an input-only predictor, and decoded neurofeedback as the human analogy) as part of it. The session transcript of 2026-10-21 shows a narrower split:
  - **The author's:** the question, whether a model can be trained to report its computation rather than its input.
  - **Claude's (Opus 5.5), in reply:** the method. That is, training on answer-fixed interventions so that the input cannot predict the label, testing transfer to natural computation against an input-only predictor, and the four-arm comparison.
  - **Claude's, in reply to the author's question about whether humans do this:** decoded neurofeedback as the human analogue.
  The frozen files are not edited. The manuscript's statement on AI use and author contribution, and its introduction, now give this split. The manuscript's pronoun for the author was also replaced with "the author".
- **Release links.** The manuscript's Files paragraph now names the repository and the Zenodo archive; `xurl` is loaded so the link breaks across lines. `make_numbers.py` regenerated `numbers.json` byte for byte, and `audit/paper_audit/recompute_paper.py` reproduced `recomputed_paper.json` byte for byte. No number changed.

- **Unlogged change found.** `code/freeze.py check` reports one mismatch: `README.md`, which `PREDICTIONS.sha256` hashed at the v1 freeze. The study README is a file index; it was extended after the freeze as v2 to v4 and Stage 0 were added, and that change was not logged at the time. It holds no decision rule, and no analysis reads it. The frozen version cannot be recovered from the study folder, so this line of `PREDICTIONS.sha256` fails. Every other hashed file matches, in all three freezes.
- **Same correction, in documents that are not frozen.** `RESULTS.md` (sections 6, 7 and 9) is corrected to the same split. `PREDICTIONS_V4.md`, a draft that was never frozen, says the core idea is the author's; it is kept as written, and this entry corrects it.

## 2026-10-25 (+04:00), later: second attribution pass after an independent evidence check

An independent check of the statements against the session transcripts refined the 2026-10-25 correction. The manuscript, `RESULTS.md` and the release README now say:
- **The author's:** the question, and also the suggestion to answer it by training. The earlier correction gave the author only the question.
- **Prior AI item:** a related item, a trained self-report on a looped model's depth (S4'), was already on the verified shortlist that the coordinating session had shown the author on 2026-10-20.
- **Approvals, as they happened:**
  - The author started the study and asked for it to be finished overnight.
  - v2 ran under v1's frozen rules without a separate approval.
  - The author approved the fourth study and asked for a quick small-scale feasibility check.
  - The coordinating session sent the first pre-freeze rule before telling the author.
  - The author approved a plan that included the second rule without being shown its 0.60 threshold.
  - Earlier text said that the author approved each stage and approved the stopping rules; that overstated it.
- **Timing correction.** The 2026-10-21 entry says the overnight request was relayed at about 22:50 (+04:00). The transcripts put the author's question at 23:10:36 and the relay at 23:10:57.

## 2026-10-25 (+04:00): final manuscript audit (`audit/AUDIT_paper_final.md`)

- **Verdict before fixes:** not ready; one blocking item (the PDF and release files predated the source) and four should-fix items. Every number checked against the raw files is correct.
- **Applied:**
  - the Files paragraph now names the Zenodo-only files and the withheld ones;
  - the Discussion no longer picks one of the two open routes (multiplication item, relation to earlier results, the inference view);
  - the reproducibility section states the one failing freeze hash;
  - nits N2 to N7 are applied;
  - `RESULTS.md` section 9 and `STAGE0B.md` now say "no longer linearly readable" and "follow", not "lost" and "read".
  - The PDF and the release files were rebuilt afterwards.
- **Correction to the 2026-10-22 14:45 entry.** Its count is wrong. The manuscript audit listed 32 issues (4 major, 11 moderate, 17 minor), not 4 and 27. Issue 32 (a plain-language summary before the abstract's numbered items) was not addressed and is left as an option.

# Paper 9 manuscript audit (`paper/main.tex`)

**Verdict: NOT READY AS WRITTEN. The numbers are sound: 337 of 338 values reproduce from the raw files, and the one wrong value is a Table 2 cell (0.32 should be 0.31). The framing is not. The title, the abstract's conclusion and several scope statements claim more than the data show: four major issues and eleven moderate ones need fixing. None of them changes a preregistered verdict.**

**Plain summary.** Almost every number in the draft is right. I recomputed them from the raw result files, re-ran every probe from the stored features, and every probe value came out identical. The problems are in what the text says the numbers mean:
- The title and abstract present one small model's results as a general finding about language models. They call linear probes "causal evidence".
- The abstract says the study "located why the training failed". The probes explain only why the swap-trained reporter had nothing to read. They do not explain why reporters trained on ordinary labels also failed.
- "At every layer from 8 to 27" is said in several places, but only six layers were probed.
- "Recompute from the question" is one of two explanations the design cannot tell apart.
- A few run descriptions, table captions and one figure caption are wrong.
- Ten references that the study's own novelty record lists as must-cite are not cited.

**Who and when.** Independent auditor (Claude Opus 5.5), 2026-10-22, after the draft of 13:46. I modified no existing file. I wrote only this file and the helper scripts and their outputs in `audit/paper_audit/`. I used the CPU only, with at most 4 threads, and downloaded nothing. I did not open any `sealed` folder or anything under `Temp\p6\replayA` or `replayB`.

## How I checked

- **`audit/paper_audit/recompute_paper.py` → `recomputed_paper.json`.** Every number is recomputed from the raw files, without importing study code:
  - v1 from `res_pilot/pilot.json` per-item data and `res/screen.json`;
  - v2 from `res_v2/premise.json` and `res/screen.json`. I re-implemented the frozen bootstrap (seed 11) and ran my own pair-cluster bootstrap;
  - v3 from `res_v3/screen.json`, `gen.json`, the nine `eval_*.json` and the eight `trainlog_*.json` files. I re-implemented `boot_ci` (seed 9), which gives identical CIs, refitted the conditional logit, and ran my own permutation tests with a different RNG;
  - Stage 0a and 0b from `probe.json`, `rows.json` and `s0b_train.json`. I re-derived the A1/A2 split and the input-only predictors from the item screen with the study's seeding rule;
  - the two-bin design from `screen.json`, and the tail positions from `tailpos.json`.
- **`reprobe.py` → `reprobe_all.json`.** I re-ran the probes from the stored features (`Temp\p6\stage0\feats.pt` and `Temp\p9\stage0b\feats_{imm,del,tailpos}.pt`), with Stage 0a's own probe code imported read-only. This covered:
  - all 72 B2 cells;
  - A at the first digit, layer 27;
  - B1 and C at Stage 0a's report position;
  - the λ re-selection on four cells;
  - the 140-cell tail-position grid.

  Every value is identical to the stored files (max |diff| = 0). An independent implementation (sklearn logistic regression with random product folds) reproduces the pattern: donor agreement 0.34–0.42 at the report positions, 0.63–0.68 at the first digit from layer 16, and 0.32–0.35 at the first tail token.
- **`extra_checks_paper.py`.** This checked:
  - the κ definition;
  - the report-tail token counts, with the locally cached Qwen3-1.7B tokenizer: immediate 27, delayed 45, both as stated;
  - which prompt Stage 0a used.
- **`build_table.py`.** This builds the table at the end. A value counts as a match if the printed value equals the raw value rounded half-up at the printed precision.
- **Other checks.**
  - The four figure PDFs were rendered with `pdftoppm` and inspected.
  - `main.pdf` text was extracted and agrees with `main.tex`.
  - The LaTeX and BibTeX logs are clean.
  - The engine semantics were read in `code/p9lib.py`, `code_v3/v3lib.py`, `code_v4/v4lib.py`, `stage0/s0a_*.py`, `stage0b/*.py` and paper 8's `patchlib2.py`.

**Counts.** The table has 177 rows covering 338 numeric values.
- 336 values match.
- One is a fair rounding: "about 8 points" for 8.5.
- One is a mismatch: issue 5.
- 19 rows carry a correct number, or no number, attached to a statement that is wrong or overstated. They are flagged CLAIM and explained below.

## Issues, ranked by severity

### Major

**1. The title and the abstract's last sentence claim a general, causal result that the evidence does not give.**
- *Location:* title (l.26); abstract, last sentence (l.45); Contribution 3 (l.71).
- *Problem:* "Language-Model Self-Reports Recompute from the Input Rather Than Read the Computation: Causal Evidence from Swaps and Probes" has three problems.
  - **Scope.** It generalises from one 1.7B model, two tasks and forced-letter reports.
  - **The two-hop task.** The only causal behavioural evidence comes from this task, and its preregistered verdict says the opposite of a demonstration: "SAME WINDOW: this bed cannot separate reading the computation from reconstructing from the input".
  - **The multiplication task.** "Recompute" rests on exploratory linear probes of the untrained model. Probes show decodability, not causation. v3's untrained report always answered "Very sure", so v3 gives no behavioural evidence about what the report reads.
- *Evidence:* `res_v2/analysis_v2.json` "verdict"; `stage0/STAGE0.md` and `stage0b/STAGE0B.md` ("exploratory"); RESULTS.md §5 ("The untrained model always answers 'Very sure'").
- *Observation the authors may want.* Stage 0b's untrained immediate report is the one behavioural signal in the multiplication task, and the paper does not report it. Its natural accuracy is 0.347, against about 0.50 expected from its answer rates. From `s0b_train.json` N.imm, it answers "Very sure" on 19 of 53 "Very sure" items and 47 of 71 "Less sure" items. So it tracks the input-implied bin, inverted. It is exploratory and on 124 items.
- *Fix:*
  - Title: "Can Qwen3-1.7B Be Trained to Report Its Own Computation? Answer-Fixed Swaps, Null Training Results, and Probes of the Report Position".
  - l.45: see issue 4.
  - Drop "Causal Evidence from ... Probes" everywhere.

**2. "Located why the training failed" overclaims: the probes explain arm I's failure, not arm O's or arm X's.**
- *Location:* abstract l.39 ("and then located why the training failed"); l.65 ("The probe and short-training stages that explain the failure"); Discussion l.317–321.
- *Problem:* The probes show that the swapped-in value is not linearly present where the report is computed, so arm I had nothing to read. They cannot explain why arm O failed in Stage 0b:
  - O was trained on natural two-label confidence;
  - that label is linearly readable at the report position: input-only probe C 0.69–0.77 at Stage 0a's report position, and natural-row CV accuracy 0.66–0.68 in the cells I re-checked, against a 0.51 natural majority;
  - question-only predictors reach 0.895 on the same items;
  - yet O never beat the majority answer (0.573).

  In v3 the text-only arm X failed as well. The paper itself says (l.193) the verdict "does not show that observational labels do better". A general learnability failure of the training set-up remains unexplained. The Discussion's first bullet ("a reporter on ordinary labels is rewarded for refining that estimate") does not fit O staying at the majority answer.
- *Evidence:* `stage0/probe.json` rep_L*_K2 C_input_probe.acc 0.688–0.773; `reprobe_all.json` lam_scores (0a rep L16 0.666, imm rep L20 0.675, del rep L20 0.663); `s0b_train.json` Pstar_A2_acc and imm_O_* acc_nat ≤ 0.573.
- *Fix:*
  - l.39: "Here we first tried to train the first, and then probed where the information the report would need is lost."
  - After abstract item (4), add: "Reporters trained on ordinary labels failed too, although the label they needed is linearly readable at the report position, so the training failure is only partly explained."
  - l.65: delete "that explain the failure".
  - Discussion l.319: replace the first bullet with "The swap labels need a route that does not exist; the ordinary labels do not, and their failure is unexplained (possibly the loss, the letter format or the budget)."

**3. Layer and position coverage is overstated in six places.**
- *Location:* l.42 and l.234 ("at every layer from 8 to 27"); l.44 and l.296 ("At layers 16–27"); l.71 ("At every later position ... at every layer"); l.299 ("there is no position after the answer at which a report could simply read the swapped-in value"); l.313 ("at every layer").
- *Problem:*
  - The report positions were probed at six layers: 8, 12, 16, 20, 24 and 27.
  - The tail positions were probed at five layers: 12, 16, 20, 24 and 27.
  - Positions were probed along the immediate report only. The delayed report was probed at its final token only.
- *Evidence:* `probe.json` keys (rep_L{8,12,16,20,24,27}); `tailpos.json` "layers": [12, 16, 20, 24, 27]; `s0b_tailpos.py` (immediate tail only).
- *Fix:*
  - l.42 and l.234: "at each of the six layers probed (8, 12, 16, 20, 24 and 27)".
  - l.44 and l.296: "at layers 16, 20, 24 and 27".
  - l.71: "At every position of the immediate report, and at the final position of the delayed report, the state follows the input-implied confidence at every layer probed."
  - l.299: "So no position of the immediate report carries the swapped-in value in a linearly readable form."
  - l.313: "at every layer probed".

**4. "Recompute confidence from the question" is not separated from a second route that the design leaves open.**
- *Location:* l.45 ("positions after the answer recompute confidence from the question"); l.297; l.310; title.
- *Problem:* The swap replaces the digit positions' block outputs only from layer 16 up. So blocks 0–16 of every later position still attend to the recipient's own digit states. Those states carry the recipient's confidence: at the first digit, layers 8 and 12 agree with the recipient on 0.62–0.66. A later position can therefore follow the recipient by reading the earlier layers of the answer positions' own computation, not only by recomputing from the question tokens. The probes cannot tell these apart.
- *Evidence:* `code_v4/v4lib.unit_swap4` (layers 16–27); `code/p9lib.py` ("Layer l means the OUTPUT of block l"); paper 8 `patchlib2.PatchModel2._hook` (forward hook on block outputs); `probe_imm` R0_L8/L12_K2 B2.agree_recipient 0.656 and 0.625. The paper itself states the block-17 boundary (l.297), which is correct.
- *Fix:*
  - l.45: "In this model, positions after the answer follow the confidence implied by the recipient's own input. The design cannot tell whether they recompute it from the question or read it from the unswapped lower layers (0–15) of the answer positions."
  - Add to Limitations: "**Swap depth.** The swap starts at layer 16, so later positions still see the recipient's own answer-position states in blocks 0–16; 'recomputes from the question' and 'reads earlier layers of the answer positions' are not separated."

### Moderate

**5. MISMATCH in Table 2: one first-option share is 0.31, not 0.32.**
- *Location:* l.277, row "O, immediate, rank 8, lr 10^-3", first-option share "0.32 / 0.00".
- *Evidence:* `s0b_train.json` runs.imm_O_r8_1e-3.evals.150.frac_pick0_nat = 0.3145161… (39/124). `numbers.json` stores 0.315, which was then re-rounded to 0.32. STAGE0B.md correctly prints .31.
- *Fix:* "0.31 / 0.00". Round two-decimal values from the raw floats, not from the three-decimal `numbers.json` values. The same trap would have turned X1's final loss, 1.3548, into 1.36; the text has 1.35, which is correct.

**6. The v1 calibration figure is described as something it is not, and the third failed check is omitted.**
- *Location:* l.152–153.
- *Problem:* "the readout named the right country for 0.116 of 164" is not what 0.116 measures.
  - The frozen code counts a city as a miss when the readout's two questions (country and capital) disagree.
  - The readout named the right country for 58 of 164 (0.354).
  - The same check also failed PG3: swap transfer at layer 0 was 0.754, against a threshold of 0.90.
- *Evidence:* `code/stage_pilot.py` l.64 and l.71 (cb is None when not reliable); `pilot.json` P1.per_item (19/164 frozen; 58/164 without the agreement filter); RESULTS.md footnote ᵃ; DEVIATIONS 23:56 correction 4; `pilot_gates.json` PG3 transfer 0.754.
- *Fix:* l.152: "the readout named the right country, with both readout questions agreeing, for 0.116 of 164 pilot cities whose country the model knows (threshold 0.80; 0.354 if disagreement is not counted as a miss);". Add the bullet: "a swap of R from layer 0 moved the readout to the donor's country on 0.754 of reliable rows (threshold 0.90)."

**7. The Stage 0b runs are misdescribed.**
- *Location:* l.255–256.
- *Problem:*
  - The rank-32 run of arm I used the immediate format only. "Arm I, immediate and delayed formats: ... and rank 32" implies both formats.
  - Arm O was run at rank 8 with lr 10^-3 only, not at 10^-4. "Arm O ... the same settings" implies both.
- *Evidence:* `s0b_train.json` runs: imm_I_r8_1e-3, imm_I_r8_1e-4, del_I_r8_1e-3, del_I_r8_1e-4, imm_O_r8_1e-3, imm_I_r32_3e-4, imm_O_r32_3e-4. There is no del rank-32 run and no O run at 10^-4.
- *Fix:*
  - "**Arm I:** rank 8 at learning rates 10^-4 and 10^-3 for up to 300 steps, in both formats; rank 32 at 3×10^-4 for up to 900 steps, immediate format only."
  - "**Arm O, immediate format:** rank 8 at 10^-3 for up to 300 steps, and rank 32 at 3×10^-4 for up to 900 steps."

**8. "The bridge" wording contradicts the study's own logged correction.**
- *Location:* l.40 ("the bridge country is held at the city's own tokens"); l.69; l.143 heading ("the bridge is held at the input tokens"); l.312 ("because the bridge lives at the input tokens").
- *Problem:* What was measured is which country a placeholder readout and the self-report recover. The two-hop computation's own bridge was not measured. The donor and recipient share the continent answer by design.
- *Evidence:* DEVIATIONS 23:56 correction 3 ("'The bridge' itself was not measured ... the confirmed claim is v2's frozen verdict wording"); RESULTS.md §3, "The instrument". "H1_bridge_held_at_city_tokens" is the preregistered hypothesis label, but the audit-corrected reading is the narrower one.
- *Fix:*
  - l.40: "the country that later questions recover is held at the city's own tokens".
  - l.143: "Two-hop questions: later questions read the country from the city tokens (preregistered verdict SAME WINDOW)".
  - l.69: "The country that a readout and the report recover is held at the city tokens up to layer 11...".
  - l.312: "because the country they name is read from the input tokens".

**9. The Table 1 caption misdescribes the T2 families and the averaging.**
- *Location:* l.198.
- *Problem:*
  - "T2: changed rows of two untrained families" is wrong for arm J. The bin-vector family (layer 18, scale 2) is exactly J's training intervention, applied to held-out items.
  - "Accuracies are averaged over two seeds", but the T3 column prints the two seeds separately.
- *Evidence:* `gen.json` train.J entries use jvec[c] at cfg.la = 18 and scale s_add = 2 (`code_v3/trainlib_v3.example_unit`); tests.T2c rows are arm "add" with the same layer and scale.
- *Fix:* "T2: changed rows of two families not used to train arm I: a single-layer swap at layer 22 (335 rows) and a bin vector at layer 18 (312 rows; this is arm J's training family, on held-out items). T3 accuracies are given per seed; all other accuracies are averaged over the two seeds."

**10. Stage 0a's report is not "the turn-2 report used in v3".**
- *Location:* l.236.
- *Evidence:* `stage0/s0a_extract.py` builds the tail as `m.tail(v4lib.Q_REPORT, v4lib.rotate2(v4lib.BINS, 0))`. That is v4's two options ("Very sure", "Less sure") in one order. v3's report had four options in four rotations. Stage 0a did keep v3's region R (the digits plus `<|im_end|>\n`) and a new user turn.
- *Fix:* "a turn-2 report as in v3 (v3's region R, a new user turn), but with v4's two options in one order (Stage 0a);".

**11. Figure 2: the caption's reading rule is wrong for the four-label panel, and a label overprints the legend.**
- *Location:* caption l.247; `fig_probe_layers.pdf`.
- *Problem:*
  - "Below the dotted chance line the probe follows the recipient" holds only for two labels. With four labels, all report-position curves lie above the 0.25 line (0.255–0.310), yet the probe agrees more with the recipient than with the donor in all 18 report cells (differences −0.008 to −0.112).
  - In the rendered PDF, the red "pre-freeze bar 0.60" text sits on top of the legend entry "immediate report (27 tokens later)".
- *Evidence:* `probe_*.json` rep_L*_K4 B2.acc and B2.agree_recipient; a render of the PDF at 500 dpi.
- *Fix:*
  - Caption: "With two labels, values below the dotted 0.5 line mean the probe follows the recipient. With four labels, a probe can agree with neither; at the report positions it agrees more often with the recipient than with the donor in every cell (not plotted)."
  - `make_numbers.py` l.188: move the label, e.g. `ax.text(27, 0.61, "pre-freeze bar 0.60", ha="right", ...)`, and place the legend at lower right or outside the axes.

**12. Bonferroni: "the 20 ... tests that were run" omits 36 per-seed tests that were also run.**
- *Location:* l.220.
- *Evidence:* `audit/v3_checks.json` has 20 pooled tests; `audit/v3_arm_I_signal.json` has 36 per-seed tests on the same rows (3 arms × 2 seeds × 6 sets). Over all 56 tests, the threshold 0.05/56 = 0.0009 keeps the T1 result (p = 0.0006) but not the bin-vector result (p = 0.0012). Both p-values are Monte Carlo with 5,000 permutations; my re-run gave 0.0004 and 0.0006.
- *Fix:* "Both survive a Bonferroni correction over the 20 pooled arm-by-row-set tests (p < 0.0025). The audit also ran 36 per-seed tests; over all 56, only the T1 result survives."

**13. Table 2's "First-option share" is not what the column measures, and the untrained row omits values that exist.**
- *Location:* l.264, l.270, l.272.
- *Problem:* The metric is the share of natural items answered "Very sure" (bin 0) after the scores are averaged over both option orders. It is not the share answered with the option shown first. The untrained row shows "--", although the values are 0.53 (immediate) and 0.96 (delayed).
- *Evidence:* `stage0b/s0b_train.py` score(): frac_pick0_nat = share with pick == 0; `code_v4/trainlib_v4.eval_reports` uses rot_scores2 over both orders and pick2 → bin index. `s0b_train.json` N.imm/del.frac_pick0_nat = 0.532 / 0.960.
- *Fix:* Rename the column '"Very sure" share' and caption it: '"Very sure" share: the share of natural items answered "Very sure" (scores averaged over both option orders); values near 0 or 1 mean a constant answer.' Fill the untrained row with "0.53 / 0.96".

**14. Ten verified references from the study's own must-cite list are missing, including the closest prior work on trained introspection.**
- *Location:* §2 Related work (l.85–96).
- *Evidence:* NOVELTY.md "Must-cite list for a write-up". All of the following are verified in CITATION_CHECK.md and present in `refs_new.bib`, and none is cited:
  - hahami2026introspection (SFT on the model's own perturbed forward passes);
  - fonsecarivera2026steering (trained detection of steering);
  - pepper2026learning (adapters that recover multi-hop bridge entities);
  - plunkett2025selfinterpretability;
  - anon2026posthoc (external reporters match self-reporters; training gains do not transfer);
  - nguyen2026prefills;
  - cheah2026faithfulness;
  - karvonen2026counterfactual;
  - morris2025bypassing (causal bypassing);
  - bhan2026neurofaith (bridge decodable on only 48–65% of items).

  NOVELTY.md also asks for a re-check, "before any write-up", of the concurrent SPAR project and of any arXiv version of 5P7dVpGNPO. No record of that re-check exists.
- *Fix:* Under "Training a model to report its own states", add, using CITATION_CHECK.md's wording: "Supervised training on a model's own perturbed forward passes improved localisation of injected concepts \citep{hahami2026introspection}, and fine-tuning taught models to detect injected steering vectors \citep{fonsecarivera2026steering}. Models have been trained to report the attribute weights behind their own choices \citep{plunkett2025selfinterpretability}, and adapters trained on labelled vectors recover multi-hop bridge entities \citep{pepper2026learning}. External reporters given the same inputs matched self-reporters, and in-domain training gains did not transfer \citep{anon2026posthoc}; introspection training did not transfer to a tampering probe \citep{nguyen2026prefills}." Under the criteria, add "and interventions must rule out causal bypassing \citep{morris2025bypassing}". Under two-hop queries, add "a correct answer often comes without a decodable bridge \citep{bhan2026neurofaith}". Record the SPAR and 5P7dVpGNPO re-check.

**15. "Every number and figure in this manuscript is generated from the result files by make_numbers.py" is not true as stated.**
- *Location:* l.369.
- *Evidence:*
  - `main.tex` has no generated macros: every number is typed. The transcription produced issue 5.
  - Several numbers are not in `numbers.json` at all:
    - 164 and 194 (v1);
    - the thresholds 0.80 and 0.70;
    - 45 tokens;
    - 5,000 permutations, 20 tests;
    - 3,000 steps;
    - 6–48 items, 41/48, 47/48, 10/10;
    - 0.90 (paper 8);
    - 28 blocks and 8 GB;
    - the combined range 0.35–0.41, which was taken by hand from three separate spans.
- *Fix:* "The numbers were read from the result files by `paper/make_numbers.py` (`numbers.json`) or from the files named in the text, and typed into the manuscript; `audit/AUDIT_paper.md` checks each against the raw files." Better still, write a `numbers.tex` of macros from the raw values and `\input` it.

### Minor

**16. Abstract item (4): the range does not match its framing.** l.43 frames the range as swap-trained reporters "asking right after the answer". For the immediate-format arm-I checkpoints it is 0.47–0.52 (minimum 0.469). The 0.46 comes from the delayed-format arm-I run (0.463) or from arm O (0.462). `make_numbers.py` summary.I_balanced_range covers arm I in both formats. *Fix:* "balanced accuracy on held-out swaps was 0.47–0.52 for the swap-trained reporters (0.46–0.52 over all runs; chance 0.50)".

**17. The first-digit and last-digit values from layer 16 are ceilings by construction, and the abstract does not say so.**
- *Location:* l.42, l.44, l.233.
- *Problem:* At layers 16 and above, the swapped rows' digit states are bit-exact copies of the donor's own states. So 0.61–0.66 and 0.63–0.66 are the probe's natural accuracy on those states. They show where a natural-trained probe can read, not where information "goes". STAGE0.md caveat 8 says this; the paper does not.
- *Fix:* Abstract (3): "The swapped-in confidence is readable where the swap places it, at the answer digits from layer 16 (by construction): ...".

**18. The timing of the second Stage 0 rule is not recorded inside the study folder.**
- *Location:* l.304 ("fixed by the coordinating session before the result"); l.357.
- *Evidence:*
  - In P9, the one-token-tag rule appears only in RESULTS.md §9 (file last written 13:46) and in DEVIATIONS 13:46. Both timestamps are after the tail-position job, which ran 13:41:05–13:41:35 according to `tools/gpu_queue.log`.
  - DEVIATIONS 13:17 says only "The coordinator decides what follows".
  - Outside P9, `Temp\p9\results_v4.py` (saved at 13:34:23), the script that wrote RESULTS.md §9, already contains the rule text. So the claim is supported, but not by a timestamped record in the study folder.
- *Fix:* Add a DEVIATIONS entry giving the time the rule was received and citing the 13:34 draft, or drop "before the result".

**19. Training-behaviour wording.**
- *Location:* l.257 ("Every one collapsed to a constant answer or flipped between the two"); Fig. 3 caption l.287 ("every run stays near ln 2").
- *Evidence:*
  - 4 of 16 checkpoints mixed the two answers at chance, with first-option shares 0.43, 0.31, 0.67 and 0.57.
  - Both delayed-format runs start at loss 4.36, which is cut off by the 2.5 axis limit.
  - imm_I_r8_1e-3 logged 1.12 at step 50.
- *Fix:* "Twelve of 16 checkpoints gave a nearly constant answer; the rest mixed the two answers at chance. Training loss settled near ln 2 within 50–150 steps." Caption: "every run settles near ln 2 (the delayed-format runs start at 4.4, above the plotted range)".

**20. Figure cosmetics and captions.**
- **Fig. 4** plots layers 16–27 only, while the text cites layer 12. The caption should say "layers 16–27; layer 12 omitted". The red 0.60 line spans all 28 positions, although the bar applies to the first three only.
- **Fig. 1**: the x-axis label is clipped at the right edge, and the tick labels 10, 11 and 12 run together ("101112"). In `make_numbers.py`, widen the figure or shorten the label, and drop tick 9 or 11.

**21. The κ chance model and the CI's origin are not stated.**
- *Location:* l.164.
- *Evidence:*
  - κ = 0.741 uses a chance agreement of 0.863, computed from the per-boundary donor/recipient/other rates. A conventional option-level κ would be 0.952.
  - The pair-resampled CI [0.951, 0.977] comes from the v1/v2 audit. The preregistered row-resampled CI is [0.956, 0.973].
- *Fix:* "(pair-resampled CI [0.951, 0.977], from the audit; preregistered row CI [0.956, 0.973]; κ = 0.741 against the 0.863 agreement expected if both followed their per-boundary donor and recipient rates independently)".

**22. The Discussion conflates an external predictor with the report's own estimate.**
- *Location:* l.319 ("The report's own input-based estimate already matches the natural label most of the time: question-only predictors reach 0.867").
- *Problem:* 0.867 is a recalibrated paraphrase-prompt predictor (`res_v4_design/design.json`, exploratory). The report position's own readable estimate is lower: 0.69–0.77 for the input-only probe and 0.66–0.68 for natural-row CV.
- *Fix:* "The input already predicts most of the natural label: a question-only predictor reaches 0.867 on two-label test items, and a linear probe reads the input-implied label at the report position at about 0.7."

**23. The v2 pairs are not all from the 306 known cities.** Recipients were drawn from all 363 usable test cities; only the donors were known-like (`code_v2/v2_premise.py`). *Fix:* at l.160, "200 pairs: a recipient from the 363 usable test cities and a known donor with a different country, the same continent and the same city token count".

**24. "1,111 train and 1,148 test items, all answered correctly"** (l.185) reads as if every item in the splits was answered correctly. These are the correctly answered items of 1,327 train and 1,329 test items. *Fix:* "Items were split by product; the 1,111 training and 1,148 test items that the model answers correctly are used."

**25. "Frozen with SHA-256 hashes of their preregistration, code and inputs"** (l.346). The hash files cover documents and code, including paper 8 and S2 modules, but no data inputs. For example, `res/screen.json`, which v2 used, is not hashed. *Fix:* "...of their preregistration and code (including the imported modules)...".

**26. Appendix A.**
- "The six that failed" includes "A sum above or below 50. Borderline; not pursued", which is not a failure.
- "Logs are in diag/" omits that the logs for candidates 3–7 are deterministic reruns made on 2026-10-22 (RESULTS.md §4). The figures 41/48, 47/48 and 10/10 match `diag/screen_mult2.log`.
- *Fix:* "The six not used:" and "Logs are in diag/ (for candidates 3–7, labelled deterministic CPU reruns made on 2026-10-22)."

**27. Citation wording.** Section "Citations" below gives the details.
- `elazar2020amnesic` (l.96) is amnesic probing: removing information to test whether it is used. It does not support "natural-trained probes" as a method. Paper 8 cited it for "decodable information need not be used". *Fix:* "...and natural-trained probes; decodable information need not be used \citep{elazar2020amnesic}, so the probes locate information but do not show its use."
- `anon2026interventional` (l.88): change "(single neuron values)" to "(such as neuron activation values)". The verified description is "neuron activation values; three signal types, four LMs".
- `shibata2011perceptual` and `cortese2016multivoxel` (l.61): decoded neurofeedback rewards people for inducing decoded patterns; it does not train them on states. *Fix:* "which rewards people according to states decoded from their own brain activity".
- `nisbett1977telling` (l.50): the verified gist is reports based on "a priori, implicit causal theories". *Fix:* "inferences from implicit theories about what caused a response".
- `fetsch2014effects` (l.327): monkeys' confidence is measured by sure-target choices. *Fix:* "...moves confidence-based choices".

**28. Methods omissions.**
- The tail-position probe fixes λ = 0.5 (`s0b_tailpos.py`), while Stage 0a and 0b select λ per cell. At layer 20, the final tail token gives 0.355 in the tail check and 0.398 (λ = 0.005) in the probe check. Both are below 0.5, but the "same natural-trained probe" (l.293) is not quite the same.
- The probes use one option order (rotation 0).
- B2's "the remainder agrees with the recipient's own label" (l.138) holds only for two labels.
- The bin vectors (l.113) are unit-norm directions scaled to 2× the mean swap-change norm at layer 18.
- In v3, the trained reporters' scores were exactly tied on 49–63% of T3 rows, and ties went to the first option (RESULTS.md §5). That bears on the arm-I signal and is not mentioned.

**29. Audit statements are broader than the audits.** l.368 says "Independent audits recomputed every v1–v3 number". l.411 says auditors "recomputed every reported number". AUDIT_v1v2.md and AUDIT_v3.md checked RESULTS.md, not this manuscript. Before this audit, nobody had independently checked Stage 0a and 0b. *Fix:* "Independent audits recomputed the v1–v3 results (`audit/AUDIT_v1v2.md`, `audit/AUDIT_v3.md`) and every number in this manuscript (`audit/AUDIT_paper.md`)."

**30. Wording strength.**
- l.40: "reading the computation and re-reading the input coincide" → "cannot be told apart in this task".
- l.230: "Probes on the untrained model separate these" → "Probes on the untrained model test the first".
- l.328: "never reaches the position that reports it" → "is not linearly detectable at the positions that report it".
- l.329: "It shows that an untrained 1.7B model does not" → "It shows no linearly detectable route in an untrained 1.7B model".

**31. Inconsistencies with the process records.** Fix these with a new DEVIATIONS correction entry; earlier entries stay unedited.
- **First token after the digits.** RESULTS.md §9, STAGE0B.md §3 and DEVIATIONS 13:46 give 0.33–0.35. The raw maximum is 0.3555, so the paper's 0.33–0.36 is right.
- **Balanced accuracy.** DEVIATIONS 13:17 gives 0.47–0.52. The raw minimum is 0.462, so 0.46–0.52 in the paper and RESULTS is right.
- **Table 2.** The paper's 0.32 against STAGE0B.md's .31: STAGE0B is right.
- **Arm-I signal.** RESULTS.md says "both seeds point the same way"; the paper says "Per seed the evidence is mixed". The per-seed p-values are 0.11 (seed 0) and 0.0006 (seed 1) on T1, and 0.0018 (seed 0) and 0.047 (seed 1) on the bin-vector rows. Say "both seeds are above their independence level on both row sets; each seed is clearly significant on only one of them".
- **Four-label readability at the first digit, layer 27.** The paper's 0.43 comes from the Stage 0b extraction. STAGE0.md and PREDICTIONS_V4.md give 0.44 from Stage 0a: the same quantity, differing by bf16 batch effects. Likewise, Stage 0a gives 0.35–0.37 at layers 8–12 where the paper gives 0.34–0.38. §2 attributes the probe method to Stage 0a, so name the extraction used.

**32. House rules.**
- **Plain language first.** The abstract opens plainly, but items (1)–(5) use undefined jargon: "placeholder readout", "rank-32 adapter", "balanced accuracy", "two-way label". Consider a two-sentence plain summary before them, as in RESULTS.md.
- **Passed:**
  - Attribution: question and core idea are the author's; design, code and analysis are Claude's; the coordinating session's v4 decisions and Stage 0a run are stated (l.407–409; l.134). This agrees with README.md.
  - No chat messages are quoted.
  - The author's background and expertise are not mentioned. "Independent Researcher" is an affiliation, as in paper 8.
  - Nothing implies publication (l.372 "if the author decides to release").
  - The author email matches paper 8's author block.

## Citations

Each sentence was checked against CITATION_CHECK.md's verified description. References that are not in CITATION_CHECK.md come from paper 8's checked bibliography (`build.py`), so I checked them against paper 8's wording instead.

| Key | Where | Matches the verified description? |
|---|---|---|
| nisbett1977telling | l.50, l.326 | Mostly. The verified gist is "a priori, implicit causal theories" (issue 27). |
| carruthers2011opacity | l.50, l.326 | Yes; it follows CITATION_CHECK's caveat ("judgments and decisions", "almost always interpretive"). |
| kiani2009representation | l.52, l.327 | Yes. |
| fetsch2014effects | l.53 yes; l.327 "reported confidence" | Minor (issue 27). |
| fleming2010relating | l.54, l.327 | Yes. |
| cortese2016multivoxel | l.55, l.327 yes; l.61 | l.61 is imprecise (issue 27). |
| shibata2011perceptual | l.61 | Imprecise (issue 27). |
| frederick2012psychophysics | l.103 | Yes. |
| fleming2017self | l.102 | Yes. |
| johansson2005failure | l.101 | Yes. |
| ramdan2026settling | l.39, l.58, l.81, l.315 | Yes. 0.90 matches paper 8 l.201 and l.213 (depth τ in the self-report rows; the location arm reports 0.91, which is the figure CITATION_CHECK quotes). "Two forced-choice self-reports", τ_rep 0.00 and the verdict RECONSTRUCTION FROM INPUT (paper 8 l.211) all match. "Moved ... under swaps of the operands" matches l.214: each report moved under one of two prompt-side swaps. Paper 8's caveat that the question-token swap moved the report less than input reconstruction predicts (l.215) is not mentioned; that is acceptable. |
| anon2026interventional | l.88 | Mostly ("single neuron values"; issue 27). |
| geiping2026proprioception | l.87 | Yes. |
| biran2024hopping | l.95 | Yes. |
| song2025privileged, singh2026reality | l.58, l.77 | Yes; same wording as paper 8 l.47 and l.82. |
| lindsey2025emergent, macar2026mechanisms | l.79 | Yes. "In large models" fits Lindsey's title; Macar has no verified description. |
| lederman2026contentagnostic, hahami2025detecting | l.79 | Yes (paper 8 l.80). |
| ferrara2026owmi | l.80, l.315 | Yes (paper 8 l.81: "eight open models ... at chance while probes read them"). |
| ramdan2026injected | l.81 | Yes; it matches the work's title. |
| binder2024looking, li2025training, guo2026introspective | l.87 | Yes; consistent with their titles and paper 8 l.83. |
| hatua2026verbalization, aoki2026neurofeedback | l.89, l.315 | Yes (paper 8 l.82–83). |
| meng2022locating, geva2023dissecting | l.95 | Yes (paper 8 l.76). |
| geiger2021inducing, zhang2023best | l.96 | Yes (paper 8 l.76). |
| elazar2020amnesic | l.96 | **No**: wrong role (issue 27). |
| yang2025qwen3 | l.65 | Yes. |
| 10 verified but uncited works | - | Issue 14. |

The bibliography builds cleanly: 33 entries, no BibTeX warnings, no undefined citations. `references.bib` carries a stray comment about Graziano left over from the dropped pepper2026learning entry; it is harmless.

## Figures and `make_numbers.py`

All four figure PDFs have the same timestamp as `numbers.json` (13:44) and are produced by `make_numbers.py` from the fields named below. I checked the fields against the raw files.

| Figure | What it plots (fields) | Matches caption? |
|---|---|---|
| `fig_v2_window.pdf` | analysis_v2 B.report and B.readout donor rates with donor_ci bands, against B; dotted line at C.report_names_donor (0.02) | Yes. Cosmetic problems: issue 20. |
| `fig_probe_layers.pdf` | numbers.json stage0 K2/K4 B2_donor for 0b_R0, 0b_rep_immediate, 0b_rep_delayed and 0a_rep_delayed_v3R, layers 8–27; chance lines at 1/K; 0.60 bar on K2 only | Values yes. The reading rule is wrong for K = 4, and the label overprints the legend (issue 11). |
| `fig_training.pdf` | Left: the eight v3 trainlog histories with ln 4. Right: the last checkpoint's loss history for each of the 7 Stage 0b runs (it holds every logged step, because training resumes from checkpoints), with ln 2 | Yes, except "stays near ln 2" (issue 19). |
| `fig_tailpos.pdf` | tailpos B2_donor at the last digit and the 27 tail tokens, layers 16, 20, 24 and 27; lines at 0.5 and 0.60 | Yes, but the caption omits that layer 12 is not plotted (issue 20). |

Other notes on `make_numbers.py`:
- Every field it reads is the right one.
- `summary.I_balanced_range` is computed over arm I in both formats, while the abstract uses it for the immediate format (issue 16).
- `r3()` rounds to three decimals before the text re-rounds to two, which caused issue 5.
- Figure 2 plots three-decimal values; the effect is negligible.

## Internal consistency (paper, RESULTS.md, DEVIATIONS.md)

- **Agree:**
  - the verdicts (SAME WINDOW, NOT SUPPORTED, the v4 stop);
  - every v1–v3 number;
  - the Stage 0a and 0b probe ranges;
  - the tail-position ranges, apart from the items listed in issue 31;
  - the attribution and the suspension note.
- **Disagree:** see issue 31 for the first-token range, the balanced-accuracy range, the 0.31/0.32 cell, the per-seed framing and 0.43/0.44.
- **Correction not carried into the paper:** RESULTS.md and DEVIATIONS (23:56) carry the bridge-wording correction; the paper does not (issue 8).
- **Caveats in RESULTS.md that the paper drops:**
  - PG3's failure (issue 6);
  - the 0.354 alternative calibration (issue 6);
  - the bf16 ties (issue 28);
  - G3 is a weak check: the probability and entropy bins come from the same distribution. The paper says only "All gates passed" (l.190); add "(G3, agreement of two labels from the same output distribution, is a weak check)".

## Full number table

Status key:
- **match:** the printed value equals the raw value rounded half-up at the printed precision.
- **rounding:** the printed value is within half a unit of the raw value.
- **MISMATCH:** a printed digit is wrong.
- **CLAIM:** the number is right, or there is no number, but the statement around it is wrong or overstated (see the issue named).

"Re-run exact" means the probe was re-run from the stored features and gave the identical value. Line numbers in the issues refer to `paper/main.tex` as of 13:46.

| # | Location | Paper value | Recomputed (raw) | Source (path: field) | Status |
|---|---|---|---|---|---|
| 1 | Abstract (1) | 0.840 / 0.795 | [0.84, 0.795] | res_v2/premise.json B rows at B=11 (= analysis_v2.json B.*.11.donor) | match |
| 2 | Abstract (1) | 200 | 200 | premise.json pairs | match |
| 3 | Abstract (1) | 0.025 / 0.025 | [0.025, 0.025] | premise.json B rows at B=12 | match |
| 4 | Abstract (1) | 0.039 / -0.007 / 0.085 | [0.0392157, -0.00653595, 0.0849673] | premise.json + res/screen.json; bootstrap seed 11 (analysis_v2 method) | match |
| 5 | Abstract (1) 'the bridge country is held at the city's own tokens' | claim | - | DEVIATIONS 23:56 correction 3; RESULTS §3 'The instrument' | CLAIM (bridge not measured (issue 8)) |
| 6 | Abstract (2) | 1148 | 1148 | res_v3/screen.json correct test items (= gen.json n_test_items) | match |
| 7 | Abstract (2) 'training loss at ln 4' | ln 4 | [1.3548, 1.43803] | res_v3/trainlog_*.json hist[-1].loss | match |
| 8 | Abstract (3) swap layer | 16 | 16 | gen.json cfg.l1; stage0/rows.json l1 | match |
| 9 | Abstract (3) | 0.61 / 0.66 | [0.609375, 0.660156] | stage0b/probe_imm/stage0/probe.json R0_L{16..27}_K2 B2.acc (re-run exact) | match (ceiling by construction (issue 17)) |
| 10 | Abstract (3) | 27 / 45 | [27, 45] | token count of v4lib.tail_imm / tail_del (cached tokenizer) | match |
| 11 | Abstract (3) 'at every layer from 8 to 27' | 20 layers | 6 probed (8, 12, 16, 20, 24, 27) | probe.json keys | CLAIM (issue 3) |
| 12 | Abstract (3) | 0.35 / 0.41 | [0.347656, 0.414062] | union of rep_L*_K2 B2.acc over stage0, probe_imm, probe_del (re-run exact) | match |
| 13 | Abstract (4) | 900 / 32 | [900, 32] | stage0b/s0b_train.json runs imm_*_r32_3e-4 | match |
| 14 | Abstract (4) | 0.46 / 0.52 | [0.461694, 0.518145] | s0b_train.json acc_swap_balanced, all 16 checkpoints | match (immediate arm-I runs only: 0.469-0.518 (issue 16)) |
| 15 | Abstract (5) 'At layers 16-27' | 16-27 | 16, 20, 24, 27 probed | tailpos.json layers | CLAIM (issue 3) |
| 16 | Abstract (5) | 0.63 / 0.66 | [0.632812, 0.65625] | stage0b/tailpos.json B2_donor L{16..27}_p0 (re-run exact) | match |
| 17 | Abstract (5) | 0.33 / 0.36 | [0.332031, 0.355469] | tailpos.json L{16..27}_p1 | match |
| 18 | Abstract (5) | 0.30 / 0.42 | [0.300781, 0.417969] | tailpos.json L{16..27}_p1..27 | match |
| 19 | Abstract (5) | 27 | 27 | tailpos.json positions minus d4 | match |
| 20 | Abstract, last sentence 'recompute confidence from the question' | claim | - | engine: blocks 0-16 of later positions see the recipient's digit states | CLAIM (issue 4) |
| 21 | Intro, paper 8 | 0.90 | 0.9 | paper8_ms/main.tex l.201, l.213 (depth tau of (iv)@21 in the self-report rows) | match |
| 22 | Intro, paper 8 'two forced-choice self-reports' | 2 | 2 | paper8_ms/main.tex self-report table (arms 1 and 2) | match |
| 23 | §2 Related work 'eight open models' (ferrara2026owmi) | 8 | 8 | paper8_ms/main.tex l.81 (same description, from paper 8's checked bibliography) | match |
| 24 | Setting, blocks | 28 | 28 | HF cache Qwen3-1.7B config.json num_hidden_layers | match |
| 25 | Setting, '8 GB laptop GPU' | 8 | 8 | PREDICTIONS.md l.40 (laptop RTX 5070, 8 GB) | match |
| 26 | Setting, 'LoRA ... all 28 blocks' | 28 | 28 | code/p9lib.py install_lora (every block, s2lib.LORA_TARGETS) | match |
| 27 | §3.1 v1 calibration | 0.116 | 0.115854 | res_pilot/pilot.json P1.per_item + res/screen.json (19/164) | CLAIM (metric misdescribed; 0.354 = 58/164 without the agreement filter (issue 6)) |
| 28 | §3.1 v1 | 164 | 164 | known-like usable pilot cities | match |
| 29 | §3.1 v1 threshold | 0.80 | 0.8 | code/analyze.py GATE min_calib | match |
| 30 | §3.1 v1 reliability | 0.309 | 0.309278 | pilot.json per_item reliable (60/194) | match |
| 31 | §3.1 v1 | 194 | 194 | pilot.json per_item | match |
| 32 | §3.1 v1 threshold | 0.70 | 0.7 | GATE min_reliab | match |
| 33 | §3.1 v2 | 306 | 306 | premise.json known | match |
| 34 | §3.1 v2 R | 0.353 / 0.301 / 0.408 | [0.352941, 0.300654, 0.408497] | premise.json A.R picks; bootstrap seed 11 | match |
| 35 | §3.1 v2 prior | 0.314 / 0.261 / 0.366 | [0.313725, 0.261438, 0.366013] | premise.json A.prior | match |
| 36 | §3.1 v2 R - prior | 0.039 / -0.007 / 0.085 | [0.0392157, -0.00653595, 0.0849673] | premise.json | match |
| 37 | §3.1 v2 'up to about 8 points' | 8 | 8.49673 | upper CI bound x 100 | rounding (8.5 points) |
| 38 | §3.1 v2 city+R | 0.961 / 0.938 / 0.980 | [0.960784, 0.937908, 0.980392] | premise.json A.city+R | match |
| 39 | §3.1 v2 pairs | 200 | 200 | premise.json pairs | match (recipients drawn from 363 usable test cities (issue 23)) |
| 40 | §3.1 v2 report B=11 / B=12 | 0.840 / 0.025 | [0.84, 0.025] | premise.json B rows | match |
| 41 | §3.1 v2 readout B=11 / B=12 | 0.795 / 0.025 | [0.795, 0.025] | premise.json B rows | match |
| 42 | §3.1 v2 agreement | 0.9645 | 0.9645 | premise.json report vs readout picks, 2,000 rows | match |
| 43 | §3.1 v2 pair-resampled CI | 0.951 / 0.977 | [0.951, 0.9765] | my pair-cluster bootstrap (seed 2026); audit/recomputed_v1v2.json ci_pair_cluster | match (audit CI, not the preregistered row CI [0.956, 0.973]) |
| 44 | §3.1 v2 kappa | 0.741 | 0.740895 | chance = 0.863 from per-B donor/recipient/other rates | match (option-level kappa = 0.952 (issue 21)) |
| 45 | §3.1 v2 R-only swap | 0.020 / 0.945 | [0.02, 0.945] | premise.json report_Rswap rows (n = 200) | match |
| 46 | §3.1 'up to about layer 11' | 11 | 11 | transition at B = 12 (premise.json) | match |
| 47 | Fig. 1 caption | 200 / 11 / 12 | [200, 11, 12] | fig_v2_window.pdf; make_numbers.py fig. 1 (analysis_v2 donor, donor_ci; C donor line) | match |
| 48 | §3.2 candidates | 7 | 7 | TASK_SEARCH.md rows 1-7 | match |
| 49 | §3.2 items | 3339 | 3339 | count of 11<=a<=b<=99 with a 4-digit product; screen.json has 3,339 | match |
| 50 | §3.2 cut points p2 | 0.982 / 0.9987 / 0.99993 | [0.982014, 0.998674, 0.999925] | exp(quartiles of lp2 over correct train items) = exp(gen.json th) | match |
| 51 | §3.2 swap layer | 16 | 16 | gen.json cfg.l1 | match |
| 52 | §3.2 different-bin test swaps | 412 | 412 | gen.json T1 diff rows; donor bin carried on 1.00, digits kept 1.00 | match |
| 53 | §3.2 split | 1111 / 1148 | [1111, 1148] | screen.json correct items by split | match (of 1,327 / 1,329 items (issue 24)) |
| 54 | §3.2 training | 8 / 300 / 8 / 0.0001 / 2 | [8, 300, 8, 0.0001, 2] | code_v3/config_base.json lora_rank; gen.json cfg | match |
| 55 | §3.2 'within about 25 steps' | 25 | [1.26843, 1.58342] | trainlog step-25 losses (1.27-1.58) | match |
| 56 | §3.2 final losses | 1.35 / 1.44 | [1.3548, 1.43803] | trainlog_*.json hist[-1].loss (X1 1.3548, I0 1.4380) | match (numbers.json 1.355 would double-round to 1.36; text is right) |
| 57 | §3.2 ln 4 | 1.386 | 1.38629 | math | match |
| 58 | §3.2 median top-letter probability | 0.26 / 0.31 | [0.257566, 0.310313] | eval_*.json T3 lp: median of max letter probability (trained arms incl. X) | match |
| 59 | Table 1 N T3 acc | 0.272 | [0.271777] | eval_*.json T3 picks vs gen.json labels | match |
| 60 | Table 1 N T3 Δ [CI] | -0.325 / -0.368 / -0.281 | [-0.324913, -0.367596, -0.281359] | re-implemented code/analyze.boot_ci (seed 9; P* re-chosen per resample) | match |
| 61 | Table 1 N T1 diff [CI], same | 0.277 / 0.233 / 0.318 / 0.217 | [0.276699, 0.23301, 0.317961, 0.217391] | eval_*.json T1 (412 diff, 345 same rows), seed-averaged | match |
| 62 | Table 1 N T2 pooled [CI] | 0.236 / 0.202 / 0.272 | [0.236476, 0.201849, 0.271756] | eval_*.json T2a + T2c changed rows; recipient-clustered bootstrap | match |
| 63 | Table 1 O T3 acc | 0.233 / 0.234 | [0.233449, 0.234321] | eval_*.json T3 picks vs gen.json labels | match |
| 64 | Table 1 O T3 Δ [CI] | -0.363 / -0.397 / -0.328 | [-0.362805, -0.396777, -0.328397] | re-implemented code/analyze.boot_ci (seed 9; P* re-chosen per resample) | match |
| 65 | Table 1 O T1 diff [CI], same | 0.242 / 0.210 / 0.274 / 0.226 | [0.241505, 0.209951, 0.274272, 0.226087] | eval_*.json T1 (412 diff, 345 same rows), seed-averaged | match |
| 66 | Table 1 O T2 pooled [CI] | 0.263 / 0.237 / 0.289 | [0.262751, 0.236518, 0.288668] | eval_*.json T2a + T2c changed rows; recipient-clustered bootstrap | match |
| 67 | Table 1 I T3 acc | 0.249 / 0.229 | [0.249129, 0.229094] | eval_*.json T3 picks vs gen.json labels | match |
| 68 | Table 1 I T3 Δ [CI] | -0.358 / -0.393 / -0.322 | [-0.357578, -0.393293, -0.321864] | re-implemented code/analyze.boot_ci (seed 9; P* re-chosen per resample) | match |
| 69 | Table 1 I T1 diff [CI], same | 0.301 / 0.267 / 0.335 / 0.229 | [0.300971, 0.26699, 0.334951, 0.228986] | eval_*.json T1 (412 diff, 345 same rows), seed-averaged | match |
| 70 | Table 1 I T2 pooled [CI] | 0.280 / 0.254 / 0.306 | [0.279753, 0.254263, 0.306251] | eval_*.json T2a + T2c changed rows; recipient-clustered bootstrap | match |
| 71 | Table 1 J T3 acc | 0.235 / 0.251 | [0.235192, 0.250871] | eval_*.json T3 picks vs gen.json labels | match |
| 72 | Table 1 J T3 Δ [CI] | -0.354 / -0.389 / -0.318 | [-0.353659, -0.388502, -0.31838] | re-implemented code/analyze.boot_ci (seed 9; P* re-chosen per resample) | match |
| 73 | Table 1 J T1 diff [CI], same | 0.212 / 0.181 / 0.245 / 0.217 | [0.212379, 0.180825, 0.245146, 0.217391] | eval_*.json T1 (412 diff, 345 same rows), seed-averaged | match |
| 74 | Table 1 J T2 pooled [CI] | 0.282 / 0.255 / 0.309 | [0.282071, 0.254658, 0.309452] | eval_*.json T2a + T2c changed rows; recipient-clustered bootstrap | match |
| 75 | Table 1 input-only | 0.597 / 0.597 / 0.245 / 0.238 | [0.59669, 0.59669, 0.244774, 0.237805] | gen.json para_bin; own conditional-logit fit (= para on every item); X seeds' mean scores; train majority | match |
| 76 | Table 1 caption | 0.25 / 1148 / 0.597 / 22 / 335 / 18 / 312 | [0.25, 1148, 0.59669, 22, 335, 18, 312] | gen.json cfg ls/la; T2 changed-row counts | match |
| 77 | Table 1 caption 'two untrained families'; 'averaged over two seeds' | claim | T2c = arm J's training family; T3 column is per seed | gen.json train.J (bin vectors at la = 18, s_add = 2) = tests.T2c | CLAIM (issue 9) |
| 78 | §3.2 permutations | 5000 | 5000 | audit/v3_checks.py (5,000 label permutations) | match |
| 79 | §3.2 signal, T1 diff | 0.301 / 0.253 | [0.300971, 0.252916] | eval_I_s*.json; independence = sum_b P(pick=b) P(label=b) | match |
| 80 | §3.2 signal, T1 diff p | 0.0006 | 0.0006 | audit/v3_checks.json (Monte Carlo); my re-run 0.0004 | match |
| 81 | §3.2 signal, n | 412 / 312 | [412, 312] | row counts | match |
| 82 | §3.2 signal, bin vector | 0.322 / 0.270 | [0.322115, 0.26994] | eval_I_s*.json T2c changed rows | match |
| 83 | §3.2 signal, bin vector p | 0.0012 | 0.00119976 | audit/v3_checks.json; my re-run 0.0006 | match |
| 84 | §3.2 'Bonferroni over the 20 ... tests that were run' | 20 | 20 pooled + 36 per-seed tests were run | v3_checks.json (20); v3_arm_I_signal.json (36) | CLAIM (issue 12) |
| 85 | §3.2 per seed (seed 1) | 0.311 / 0.0006 | [0.31068, 0.0006] | audit/v3_arm_I_signal.json; my re-run p 0.0002 | match |
| 86 | §3.2 per seed (seed 0) | 0.291 / 0.11 | [0.291262, 0.1058] | audit/v3_arm_I_signal.json; my re-run p 0.110 | match |
| 87 | §3.2 rank correlation | 0.91 | 0.908109 | screen.json lp2 vs para_lp2 on the 1,148 T3 items (Spearman) | match |
| 88 | §3.2 recalibrated paraphrase | 0.718 | 0.71777 | screen.json para_lp2 binned at its own train quartiles, vs T3 labels | match |
| 89 | §3.3 rows | 590 / 474 / 256 | [590, 474, 256] | stage0/rows.json (same in probe_imm and probe_del rows.json) | match |
| 90 | §3.3 median p2 | 0.9987 | 0.998674 | exp(rows.json med) | match |
| 91 | §3.3 exact swap; digits kept | exact; all | max abs(lp2 - donor) = 0.0; kept 1.0 | rows.json lp2_meas vs lp2_y; correct_meas (all three extractions) | match |
| 92 | §3.3 first digit, layers 16+ | 0.61 / 0.66 | [0.609375, 0.660156] | probe_imm R0_L{16..27}_K2 B2 (Stage 0a: also 0.609-0.660) | match |
| 93 | §3.3 first digit, layers 8, 12 | 0.34 / 0.38 | [0.34375, 0.375] | probe_imm R0_L8/L12_K2 B2 (Stage 0a: 0.348-0.367) | match |
| 94 | §3.3 first digit, layer 27, all rows | 0.74 / 0.43 | [0.738722, 0.425752] | probe_imm R0_L27 A.acc (re-run exact) | match (Stage 0a: 0.737 / 0.441; STAGE0.md and PREDICTIONS_V4 say 0.44) |
| 95 | §3.3 report positions | 0.35 / 0.41 | [0.347656, 0.414062] | rep_L*_K2 B2 over the three formats | match |
| 96 | §3.3 'at every layer from 8 to 27' | 20 layers | 6 probed | probe.json | CLAIM (issue 3) |
| 97 | §3.3 'the turn-2 report used in v3 (Stage 0a)' | claim | v4 two-option prompt, one option order | stage0/s0a_extract.py | CLAIM (issue 10) |
| 98 | §3.3 token distances | 45 / 27 | [45, 27] | token count (cached tokenizer) | match |
| 99 | §3.3 'with four labels ... majority level' | majority | [[0.319549, 0.334586], [0.31203, 0.328947], 0.295113] | probe_imm/del rep_L*_K4 A.acc vs A.majority 0.295 | match (0.31-0.33 against 0.30: 'near') |
| 100 | §3.3 pitfall, swapped-only probe | 0.68 / 0.74 | [0.675781, 0.738281] | stage0/probe.json rep_L*_K2 B1.acc (re-run exact) | match |
| 101 | §3.3 pitfall, input-only probe | 0.69 / 0.77 | [0.6875, 0.773438] | stage0/probe.json rep_L*_K2 C_input_probe.acc (re-run exact) | match |
| 102 | Fig. 2 caption | 256 / 474 / 0.60 | [256, 474, 0.6] | rows.json; DEVIATIONS 12:33 bar | match |
| 103 | Fig. 2 caption 'Below the dotted chance line the probe follows the recipient' | claim | four-label report curves 0.255-0.310, above 0.25 | probe_*.json rep_L*_K4 B2 | CLAIM (issue 11) |
| 104 | §3.4 training / held-out items | 190 / 124 | [190, 124] | re-derived from screen.json with seeded('v4-half') and seeded('s0b-split') | match |
| 105 | §3.4 arm I settings | 8 / 0.0001 / 0.001 / 300 / 32 / 0.0003 / 900 | [8, 0.0001, 0.001, 300, 32, 0.0003, 900] | s0b_train.json runs | CLAIM (rank-32 run was immediate only (issue 7)) |
| 106 | §3.4 'Arm O, immediate format: the same settings' | claim | O: rank 8 at lr 1e-3 only, plus rank 32 | s0b_train.json runs | CLAIM (issue 7) |
| 107 | §3.4 'Every one collapsed to a constant answer or flipped' | claim | 4 of 16 checkpoints mixed (share 0.31-0.67) | s0b_train.json frac_pick0_nat | CLAIM (issue 19) |
| 108 | §3.4 'loss near ln 2' | 0.693 | [0.579697, 0.780774] | s0b_train.json loss_hist (final losses 0.580-0.781) | match |
| 109 | §3.4 balanced accuracy | 0.46 / 0.52 | [0.461694, 0.518145] | s0b_train.json acc_swap_balanced, all checkpoints | match |
| 110 | §3.4 arm O natural, maximum | 0.573 | 0.572581 | s0b_train.json acc_nat (arm O) | match |
| 111 | §3.4 majority answer | 0.573 | 0.572581 | re-derived from screen.json (A1 majority bin 1, scored on A2) | match |
| 112 | §3.4 question-only predictors | 0.895 | 0.895161 | re-derived recal_para on A2 (file: learned 0.895 too) | match |
| 113 | §3.4 largest budget | 32 / 900 / 8 | [32, 900, 8] | s0b_train.json; trainlib_v4 batch 8 | match |
| 114 | Table 2 untrained balanced imm / del | 0.506 / 0.487 | [0.506048, 0.486559] | s0b_train.json N | match |
| 115 | Table 2 untrained natural imm / del | 0.347 / 0.419 | [0.346774, 0.419355] | s0b_train.json N | match |
| 116 | Table 2 untrained first-option share | -- | [0.532258, 0.959677] | s0b_train.json N frac_pick0_nat | CLAIM (values exist, 0.53 / 0.96 (issue 13)) |
| 117 | Table 2 imm_I_r8_1e-3 steps | 150 / 300 | [150, 300] | s0b_train.json evals keys | match |
| 118 | Table 2 imm_I_r8_1e-3 balanced | 0.510 / 0.501 | [0.510081, 0.501344] | acc_swap_balanced | match |
| 119 | Table 2 imm_I_r8_1e-3 natural | 0.556 / 0.452 | [0.556452, 0.451613] | acc_nat | match |
| 120 | Table 2 imm_I_r8_1e-3 first-option share | 0.03 / 0.93 | [0.0322581, 0.927419] | frac_pick0_nat (share answered 'Very sure', both orders averaged) | match |
| 121 | Table 2 imm_I_r8_1e-3 final loss | 0.665 / 0.707 | [0.664682, 0.70749] | loss_hist[-1].loss | match |
| 122 | Table 2 imm_I_r8_1e-4 steps | 150 / 300 | [150, 300] | s0b_train.json evals keys | match |
| 123 | Table 2 imm_I_r8_1e-4 balanced | 0.518 / 0.474 | [0.518145, 0.47379] | acc_swap_balanced | match |
| 124 | Table 2 imm_I_r8_1e-4 natural | 0.573 / 0.427 | [0.572581, 0.427419] | acc_nat | match |
| 125 | Table 2 imm_I_r8_1e-4 first-option share | 0.00 / 0.97 | [0, 0.967742] | frac_pick0_nat (share answered 'Very sure', both orders averaged) | match |
| 126 | Table 2 imm_I_r8_1e-4 final loss | 0.623 / 0.699 | [0.622847, 0.698926] | loss_hist[-1].loss | match |
| 127 | Table 2 del_I_r8_1e-3 steps | 150 / 300 | [150, 300] | s0b_train.json evals keys | match |
| 128 | Table 2 del_I_r8_1e-3 balanced | 0.463 / 0.498 | [0.463038, 0.497984] | acc_swap_balanced | match |
| 129 | Table 2 del_I_r8_1e-3 natural | 0.532 / 0.435 | [0.532258, 0.435484] | acc_nat | match |
| 130 | Table 2 del_I_r8_1e-3 first-option share | 0.43 / 0.88 | [0.427419, 0.879032] | frac_pick0_nat (share answered 'Very sure', both orders averaged) | match |
| 131 | Table 2 del_I_r8_1e-3 final loss | 0.683 / 0.781 | [0.683236, 0.780774] | loss_hist[-1].loss | match |
| 132 | Table 2 del_I_r8_1e-4 steps | 150 / 300 | [150, 300] | s0b_train.json evals keys | match |
| 133 | Table 2 del_I_r8_1e-4 balanced | 0.518 / 0.482 | [0.518145, 0.481855] | acc_swap_balanced | match |
| 134 | Table 2 del_I_r8_1e-4 natural | 0.573 / 0.427 | [0.572581, 0.427419] | acc_nat | match |
| 135 | Table 2 del_I_r8_1e-4 first-option share | 0.00 / 1.00 | [0, 1] | frac_pick0_nat (share answered 'Very sure', both orders averaged) | match |
| 136 | Table 2 del_I_r8_1e-4 final loss | 0.649 / 0.697 | [0.648845, 0.696731] | loss_hist[-1].loss | match |
| 137 | Table 2 imm_O_r8_1e-3 steps | 150 / 300 | [150, 300] | s0b_train.json evals keys | match |
| 138 | Table 2 imm_O_r8_1e-3 balanced | 0.462 / 0.518 | [0.461694, 0.518145] | acc_swap_balanced | match |
| 139 | Table 2 imm_O_r8_1e-3 natural | 0.516 / 0.573 | [0.516129, 0.572581] | acc_nat | match |
| 140 | Table 2 imm_O_r8_1e-3 first-option share | 0.32 / 0.00 | [0.314516, 0] | frac_pick0_nat (share answered 'Very sure', both orders averaged) | MISMATCH (issue 5: 39/124 = 0.3145 prints as 0.31; 0.32 is a double rounding of numbers.json 0.315) |
| 141 | Table 2 imm_O_r8_1e-3 final loss | 0.580 / 0.735 | [0.579697, 0.73459] | loss_hist[-1].loss | match |
| 142 | Table 2 imm_I_r32_3e-4 steps | 300 / 600 / 900 | [300, 600, 900] | s0b_train.json evals keys | match |
| 143 | Table 2 imm_I_r32_3e-4 balanced | 0.501 / 0.489 / 0.469 | [0.501344, 0.488575, 0.469086] | acc_swap_balanced | match |
| 144 | Table 2 imm_I_r32_3e-4 natural | 0.460 / 0.452 / 0.419 | [0.459677, 0.451613, 0.419355] | acc_nat | match |
| 145 | Table 2 imm_I_r32_3e-4 first-option share | 0.90 / 0.67 / 0.99 | [0.903226, 0.669355, 0.991935] | frac_pick0_nat (share answered 'Very sure', both orders averaged) | match |
| 146 | Table 2 imm_I_r32_3e-4 final loss | 0.764 / 0.723 / 0.669 | [0.763628, 0.723056, 0.66947] | loss_hist[-1].loss | match |
| 147 | Table 2 imm_O_r32_3e-4 steps | 300 / 600 / 900 | [300, 600, 900] | s0b_train.json evals keys | match |
| 148 | Table 2 imm_O_r32_3e-4 balanced | 0.518 / 0.518 / 0.476 | [0.518145, 0.518145, 0.475806] | acc_swap_balanced | match |
| 149 | Table 2 imm_O_r32_3e-4 natural | 0.573 / 0.573 / 0.500 | [0.572581, 0.572581, 0.5] | acc_nat | match |
| 150 | Table 2 imm_O_r32_3e-4 first-option share | 0.00 / 0.00 / 0.57 | [0, 0, 0.572581] | frac_pick0_nat (share answered 'Very sure', both orders averaged) | match |
| 151 | Table 2 imm_O_r32_3e-4 final loss | 0.740 / 0.694 / 0.700 | [0.739687, 0.693546, 0.699504] | loss_hist[-1].loss | match |
| 152 | Table 2 caption | 24 / 62 / 0.50 / 124 / 0.573 / 0.895 | [24, 62, 0.5, 124, 0.572581, 0.895161] | s0b_train.json n_diff / n_same / n_nat; re-derived P* | match |
| 153 | Table 2 caption 'First-option share: ... answered with the first option' | claim | metric = share answered 'Very sure' (bin 0), both orders averaged | stage0b/s0b_train.py score(); code_v4/trainlib_v4.eval_reports | CLAIM (issue 13) |
| 154 | Fig. 3 caption | 8 / 4 / 2 / 300 / 32 / 900 | [8, 4, 2, 300, 32, 900] | trainlog_*.json (8 runs); s0b_train.json | match |
| 155 | Fig. 3 caption 'every run stays near ln 2' | claim | delayed runs start at 4.36; logged peaks up to 1.12 | s0b_train.json loss_hist | CLAIM (issue 19) |
| 156 | §3.5 positions; swaps | 27 / 256 | [27, 256] | tailpos.json | match |
| 157 | §3.5 last digit, layers 16+ | 0.63 / 0.66 | [0.632812, 0.65625] | tailpos.json L*_p0 | match |
| 158 | §3.5 last digit, layer 12 | 0.36 | 0.359375 | tailpos.json L12_p0 = 0.3594 | match |
| 159 | §3.5 first token after the digits | 0.33 / 0.36 | [0.332031, 0.355469] | tailpos.json L*_p1 (0.3320-0.3555) | match (RESULTS, STAGE0B and DEVIATIONS say 0.33-0.35 (wrong there, issue 31)) |
| 160 | §3.5 first three positions | 0.39 | 0.390625 | tailpos.json L*_p1..3 maximum (incl. layer 12: still 0.3906) | match |
| 161 | §3.5 all 27 positions | 0.30 / 0.42 | [0.300781, 0.417969] | tailpos.json L16..27, p1..27 | match |
| 162 | §3.5 'at layers 16-27' | 16-27 | 16, 20, 24, 27 | tailpos.json layers | CLAIM (issue 3) |
| 163 | §3.5 swap from layer 16; block 17 | 16 / 17 | [16, 17] | code/p9lib.py (layer l = output of block l); paper8 patchlib2 forward hook on block outputs | match |
| 164 | §3.5 'no position after the answer' | claim | immediate tail only (plus the last delayed token) | tailpos.json; probe_del | CLAIM (issue 3) |
| 165 | Fig. 4 caption | 256 / 0.60 | [256, 0.6] | tailpos.json; rule text in Temp\p9\results_v4.py at 13:34:23 | match (layer 12 not plotted (issue 20)) |
| 166 | §4 token distances | 27 / 45 | [27, 45] | token count | match |
| 167 | §4 'at every layer' | claim | 6 probed layers | probe.json | CLAIM (issue 3) |
| 168 | §4 question-only predictors, two labels | 0.867 | 0.866725 | res_v4_design/design.json recal_para; re-derived from screen.json | match (an external predictor, not 'the report's own estimate' (issue 22)) |
| 169 | §4 'modestly linear' | 0.74 | 0.738722 | probe_imm R0_L27_K2 A | match |
| 170 | §4 limitations, budgets | 32 / 900 / 190 / 8 / 300 / 3000 | [32, 900, 190, 8, 300, 3000] | s0b_train.json; gen.json cfg; PREDICTIONS_V4.md ladder (3,000 steps) | match |
| 171 | §4 limitations, different-label swaps | 24 | 24 | s0b_train.json n_diff | match |
| 172 | §5 freezes | 3 | 3 | PREDICTIONS.sha256, PREDICTIONS_V2.sha256, PREDICTIONS_V3.sha256 (+ .amend) | match (hashes cover documents and code, not data inputs (issue 25)) |
| 173 | §5 bars | 0.60 / 0.60 / 3 | [0.6, 0.6, 3] | DEVIATIONS 12:33; RESULTS §9 | match |
| 174 | §5 Stage 0b result | 0.35 / 0.40 / 0.39 | [0.347656, 0.398438, 0.390625] | probe_imm rep_L*_K2 B2; tailpos first three positions | match |
| 175 | App. A screens | 7 / 6 / 48 | [7, 6, 48] | TASK_SEARCH.md (item counts 6-48) | match ('six that failed' includes one 'not pursued' (issue 26)) |
| 176 | App. A candidate 7 | 41/48; 47/48; 10/10 | 41/48; 47/48; 10/10 | TASK_SEARCH.md row 7; diag/ rerun logs | match |
| 177 | AI-use section model id | claude-opus-5-5 | claude-opus-5-5 | session model id | match |

## Re-check of the revision

**Re-check verdict: nearly ready.**
- Of issues 1–31, 23 are resolved, 8 are partly resolved, and none is unresolved.
- I recomputed 46 new or changed values from the raw files.
  - 44 are correct.
  - 1 is wrong: l.316 "at least 88%"; one of the twelve checkpoints is at 87.9%.
  - 1 is right only for arms O, I and J: the ties range 49–63% at l.248.
- One build problem: `main.pdf` (14:44:37) embeds an older Figure 2 than the current `fig_probe_layers.pdf` (14:45:44). In that older version, the 0.60 line runs through the legend and the bar label sits on the blue curve. Rebuild the PDF.
- Everything else left is wording.

**What I checked.**
- `paper/main.tex` (14:44:03); `main.pdf` (14:44:37), text extracted and the figure pages rendered at 90 and 300 dpi.
- The regenerated `fig_*.pdf` and `make_numbers.py` (14:45:41/44).
- `references.bib`. The build is clean: 43 entries, no BibTeX warnings, and all ten new keys are in `main.bbl`.
- The edited RESULTS.md, STAGE0B.md and the DEVIATIONS 14:45 entry.
- New helper `audit/paper_audit/recheck_revision.py` → `recheck_revision.json`. It uses CPU, numpy and scipy only, and imports no study code.

### Issues 1–31

| # | Status | Reason |
|---|---|---|
| 1 | partly | Now scoped to Qwen3-1.7B, and "causal evidence from probes" is gone. "Self-Reports ... Follow the Input, Not the Swapped-In Computation" still states as a finding what, in the multiplication task, rests on probes of the report positions; the trained reports were at chance. Consider "In Qwen3-1.7B, Self-Reports Do Not Read a Swapped-In Computation: Answer-Fixed Swaps, Probes and Null Training Results". |
| 2 | resolved | l.38, the arm-O sentence in abstract (5), l.67, §4.5 l.320–324 and the Discussion all say the failure is only partly explained. |
| 3 | partly | Fixed at l.41, l.42, l.74, l.269, l.295–299, l.367 and l.392. New instance at l.374: "not linearly present at any position after the digits", although only the immediate report and the delayed report's final token were probed. |
| 4 | partly | The two-routes caveat is in the abstract (l.44), Setting (l.119), Discussion (l.369) and Limitations (l.391). But l.364 still says the report positions carried "not what the computation they follow contained". That contradicts the second route. Its "Across both tasks, the report positions" is also wrong for the two-hop task, where report positions were not probed. |
| 5 | resolved | Table 2 shows 0.31 (raw 0.3145). |
| 6 | resolved | l.161–163: 0.116 with agreement, 0.354 without (58/164), PG3 0.754 (49/65 reliable rows), all checked. |
| 7 | resolved | l.312–313 match `s0b_train.json`. |
| 8 | resolved | l.39, l.72, l.152 and l.366 now speak of the country that later questions recover. |
| 9 | resolved | The Table 1 caption names J's family and the per-seed T3 column. |
| 10 | resolved | l.271 describes the two-option, one-order prompt. |
| 11 | partly | The caption and the current figure file are fixed. The built `main.pdf` embeds an intermediate Figure 2: legend inside the panel with the dashed 0.60 line through it, and the bar label over the blue curve near layer 24 (page 9 at 300 dpi). Re-run pdflatex. |
| 12 | resolved | l.240 (20 + 36 = 56 tests; only the T1 result survives at 0.05/56 = 0.0009). |
| 13 | resolved | The column is renamed and the untrained row filled (0.53 / 0.96). |
| 14 | resolved | Ten references added. All ten sentences match CITATION_CHECK.md items 15–24. The concurrent-work re-check is recorded in DEVIATIONS 14:45; I cannot verify it offline. |
| 15 | resolved | l.422 is accurate in substance. Nit: make_numbers.py does not "type" anything, and its docstring still says "nothing typed by hand". |
| 16 | resolved | 0.47–0.52 for immediate arm I (raw 0.469–0.518); 0.46–0.52 over all runs (raw 0.462–0.518). |
| 17 | resolved | "By construction" at l.41 and l.265. |
| 18 | resolved | DEVIATIONS 14:45 records the rule's receipt before the 13:41 job, citing the 13:34 file. l.414 and the Fig. 4 caption agree with it. |
| 19 | partly | The Fig. 3 caption and axis are fixed (the 4.36 start is now visible). l.316 "at least 88%" is false for del I r8 1e-3 at 300 steps (109/124 = 87.9%): only 11 checkpoints are at 88% or above. |
| 20 | resolved | Fig. 1 label and ticks, Fig. 4 caption and the bar segment are fixed in both the files and the PDF. |
| 21 | resolved | l.179 states the 0.863 chance model and the CI's origin. |
| 22 | resolved | l.375. |
| 23 | resolved | l.175. 177 of 200 recipients are known-like; every donor is. |
| 24 | resolved | l.199. |
| 25 | resolved | l.403. |
| 26 | resolved | l.437 and l.454. |
| 27 | resolved | l.49, l.52, l.60, l.93, l.103 and l.383. |
| 28 | partly | All items were added (l.120, l.144–147, l.248, l.293). The ties range 49–63% (l.248) holds for arms O, I and J (48.5–63.4%), but not for "the trained reporters" as a whole: X0 is 30.7%. |
| 29 | resolved | l.421 and l.472 are accurate once this re-check is included. |
| 30 | partly | l.39, l.258 and l.385 are fixed. l.384 adopted my suggested wording, "is not linearly detectable at the positions that report it", and that wording was wrong. The natural, input-implied confidence *is* linearly readable there (l.323, l.375; input-only probe 0.69–0.77). Fix: "its swapped-in value is not linearly detectable at the positions that report it". |
| 31 | partly | Done: 0.33–0.36 in RESULTS.md and STAGE0B.md, the DEVIATIONS corrections, the per-seed wording, and the 0.43/0.44 provenance. Still wrong: RESULTS.md l.272 "Answers were constant, losses stayed at ln 2" and l.273 "layers 16–27"; STAGE0B.md l.49 "'1st option' ... first-option picks", l.64–65 "Every one collapses to a constant answer ... Losses stay at ln 2" and l.85 "stay at constant answers". |

### New problems

1. **Stale Figure 2 in `main.pdf`** (under issue 11). Rebuild after `make_numbers.py`.
2. **l.316, "at least 88%".**
   - The minimum among the twelve near-constant checkpoints is 109/124 = 87.9%.
   - Fix: "on at least 87% of natural items (109 or more of 124)".
3. **l.248, the ties range.**
   - Fix: "tied the option scores of arms O, I and J exactly on 49–63% of natural test rows (arm X: 31% and 63%)".
   - Ties went to the first maximum on every tied row: checked.
4. **l.374.**
   - Fix: "...is not linearly present at any position probed after the digits (every position of the immediate report and the delayed report's final token)".
5. **l.364.**
   - Fix: "Across both tasks, the reports and report positions followed what the question implies, not the swapped-in part of the computation they follow."
6. **l.384.** My own earlier suggestion, now corrected; see issue 30.
7. **"Lost" and "do not carry" state an absence that the probes show only as not linearly readable.** This was not raised in the first pass. It occurs at abstract (4) l.42, the §4.4 heading l.291 and l.310; l.38 uses my suggested "is lost".
   - Fix: "is no longer linearly readable" and "do not carry ... in a linearly readable form".
8. **Records.** Besides the items under issue 31:
   - DEVIATIONS 14:45 says the audit raised "4 major and 27 further issues" and that the manuscript "was revised for all of them". The audit lists 32 (4 major, 11 moderate, 17 minor); issue 32 is unaddressed, and residuals remain.
   - A new correction entry should say so.
9. **Optional.**
   - l.323 cites Stage 0a's report position for arm O's readable label, but O was trained on the immediate format. There, the input-only probe gives 0.67–0.72; cite both.
   - Issue 32: a plain-language summary before the abstract's numbered items is still absent.
   - In `references.bib`, comments copied from `refs_new.bib` sit next to the wrong entries. This is cosmetic and not visible in the PDF.

### New or changed numbers, recomputed from the raw files

| Location | Revised value | Recomputed | Source | Status |
|---|---|---|---|---|
| Abstract (3), l.41, l.269 | six layers probed (8, 12, 16, 20, 24, 27) | same | `probe.json` keys | match |
| Abstract (4), l.42, l.295–296 | layers 16, 20, 24, 27 | same (12 also run) | `tailpos.json` layers | match |
| Abstract (5), l.43, l.318 | 0.47–0.52, immediate arm I | 0.469–0.518 | `s0b_train.json` acc_swap_balanced | match |
| l.318 | 0.46–0.52, all runs | 0.462–0.518 | same | match |
| l.161 | 0.354 | 58/164 = 0.3537 | `pilot.json` per_item, `res/screen.json` | match |
| l.163 | 0.754; threshold 0.90 | 49/65 = 0.7538 (reliable rows) | `pilot.json` P2.from0; `pilot_gates.json` PG3; PREDICTIONS.md PG3 | match |
| l.172 | up to 0.085 | 0.08497 | `analysis_v2.json` | match |
| l.175 | 363 usable test cities | 363 | `premise.json` n_test | match |
| l.179 | κ 0.741 against 0.863 | 0.7409; 0.86299 | `extra_checks_paper.json` | match |
| l.240 | 20 pooled; p < 0.0025; 36 per-seed; 56; only T1 survives | 20; 0.0025; 36; 56; T1 p 0.0006 < 0.00089, bin-vector p 0.0012 > 0.00089 | `v3_checks.json`, `v3_arm_I_signal.json` | match |
| l.244–245 | 0.311 (p 0.0006); 0.291 (p 0.11); p 0.0018; p 0.047 | 0.311/0.0006; 0.291/0.1058; 0.0018; 0.0466 | `v3_arm_I_signal.json` | match |
| l.248 | ties 49–63%, to the first option | O, I, J: 48.5–63.4%; X0 30.7%, X1 63.4%; first maximum on every tied row | `eval_*.json` T3 scores | **scope** (new problem 3) |
| l.267 | Stage 0a 0.74 and 0.44 | 0.737; 0.441 | `stage0/probe.json` R0_L27 A | match |
| l.275, Fig. 2 caption | recipient > donor in every four-label report cell | 18 of 18, by 0.008–0.112 | `probe_*.json` rep_L*_K4 B2 | match |
| l.144, l.293 | λ = 0.5; layers 12–27 | same | `stage0b/s0b_tailpos.py` | match |
| l.316 | 12 of 16 near-constant "on at least 88%" | 12 at 87.9–100%; 11 at ≥ 88% | `s0b_train.json` frac_pick0_nat | **MISMATCH** (new problem 2) |
| l.316 | the other four "at about chance" | natural accuracy 0.45–0.53 | same | match |
| l.317 | settled near ln 2 within 50–150 steps | all 7 within ±0.2 of ln 2 by step 150; within ±0.15 only 5 of 7 (single-batch spikes to 0.86) | loss_hist | match (approximate) |
| l.323 | 0.69–0.77 | 0.688–0.773 (Stage 0a); immediate position 0.672–0.723 | `probe.json` C_input_probe | match |
| l.326 | 0.347; 47 of 71; 19 of 53 | 43/124; from 53/71 labels (rowsO) and 66 "Very sure" picks: exactly 19/53 and 47/71; Fisher p = 0.001; independence level 0.495 | `s0b_train.json` N.imm, rowsO | match |
| Table 2 | 0.31; 0.53 / 0.96 | 0.3145; 0.532 / 0.960 | `s0b_train.json` | match |
| l.119, l.391 | blocks 16–27 replaced; donor visible from block 17; recipient in blocks 0–16 | same | `v4lib.unit_swap4`, `p9lib`, paper 8 `patchlib2` | match |
| l.392 | six layers (five in the tail check); only the final delayed position | same | `probe.json`, `tailpos.json` | match |
| Figs. 1, 3, 4 (PDF and files) | Fig. 1 ticks without 9 and 11; Fig. 3 bar segment at positions 1–3, layer 12 omitted; Fig. 4 axis to 4.6 | same | rendered | match |
| Fig. 2 | legend below, label clear of curves | file: yes; `main.pdf`: no | rendered | **stale PDF** |
| Ten new citation sentences (l.82, l.91, l.92, l.94, l.101) | | match CITATION_CHECK items 15–24 | CITATION_CHECK.md | match |

Unchanged values were not re-derived again; they are as in the main table above. They are still present and unchanged in the revision, except Table 2's 0.31, which is now correct.

# Paper 9 preregistration: training a self-report to read the model's own computation

## Plain-language summary

In paper 8, a small language model (Qwen3-1.7B), asked about its own computation, answered from what it could see in the question rather than from what had happened inside it. This study asks whether training can change that. The model answers a two-step question, "Which continent is the country containing Lyon in?". The hidden middle step is the country, and in a follow-up turn we ask the model which country it used.

One version of the model is trained on ordinary examples, where the right country can always be worked out from the city's name. Another is trained on examples where we secretly replaced the model's computation after the city name with its computation for a different city, in another country on the same continent. The question and the answer stay the same, but the country the computation went through changes, so the only way to answer correctly is to read that computation.

The decisive test uses ordinary, untouched questions. Does the trained model name the country its computation actually carried, measured by a separate causal test, more often than the best predictor that sees only the question? Interventional training counts as a success only if it clears that bar while ordinary training does not, and only if the skill carries over to kinds of internal change it was never trained on. If ordinary questions almost never carry a different country from the one the question implies, the test cannot decide, and we will say so rather than guess.

## 1. Status and attribution

- **Idea.** The core idea is the author's (Sam Ramdan, 2026-10-21):
  - training a self-report on rows where an answer-fixed activation intervention changed an internal step;
  - the prediction that observational training teaches inference from the input;
  - testing natural transfer against an input-only predictor;
  - the analogy with decoded neurofeedback.
- **Design.** Claude (Opus 5.5) made every design choice in this document, delegated by the author, and wrote all code. That covers the task details, the causal ground truth, the intervention families, the arms (including arm X), the tests, the thresholds, the decision rule, the power analysis and the pilot rules.
- **Novelty.** See `NOVELTY.md`. The narrowed claim is the first head-to-head comparison of label sources for a task intermediate under answer-fixed interventions, judged on natural computation against an input-only predictor. Observational and interventional training have already been compared for low-level signals (OpenReview 5P7dVpGNPO), and in-place self-reading has been done (Geiping, arXiv:2610.17495).
- **What exists at freeze time: no Qwen3-1.7B output of any kind.**
  - The code was dry-run on CPU with SmolLM2-135M-Instruct on 112 cities. Its outputs sit in `res_tiny/`, `res_pilot_tiny/` and `res_selftest_tiny/`: code tests only, never analysed.
  - The engine self-test (`code/selftest_engine.py`) passed on SmolLM2-135M:
    - bit-exact copies and readouts;
    - an exact float32 match between cached turn 2 and a full pass;
    - LoRA gating;
    - a falling training loss.
- **Order of work.** A chained GPU job (section 12) runs, in order: the engine self-test on Qwen3-1.7B, the item screen, the pilot, the end-to-end self-test on pilot cities, the frozen bed checks, and only then the main study. Pilot parameters are set by the frozen rules in section 10, not by hand.

## 2. Hypotheses

- **H-read (the author's prediction, made concrete).** A reporter trained on interventional labels (arm I) names the causally measured natural bridge on unpatched test items more accurately than the best input-only predictor. A reporter trained on observational labels (arm O) does not.
- **H-family.** Arm I's skill transfers above chance to intervention families it was not trained on (T2).
- **Alternatives:**
  - (a) Neither trained arm beats the input-only bar: training teaches only the trained intervention signature, or natural computation rarely departs from the input.
  - (b) Both beat it: label source does not matter.
  - (c) Arm I beats it only through tamper detection (T4).

## 3. Setting

**Model.** Qwen/Qwen3-1.7B (28 blocks, bf16, thinking off) on the laptop RTX 5070 (8 GB), using the cached weights only. Layer l is the output of block l (0-indexed).

**Conversation:**
- **Turn 1, user:** "Which continent is the country containing {city} in? Answer with the continent name only."
- **Turn 1, assistant:** the chat template's empty think block, then the continent name (teacher-forced), then `<|im_end|>`.
- **Turn 2, user:** a question with four lettered options, ending "Answer with the letter."
- **Turn 2, assistant:** the empty think block, then the prefill "The answer is (". The report is the next-token letter.

**Items.** `code/cities.py` lists 1,018 non-capital cities in 109 countries on five continents. Oceania is excluded, as are countries of disputed continent, cities whose names contain a country name, and names shared by well-known cities elsewhere.
- **Split.** Within each country, a seeded order cycles train, test, pilot, train, test (`build.split_cities`). Roughly 40% are train, 40% test and 20% pilot.
- **Usable.** A city is usable if:
  - it has 1–5 tokens, which excludes 10 cities;
  - it tokenises cleanly, with the turn-1 ids starting with the 2-hop prompt's ids;
  - its 2-hop forced choice is right. The 2-hop answer is the argmax over the five continents' first tokens at the answer position.
- **Candidates.** Each city has four candidate countries on its continent, in a seeded order: its gazetteer country, its free 1-hop answer if that is a different country on the same continent, then countries from the same sub-region, then from the continent.

**Region R.** R is every turn-1 position after the city tokens: " in? Answer with the continent name only.", the chat template, the answer and the closing `<|im_end|>\n`. The city tokens are never intervened on.

## 4. Interventions (all on R only; donors and recipients share the token count and the continent)

| Family | Definition | Used for |
|---|---|---|
| F1 (training swap) | R's block outputs at layers l1..27 replaced by the donor's (in-batch, bit-exact; paper 8's engine) | I training, T1 |
| F2a (single-layer swap) | R's block output at layer ls only replaced by the donor's; later layers recompute | T2 (unseen) |
| F2c (additive country vector) | at layer la, add s_add·b·v̂_c to every R position | J training, T2 (unseen for I) |
| T4-kl | at layer la, add a seeded Gaussian unit direction scaled to s_kl·b | tamper control (KL-matched, primary) |
| T4-norm | the same at s_add·b (norm of F2c) | tamper control (secondary) |

**Definitions:**
- **v̂_c:** country c's train-item mean of the layer-la R states (averaged over positions), minus the mean over its continent's train items, normalised. It is computed for countries with ≥ 3 usable train items.
- **b:** the mean over 300 train (recipient, different-country donor) pairs of the per-position norm of the layer-la R-state difference.
- **Pilot-set parameters.** l1, ls, la, s_add and s_kl are set by the pilot's frozen rules (section 10).
- **Answer-fixed.** The turn-1 answer text is the same continent for donor and recipient. Under F1, R includes the answer-predicting position, so the predicted continent is the donor's, which is the same continent.

## 5. Ground truth: the causal bridge (CB)

**Readout.** For any turn-1 run (natural or intervened), a placeholder row is run in the same batch. It is the same conversation, with the city replaced by a placeholder of equal token count (` X`, ` city X`, ` the city X`, ` the city of X`, ` the city called X Y`). Its R block outputs at **every** layer are copied from the run, so the only country information it can use is the run's computation after the city.

**Copying rules.** For a swap source, the placeholder copies the swapped layers straight from the donor, so no copy is chained within one layer. For an additive source, it copies the pre-add output and gets the same add. Either way it reproduces the source's R states exactly.

**Readout questions.** It then answers two questions, each in four option rotations:
- "Which country was the question about?" over the item's four candidates;
- "What is the capital of the country the question was about?" over their capitals.

**CB.** CB is the argmax of the rotation-averaged country scores. It is **reliable** if the capital readout's argmax names the same country, and unreliable items are excluded wherever CB is the truth.

**Why this counts as causal, not a lens.**
- It is an interchange intervention. The model's own downstream computation is fed only the R states and has to produce a country-specific answer.
- It is not a third-party probe or lens.

**Caveat.** R may carry the city's identity as well as an extracted country, and the readout cannot tell those apart. Both are the model's own computation after the city.

## 6. Reports and arms

**Report question:** "Which country did you use to work out your answer?" with four country options.
- **Scoring.** Each country's score is the mean over the four cyclic option rotations of its letter's log-probability, with the softmax taken over the four letter logits. This cancels letter bias, and the report is the argmax.
- **Training** uses one random rotation per example, with full-vocabulary cross-entropy on the letter.

**Common budget (every trained arm; `code/config_base.json`).**
- LoRA rank 8, α 16, on the seven linear modules of all 28 blocks (S2's `s2lib.LoRALinear`).
- Adam at lr (set by the pilot rule), 10-step warm-up, gradient clip 1.0.
- 300 steps × batch 8, over a fixed list of 2,400 examples per arm.
- Seeds 0 and 1.
- For in-place arms the adapter is active only in turn 2. Turn 1, the computation being reported, is always the unmodified base model's.

| Arm | Turn-1 rows | Label | Notes |
|---|---|---|---|
| N | — | — | untrained base model |
| O | natural train items with reliable CB | CB of the item | observational |
| I | F1 swaps on train pairs, 50% different-country donor / 50% same-country donor | donor's bridge | interventional; same-country rows stop a "never name the city's country" shortcut |
| J | F2c on train items, 50% injected country ≠ / 50% = the recipient's | injected country | IFT-style injection labels |
| X | none: a fresh chat with the transcript as text ("Here is a question that was put to an assistant, and the assistant's answer. ... Which country did the assistant use to work out its answer?") | same examples as O | external observer: same weights and budget, cannot read the original computation; part of the input-only bar |

**Donors and options:**
- **Known donor:** a usable item whose free 1-hop answer, 1-hop forced choice and reliable CB all equal its gazetteer country. For a known donor, the donor's bridge is its gazetteer country.
- **Reference country ref(x):** a recipient's reliable CB, or else its gazetteer country.
- **"Different-country":** a donor's bridge, or an injected country, that differs from both ref(x) and x's gazetteer country. So the label is never the country the input implies.
- **Option sets for I and J rows, and for T1/T2 rows:** {label, ref(x), x's gazetteer country}, filled to four from the sub-region and continent.
- **Training lists alternate the two kinds.** Each recipient is drawn from the train items that have a known donor (I) or a country vector (J) of the needed kind, so every list has exactly 2,400 examples (asserted).
- **Disjoint splits:** train and test use disjoint cities, both as recipients and as donors.

## 7. Tests (test cities only; base-model ground truth from `code/stage_gen.py`)

- **T1, same family, held-out cities.** Each usable test recipient gets one different-country and one same-country F1 swap, where a known donor exists.
  - Primary: accuracy naming the donor's bridge on different-country rows.
  - Also reported: same-country accuracy and the rate of naming ref(x).
- **T2, unseen families.**
  - **Families:** F2a with a different-country donor, and F2c with an injected country ≠ ref(x).
  - **Truth:** the intervened run's CB, with "changed" rows only: reliable CB that differs from both ref(x) and x's gazetteer country.
  - **Chance level is 1/3, not 1/4.** A reporter that only detects tampering and avoids the city's own country picks among three options.
  - **Statistic:** accuracy pooled over both families and averaged over seeds, with a 95% cluster bootstrap CI (clustered by city, 10,000 resamples).
  - **Per-family estimates:** a family is evaluable if it has ≥ 20 changed rows.
- **T3, natural transfer (decisive).** Unpatched usable test items with reliable CB; the truth is CB.
  - **Input-only predictors:**
    - (1) the gazetteer country;
    - (2) the model's fresh-chat 1-hop forced choice;
    - (3) a conditional-logit model over the four candidates, fit on the train items' CB. Its features are: is-gazetteer; is-1-hop-choice; is-free-answer; the rotation-averaged 1-hop log-probability, clipped at −10 and standardised with train statistics. It uses an L2 penalty of 0.01 and is fit by L-BFGS to convergence (max |gradient| < 1e-5 asserted);
    - (4) arm X (two seeds, rotation-averaged scores averaged).
  - **P\*:** the most accurate of the four on the T3 items, re-chosen in every bootstrap resample, which is conservative.
  - **Statistic:** Δ_arm = (arm accuracy against CB, averaged over seeds) − (P\* accuracy), with a 95% bootstrap CI (10,000 resamples of items).
  - **Secondary:** on the items where P\* errs ("divergent"), the rate at which each arm names CB versus names P\*'s choice.
- **T4, tamper control.** For each test item there is a KL-matched random direction (primary) and a norm-matched one (secondary).
  - **Truth:** the perturbed run's CB.
  - **Statistic:** the increase in the report's error rate against CB under perturbation, relative to the same item's clean T3 report, with a 95% bootstrap CI.
  - **Pass:** the upper CI bound ≤ 0.15.
- **T5, capability drift.**
  - **Gated:** 48 general multiple-choice questions (`code/mcq.py`) asked in turn 2, with accuracy compared against N.
  - **Ungated:** the adapter applied to every position on the test cities' 2-hop and 1-hop questions.
  - **Flag:** a drop > 10 points against N, in any of the three. This is descriptive, not part of the verdict.

## 8. Decision rule (applied by `code/analyze.py`; thresholds in its `GATE` dict)

**Validity gates.** Any failure makes the verdict **UNINFORMATIVE**.
- **G1:** 2-hop accuracy on test cities ≥ 0.70, and at least 150 usable test items.
- **G2:** the answer is kept in ≥ 95% of T1 swaps (continent argmax at the answer-predicting position, clean against swapped).
- **G3:** the natural CB readout is reliable on ≥ 0.70 of test items, and CB equals the gazetteer on ≥ 0.80 of "known-like" test items (free and forced-choice 1-hop answers equal the gazetteer).
- **G4:** CB of the different-country T1 swapped rows equals the donor's bridge on ≥ 0.90.
- **G5 (natural variation):** the number of T3 items where P\* errs is ≥ 25, and the simulated power for a strong reader is ≥ 0.80. The strong reader is right on 97% of the items where P\* is right and on 70% of the divergent items, at the realised n and divergence rate. Power is computed with the normal-approximation CI that `power_sim` uses.

Also **UNINFORMATIVE**: no T2 family has ≥ 20 changed rows.

**SUPPORTED requires all of:**
1. **Natural transfer:** Δ_I's 95% CI lower bound > 0.
2. **O does not:** Δ_O's lower bound ≤ 0.
3. **Unseen families:** T2 pooled lower bound > 1/3, and every evaluable family's point estimate > 1/3.
4. **Tamper control:** I passes T4-kl.

**NOT SUPPORTED:** the gates pass but any condition fails. The reasons are listed:
- "I does not beat the input-only bar";
- "O also beats it": observational labels suffice. This is what an O ≈ I result means here, a substantive finding against the author's specific prediction even if both arms read;
- "transfer to unseen families at chance";
- "tamper-sensitive": T2 is not credited.

**Descriptive only.** J's results and X versus O are reported but are not part of the verdict.

## 9. Power and the natural-variation gate

The power figures below come from the frozen `power_sim` in `code/analyze.py`, using the strong-reader model in G5 and 2,000 simulations per cell. With 300 T3 items, power is 0.39 at an 8% divergence rate, 0.74 at 10%, 0.93 at 12% and 1.00 at 15%. With 250 items it is 0.65 at 10% and 0.87 at 12%; with 350 items, 0.80 at 10%. So the test needs roughly 10–12% of natural items to diverge from the best input-only predictor. G5 is evaluated on the realised test data with this frozen rule, so the gate cannot be tuned after the fact. How often natural bridges diverge from what the input implies is unknown. The pilot reports it on pilot cities only, for information.

## 10. Pilot (pilot cities only; frozen rules in `code/stage_pilot.py`, frozen bed checks in `code/make_config.py`)

**Parameters set by the pilot:**

| Parameter | Fixed rule |
|---|---|
| l1 (F1 boundary) | largest B in {0, 4, 8, 12, 16} whose swap-from-B transfer (CB of swapped run = donor's bridge, reliable readouts) is ≥ 0.90; 0 if none |
| ls (F2a layer) | single layer in {6, 10, 14, 18, 22} with the largest transfer (ties: earlier) |
| la, s_add | over la ∈ {10, 14, 18} × s ∈ {1, 2, 4}: the largest change-to-injected rate with answer retention ≥ 0.90 (ties: first in grid); (14, 1) if none qualifies |
| s_kl | random-direction scale (× b) whose median R-KL equals F1's median R-KL (log-log interpolation over 1/16..2) |
| lr | the better of {1e-4, 3e-4} on arm I trained on pilot half A (150 steps) and tested on half B (acc_diff + acc_same, a missing value counting as 0; ties: lower) |

**Bed checks.** If any check fails, the chain stops before gen and the bed is reported as failed. A new design then gets its own preregistration.
- **PG1:** 2-hop accuracy over all screened cities ≥ 0.70, and ≥ 150 usable test items.
- **PG2:** on pilot known-like items, readout calibration ≥ 0.80 and reliability ≥ 0.70.
- **PG3:** F1 transfer at the chosen l1 ≥ 0.90.
- **PG5:** the end-to-end self-test on pilot cities completes.
- **For information only:** the pilot learning-rate check's accuracies (PG4) and the pilot divergence rate.

## 11. My predictions (Claude's priors, recorded before any data)

| Outcome | Prior |
|---|---|
| UNINFORMATIVE through G5 (too little natural variation) | 0.45 |
| NOT SUPPORTED with I failing T3 | 0.25 |
| NOT SUPPORTED with O also beating the bar, or T2 at chance | 0.15 |
| SUPPORTED | 0.15 |

- **T1:** I learns the training family (different-country accuracy ≥ 0.8), and O and N name the city's own country on different-country rows (≥ 0.8 of rows).
- **T2:** F2c transfer for I is weaker than F2a.
- **Untrained N:** follows the city, not the swapped computation, as in paper 8.

## 12. Run plan and budget

- **Driver.** `code/run_chunks.sh` runs `code/job.py pilot`, then `code/job.py main`, through `tools/gpu_lock.py` with labels `p9-pilot-k` and `p9-main-k`.
- **Time slices.** A job starts no new step after 45 minutes and re-queues itself (exit 3), so each job stays near or under 60 minutes.
- **Power.** On battery below 30%, the driver waits for mains power.
- **Order.** Analysis (`code/analyze.py`, CPU) runs after `MAIN_DONE`.
- **Estimate:** pilot about 0.6 GPU-h, main about 1.5 GPU-h, total about 2–2.5 GPU-h (cap 4).

## 13. Deviations

This file is never edited after the freeze. Every change goes in `DEVIATIONS.md` with time, reason and whether analysis data had been seen.

**What the freeze covers.** `PREDICTIONS.sha256` holds the hashes of this file, `NOVELTY.md`, `README.md`, every file in `code/`, and the six files imported read-only from other projects:
- paper 8's `srlib.py`, `patchlib2.py`, `patchlib.py`, `p8lib.py` and `p6lib.py`;
- S2's `s2lib.py`.

**How it is enforced.** `code/job.py` refuses to run unless `code/freeze.py check` passes. A file changed after the freeze must be logged in `DEVIATIONS.md` and its new hash added to `PREDICTIONS.sha256.amend`. This file itself can never be amended. An independent subagent will recompute every number from the raw files and check the verdict before `RESULTS.md` is final.

## 14. Planned follow-ups (not run now)

See `README.md`:
- replication on SmolLM2-1.7B-Instruct, OLMo-2-0425-1B and Qwen3-4B;
- 8–14B models on a rented GPU, only with the author's approval;
- a chain-of-thought faithfulness extension.

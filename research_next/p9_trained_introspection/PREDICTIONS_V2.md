# Paper 9, v2 preregistration: where the two-hop bridge is held, and whether self-report reads it from the same place

## Plain-language summary

The planned training study (v1, `PREDICTIONS.md`) needed a part of the model's processing that changes which country it "used" while leaving its reading of the city name alone. The v1 safety checks stopped the run before any main data existed: the check meant to read out which country the model used did not work. A quick diagnosis on 12 practice cities suggested why. The country is held where the model processes the city's name, at about layers 8–11. It is not stored anywhere after the city name, and the model's own answer about which country it used reads the same place.

If that holds, this task cannot separate "reading the computation" from "reconstructing from the input": the model's knowledge of the country and its reading of the city name are the same thing, in the same place. This v2 checks the diagnosis properly on about 300 held-out cities before anyone draws that conclusion. It trains nothing.

## 1. Status and attribution

- **Context.** v1 is frozen (`PREDICTIONS.md`, `PREDICTIONS.sha256`), and its bed check failed at 22:17 (see `DEVIATIONS.md`). Following v1 section 10, this new design gets its own preregistration.
- **Attribution.** The idea behind paper 9 is the author's (Sam Ramdan). v2's design, hypotheses and thresholds are Claude's (Opus 5.5).
- **What existed at freeze time:**
  - the item screen (`res/screen.json`, all cities: 2-hop and 1-hop answers, candidates);
  - the v1 pilot (`res_pilot/`, pilot cities);
  - the v1 self-test (`res_selftest/`, pilot cities);
  - the diagnoses on 12 pilot cities (`diag/`).
  No readout, report or swap had been run on any test city. A CPU dry run of the v2 code used SmolLM2-135M on the tiny screen (`res_v2_tiny/`).

## 2. Design (Qwen3-1.7B, thinking off; frozen v1 engine `code/p9lib.py` and `code/build.py`, read-only)

**Items.**
- Usable test-split cities: 2-hop answer correct and tokenised cleanly; 363 of them.
- "Known-like" cities are those whose free and forced-choice 1-hop answers both equal the gazetteer country.
- Pairs: up to 200 recipients x, in seeded random order. Each gets a known-like donor y with the same city token count and continent, from a different country.

**Readout.** The readout question is "Which country was the question about?", with four options in four rotations, scored as in v1 (rotation-averaged letter log-probabilities, softmax over the four letters). It is asked in a placeholder context: the city is replaced by a placeholder with the same token count, and states are copied in from the source run.

**Part A (known-like test cities).** Three readout variants:
- **prior:** nothing copied;
- **R:** the post-city region R copied at every layer (v1's readout);
- **city+R:** the city tokens and R copied at every layer.

The statistic is calibration: the readout names the gazetteer country.

**Part B (pairs).** The city tokens and R are swapped from y at layers ≥ B, for B ∈ {4, 6, 8, 9, 10, 11, 12, 14, 16, 20}. Two measures:
- **report:** the untrained self-report "Which country did you use to work out your answer?", city visible, over {y's country, x's country, two fillers};
- **readout:** a placeholder context fed the swapped run's city tokens and R at every layer (y's states at layers ≥ B, x's below). This shows which country the swapped computation makes available downstream.

**Part C (pairs).** R alone is swapped from y at every layer (v1's training family), and the untrained report is measured.

## 3. Hypotheses (predictions from the 12-city diagnosis)

- **H1, the bridge is held at the city tokens, not after them.** Both of the following:
  - calibration(city+R) − calibration(R) has a 95% bootstrap CI lower bound > 0.3;
  - calibration(R) − calibration(prior) has a 95% CI inside [−0.10, 0.10].
- **H2, the report and the causal readout read the same window.** All of the following:
  - item-level agreement between the report's pick and the readout's pick, over all pairs and boundaries, is ≥ 0.85;
  - both curves name the donor ≥ 0.8 at B = 4 and ≤ 0.2 at B = 20;
  - their transition boundaries (the first B where the donor rate is ≤ 0.2) are equal or adjacent in the grid.
- **H3, descriptive:** the untrained report names the donor ≤ 0.10 under the R-only swap.

## 4. Verdict (`code_v2/analyze_v2.py`)

| Outcome | Verdict |
|---|---|
| H1 and H2 | **SAME WINDOW** — this bed cannot separate reading the computation from reconstructing from the input, so the v1 training contrast cannot be built here |
| H1 only | **BRIDGE AT CITY TOKENS, but report and readout diverge** — that divergence would itself be a reason to revisit a training study at the divergent boundaries |
| otherwise | **NOT CONFIRMED** |

All CIs come from 10,000 bootstrap resamples, over items (A) and over pair × boundary rows (agreement).

## 5. Run

- One GPU job, `p9v2-premise`, through `tools/gpu_lock.py`, driven by `code_v2/run_v2.sh`. It waits for mains power below 30% battery.
- `code_v2/job_v2.py` refuses to run unless `PREDICTIONS_V2.sha256` verifies.
- The analysis runs on the CPU afterwards.
- Estimate: under 0.3 GPU-h.
- Changes after the freeze go in `DEVIATIONS.md`.

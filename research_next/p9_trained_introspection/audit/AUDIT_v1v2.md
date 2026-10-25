# Paper 9 audit: the v1 bed failure and v2 (independent recomputation)

**Who and when.** Independent auditor subagent (Claude, Opus 5.5), 2026-10-21, about 23:50 (+04:00).

**Scope.** v1 and v2 only, under AUDIT_BRIEF.md sections 1, 2 and 4. v3 was still running and is not covered.

**Method.**
- **Code.** My own code in `audit/`: `recompute_v1v2.py`, `extra_checks.py` and `explore_structure.py`. It uses numpy and the standard library only, imports no study module, and ran on the CPU.
- **Inputs.**
  - `res/screen.json`
  - `res_pilot/pilot.json`, `pilot_gates.json` and `config_pilot.json`
  - `res_selftest/gen.json` (split check only)
  - `res_v2/premise.json`
  - `res_v2/analysis_v2.json` was read for comparison only.
- **Output.** `audit/recomputed_v1v2.json`.
- **Bootstrap.** 10,000 percentile resamples with my own seed (90210; the study used 11).
- **Pick check.**
  - I re-derived every pick from the stored per-rotation letter log-probabilities, using the preregistered rotation rule. There were 0 mismatches over 918 Part A rows and 4,200 Part B/C rows.
  - The opposite rotation convention gives 491 and 2,056 mismatches, so the check is not vacuous.

## Verdicts

- **v1: bed failed. CONFIRMED.**
  - PG1 passes, PG2 and PG3 fail, and PG5 passes. PG2 fails under every definition of calibration and reliability I tried.
  - The chain stopped at `make_config.py` before any main data existed. `res/` holds only `screen.json`.
- **v2: SAME WINDOW. CONFIRMED.**
  - H1, H2 and H3 all hold on recomputation.
  - All 54 point estimates are identical to `analysis_v2.json`, and all 52 CI bounds are within 0.005. There are no mismatches beyond bootstrap noise.
  - The verdict is correct under the frozen rule. Several sentences in RESULTS.md go beyond what v2 measured (items 1–6 below).

## Freeze checks (run from the folder root)

| Check | Result |
|---|---|
| `python code\freeze.py check` | All 30 files OK ("all match", exit 0). There is no `PREDICTIONS.sha256.amend`, so no v1 file has been amended or changed since the 17:28 freeze. All 21 files in `code/` are covered. |
| `python code_v2\freeze_v2.py check` | All 15 files OK ("all match", exit 0). This covers all 5 files in `code_v2/`, the v1 engine files they import, and the paper 8 and S2 dependencies. |
| Hash prefixes | PREDICTIONS.md 4057de36… and PREDICTIONS_V2.md 544f4758… both match DEVIATIONS.md. |

## Recomputed against reported numbers

### v1 (`res_pilot/`, `res/screen.json`)

| Quantity | Recomputed | Reported | Status |
|---|---|---|---|
| PG1: 2-hop accuracy over usable cities (`ok`) | 0.9544 (962/1,008) | 0.954 | match; pass |
| PG1: usable test cities (test split, `acc2` true) | 363 | 363 | match; pass |
| Usable pilot cities / known-like pilot cities | 194 / 164 | 194 / 164 | match |
| PG2 calibration, frozen definition (an unreliable item counts as a miss) | 0.116 (19/164) | 0.116 | match; **fail** |
| PG2 calibration, readout pick = gazetteer, reliability ignored | 0.354 (58/164) | not reported | fail (item 9) |
| PG2 calibration among reliable known-like cities | 0.388 (19/49) | not reported | fail |
| PG2 reliability over all usable pilot cities | 0.309 (60/194) | 0.309 | match; **fail** |
| PG2 reliability over known-like cities only | 0.299 (49/164) | DEVIATIONS gives 0.309 for "164 known-like" | wording error (item 8) |
| `reliable` flag equals country/capital readout agreement | true for all 194 | – | consistent |
| PG3: l1 by the frozen rule | 0 (no B in {0, 4, 8, 12, 16} reaches 0.90) | 0 | match |
| PG3: transfer at l1 = 0 | 0.754 (49/65 reliable, out of 120 pairs) | 0.754 (n = 65) | match; **fail** |
| Transfer range over all 11 swap conditions | 0.732–0.784 | "0.73–0.78" | match |
| Untrained report under the swap from layer 0: donor / recipient | 0/120 / 0.858 | 0% / 86% | match |
| Other pilot parameters, re-derived by the frozen rules | ls 6; (la, s_add) (10, 4); lr 1e-4 | same | match |
| PG4 (for information): acc_diff at lr 1e-4 / 3e-4 | 0.464 / 0.486 | "0.46–0.49" | match |
| PG5: `res_selftest/analysis.json` exists | yes | pass | match |
| Contents of `res/` | `screen.json` only; no root `config.json` | "only screen.json" | match |
| Cities touched by the self-test | 133, all pilot split (0 train, 0 test) | "pilot cities" | match |

### v2 (`res_v2/premise.json`; gazetteer countries from `res/screen.json`)

| Quantity | Recomputed [95% CI] | Reported [95% CI] | Status |
|---|---|---|---|
| Usable test cities / known-like / pairs | 363 / 306 (identical list) / 200 | 363 / 306 / 200 | match |
| A: calibration of prior | 0.314 (96/306) [0.261, 0.366] | 0.314 [0.261, 0.366] | match |
| A: calibration of R | 0.353 (108/306) [0.301, 0.408] | 0.353 [0.301, 0.408] | match |
| A: calibration of city+R | 0.961 (294/306) [0.938, 0.980] | 0.961 [0.938, 0.980] | match |
| A: city+R − R | +0.608 [0.552, 0.663] | +0.608 [0.549, 0.660] | match (bootstrap noise) |
| A: R − prior | +0.039 [−0.007, 0.085] | +0.039 [−0.007, 0.085] | match |
| A: R pick = prior pick | 0.686 | 0.686 | match |
| B report: donor rate at B = 4, 6, 8, 9, 10, 11, 12, 14, 16, 20 | .965 .960 .935 .935 .905 .840 .025 .015 .015 .015 | JSON identical. RESULTS shows .96 .96 .94 .94 .91 .84 .03 .01 .01 .01 | match (rounding, item 13) |
| B readout: donor rate | .965 .955 .910 .900 .890 .795 .025 .020 .020 .020 | JSON identical. RESULTS shows .96 .95 .91 .90 .89 .80 .03 .02 .02 .02 | match |
| B report: recipient rate | .005 .005 .005 .005 .020 .065 .935 .950 .950 .950 | identical | match |
| B readout: recipient rate | .005 .005 .025 .025 .035 .125 .945 .945 .945 .945 | identical | match |
| B: donor-rate CIs (20 cells) | all within 0.005 of the reported bounds | – | match |
| Transition boundary (first B with donor rate ≤ 0.2) | report 12, readout 12 | 12, 12 | match |
| Item-level agreement over 2,000 pair × boundary rows | 0.9645 [0.956, 0.9725] | 0.9645 [0.956, 0.9725] | match |
| Agreement CI when whole pairs are resampled (robustness) | [0.951, 0.977] | not reported | – |
| Agreement expected if the two picks were independent within each B; Cohen's κ | 0.863; κ = 0.74 | not reported | item 5 |
| C (R-only swap), report: donor / recipient | 0.020 (4/200) / 0.945 (189/200) | 0.02 / 0.945 | match |
| H1 / H2 / H3 | true / true / true | true / true / true | match |
| Verdict | SAME WINDOW | SAME WINDOW | match |

**Design checks (v2).**
- **Recipients.** 200 distinct recipients, all usable test cities; 177 of them are known-like.
- **Donors.** 144 distinct known-like donors, each used at most 5 times.
- **Pair constraints.** Every pair has the same city token count and continent, different countries, and x ≠ y.
- **Option sets.** Every option set contains both countries and is identical across the pair's 21 rows.
- **H1 margin.** H1's second condition holds with a margin of 0.015 (about 5 of 306 items), and the result is identical across six bootstrap seeds.
- **Known-like recipients only.** Restricted to known-like recipients, the recipient rate at B ≥ 12 is 0.96–0.97.

## DEVIATIONS.md (task 4)

| Claim | Evidence | Status |
|---|---|---|
| The v1 bed check failed at 22:17, and the chain stopped before `stage_gen.py` | `logs/job_pilot.log` shows `make_config.py` rc 1 at 22:17:44 and "chunk pilot FAILED at make_config.py". `logs/run_chunks.log` shows end rc=1. `tools/gpu_queue.log` shows p9-pilot-1 from 22:02:12 to 22:17:44, rc=1. No p9 main job was ever queued. | confirmed |
| v2 was frozen at 22:40 | PREDICTIONS_V2.md was last modified 22:40:03 and PREDICTIONS_V2.sha256 written 22:40:10. The files in `code_v2/` were last modified 22:33:52–22:34:41, before the freeze. | confirmed |
| Refused early start: queued 22:33, given the GPU at 22:38, refused, no data | `logs/run_v2.log`: queued 22:33:59, end rc=1 at 22:38:59. `logs/job_v2.log` first line: "v2 not frozen … refusing to run". `tools/gpu_queue.log`: START 22:38:59, END rc=1 after 0 s. | confirmed |
| Re-queued at 22:40, after the freeze | run_v2.log and gpu_queue.log both show 22:40:16, six seconds after the hash file | confirmed |
| v2 data were produced only after the freeze | gpu_queue.log: 23:28:16–23:31:50. `premise.json` written 23:31:49; analysis at 23:31:52. | confirmed |
| Only pilot data were seen before the v2 freeze | See the list below. | confirmed |
| No frozen rule was changed after data were seen | Both freeze checks pass. There is no v1 amend file. v2 is a new preregistration, as v1 section 10 requires. | confirmed |

Evidence that only pilot data were seen before the v2 freeze:
- `pilot.json` covers exactly the 194 usable pilot cities.
- The self-test touched 133 cities, all pilot split.
- Both diagnosis scripts filter on `split == "pilot"`; their logs are dated 22:22 and 22:29.
- The SmolLM2 dry run is dated 22:39.
- `res/screen.json` (the 1-hop and 2-hop answers for every city, test cities included) also existed. PREDICTIONS_V2 section 1 discloses it.

The 22:17 entry has two wording problems (items 8 and 12).

## Errors and overclaims (most important first; none changes a verdict)

1. **Overclaim in RESULTS section 3, "Reading": "the country used to answer the two-hop question is read from the city tokens, at layers up to about 11."**
   - v2 never measures how the two-hop answer depends on the country. By design, donor and recipient share the continent, and `premise.json` does not even record the turn-1 answer.
   - Parts B and C measure only which country two turn-2 questions name.
   - *Fix:* "The country that later questions name is taken from the city-token states at layers ≤ 11. Swapping city+R from layer 11 upward moves both the report and the placeholder readout to the donor (0.84 and 0.795). Swapping from layer 12 upward moves neither (0.025)."

2. **Overclaim in RESULTS section 3: "Nothing after the city carries it."**
   - The data bound the effect but do not show it is zero. R − prior is +0.039, 95% CI [−0.007, +0.085], so a gain of up to about 8 points is not excluded.
   - Copying R changes the readout's pick on 31% of items (96/306). It does not raise calibration significantly: R is right where the prior is wrong on 32 items, and the reverse on 20 (exact McNemar p = 0.13).
   - Only one readout question was used, and no probe.
   - *Fix:* "R gives the placeholder readout no detectable country information (R − prior +0.04, 95% CI −0.01 to +0.08), and an R-only swap moves the report to the donor on 2% of rows."

3. **Overclaim in RESULTS section 3: "The untrained self-report reads the same window as the model's own downstream use. This task therefore offers no state that the computation uses but the report ignores…"**
   - "The model's own downstream use" is in fact the placeholder readout: the same model answering a different question ("Which country was the question about?") from the same copied states. It is not the two-hop computation's own use of the country.
   - "No state that the computation uses but the report ignores" goes beyond the frozen verdict. v2 tested two intervention families (city+R swaps at each layer boundary, and an R-only swap), one report wording, one readout question and one model.
   - *Fix:* Use the frozen verdict wording with its scope: "Under the frozen v2 rule (SAME WINDOW), this bed cannot separate reading the computation from reconstructing from the input in Qwen3-1.7B, so v1's training contrast cannot be built in this task."

4. **RESULTS misdescribes the instrument: v2 rests on a placeholder readout, and RESULTS does not say so.**
   - Sections 2 and 3 call it "the causal readout" and the "causal readout of the swapped run". It is the same placeholder instrument whose R-only form failed PG2.
   - Part A does validate its city+R form: 0.961 on 306 held-out known-like cities.
   - However, by the caveat in PREDICTIONS.md section 5, the readout cannot tell the city's identity from an extracted country. Below layer B it is also fed the recipient's own city-token states, which encode the input.
   - *Fix:* Call it the "placeholder readout (city+R states copied)", and add: "v2's conclusions rest on this readout. It shows which country a second question can recover from the copied states, not which country the two-hop computation used."

5. **The H2 agreement is reported without its chance level (RESULTS section 3).**
   - Suppose two readers' picks were independent within each boundary, but followed these two curves. They would be expected to agree on 0.863 of rows, which is already above the preregistered 0.85 bar. So this part of H2 adds little beyond the curve criteria.
   - Chance-corrected agreement is κ = 0.74 overall. Per boundary it ranges from 0.58 to 1.00, and it is 0.68 at B = 11.
   - The CI on rows treats the 2,000 rows as independent. Resampling whole pairs gives [0.951, 0.977].
   - The two readers are also not interchangeable item by item. At B = 4–11 the report names the donor without the readout on 40 rows, and the readout without the report on 15 (exact McNemar p = 0.001). The report follows the swap slightly more often.
   - The frozen verdict stands. *Fix:* Report the 0.863 chance level and κ next to 0.965, and mention the asymmetry.

6. **Exploratory diagnosis stated categorically (RESULTS section 2): "it follows a swap whenever the readout does, and only then."**
   - The section's own table contradicts this. At l = 12, 16 and 20 the readout names the donor 4/12, 3/12 and 2/12 times, while the report does 0/12 times. v2 has 71 discordant rows out of 2,000.
   - The paragraph also states the 12-city results as facts ("the country is held at the city tokens").
   - The "Conclusion" of DEVIATIONS.md's 22:17 entry is equally categorical: "the v1 intervention (swapping R) did not change the bridge". That rests on 12 cities, and the bridge itself was not measured.
   - *Fix:* "On 12 pairs, both switched between l = 8 and l = 12 (exploratory; confirmed at the curve level by v2)."

7. **RESULTS section 4 is incomplete and out of date.**
   - It presents candidate 1 (parity) as "a replacement task screened tonight". It concludes "This task cannot carry the training test in this model without a different answer format and a validated readout" from a 16-item exploratory screen.
   - TASK_SEARCH.md records seven candidates. Candidate 7 passed and became v3 (PREDICTIONS_V3.md, frozen 23:34).
   - Section 8 (Files) omits TASK_SEARCH.md, PREDICTIONS_V3.md and `code_v3/`.
   - The numbers in section 4 are correct against `diag/diag_arith.log`, checked row by row: parity 8/16; sum readout 1/16, 5/16 and 6/16; under the R-only swap the report named the recipient 8/16 and the donor 3/16.
   - *Fix:* Summarise all seven candidates and the gate each failed, point to v3, and write "failed a 16-item exploratory screen" instead of "cannot".

8. **Wording error in DEVIATIONS.md at 22:17: the reliability is attributed to the 164 known-like cities.**
   - The 0.309 is over all 194 usable pilot cities (60/194). Over the 164 known-like cities it is 0.299 (49/164).
   - PREDICTIONS.md section 10 says PG2 is computed "on pilot known-like items", but the frozen `stage_pilot.py` computes reliability over all usable pilot items. This text/code discrepancy predates any data.
   - It does not matter for the verdict: PG2 fails under every definition.
   - *Fix:* Add a correction note to DEVIATIONS.md, and in RESULTS section 1 write "0.309 (60/194 usable pilot cities; 0.299 on the 164 known-like)".

9. **An undefined calibration makes v1 and v2 look inconsistent (RESULTS sections 1 and 3; TASK_SEARCH.md row 0).**
   - v1's 0.116 counts unreliable known-like items as misses (frozen code).
   - On v2's definition (pick = gazetteer, no reliability filter), the same pilot data give 0.354 (58/164). That is almost identical to v2's calibration of R on test cities (0.353).
   - *Fix:* Footnote the definition, and give 0.354 next to 0.116.

10. **Overstated in RESULTS section 1: "So the readout was not reading R at all."**
    - As in item 2, an effect of up to about 8 points is not excluded.
    - The flat 0.73–0.78 "transfer" also needs an explanation. A likely one, not tested here: the 19 known donors were selected because their placeholder readout already named their own country. Seven of the 19 are in India or Mexico, which are among the readout's most frequent picks. A readout that ignores R would then still "transfer" to these donors at a similar rate, whatever layers were swapped.
    - *Fix:* Write "showed no detectable dependence on R", and add this explanation, labelled as likely.

11. **Minor (RESULTS section 1 table caption): "(res_pilot/pilot_gates.json; pilot cities only)".** PG1 is computed over all 1,008 usable screened cities and counts test cities. *Fix:* "PG2–PG3 on pilot cities; PG1 on the full item screen."

12. **Minor (DEVIATIONS.md at 22:17): "No train/test readouts, no reporter training and no evaluation exist."**
    - Reporters were in fact trained and evaluated, on pilot cities only: in the learning-rate check (arm I, 2 × 150 steps) and in the self-test (four arms).
    - I verified that all 133 self-test cities are in the pilot split.
    - RESULTS has the accurate wording: "No reporter was trained or evaluated on test cities." *Fix:* Use that wording in an appended note.

13. **Cosmetic (RESULTS section 3, Part B table).**
    - Rates out of 200 are rounded inconsistently: 0.965 → 0.96 but 0.935 → 0.94, and 0.025 → 0.03 but 0.015 → 0.01.
    - These are floating-point artefacts, all within ±0.005. *Fix:* Print three decimals.

14. **Minor (RESULTS section 2 and DEVIATIONS.md): "letters carry essentially all the next-token mass".**
    - D1 checks a single city (`known[0]`) under five prompts.
    - The stronger evidence that the format is not the problem is D2's country-name column, which shows the same pattern (4/12 against 11/12). *Fix:* Say "(one city)" and cite the name column.

15. **Status: RESULTS.md is not final.**
    - It still contains the template placeholders `{{SUMMARY}}` and `{{MEANING}}`. These plain-language parts are where overclaims are most likely, so they must be audited before release.
    - Sections 1–4 quote no chat messages from the author and make no general claim about "introspection". Their claims are scoped to Qwen3-1.7B and this task, apart from items 1–3.

## Other observations (not errors in v1 or v2)

- **s_kl.** s_kl = 2.0 is the top of its grid. Random directions at 2·b reached a median R-KL of 0.49, against 2.76 for the swap, so T4-kl would not have been KL-matched if the chain had continued. This is moot because the bed failed.
- **For the v3 audit: a forward-dated entry.**
  - The last entry in DEVIATIONS.md and the lines in `PREDICTIONS_V3.sha256.amend` are labelled 23:41.
  - Both files were last written at 23:36:19–23:36:28, while the audit clock read 23:46. The entry appears to be forward-dated by about five minutes.
  - The v3 job had been queued at 23:35:05, before that amendment.

## Files

- `audit/recompute_v1v2.py`: the recomputation and the comparison against `analysis_v2.json`.
- `audit/extra_checks.py`: the McNemar tests and the composition of the v1 known-donor pool.
- `audit/explore_structure.py`: a structure dump of the raw JSON files.
- `audit/recomputed_v1v2.json`: all recomputed numbers.

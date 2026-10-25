# Paper 9 task search, 2026-10-21 (exploratory screens; Qwen3-1.7B, thinking off; CPU at most 4 threads)

**Goal.** Replace the two-hop bed with a task whose intermediate is computed after the input tokens. The candidate must pass v1's gates: task accuracy; a causal ground truth that is above chance, with readouts that agree; and an answer-fixed swap rate. Each screen used ad-hoc items, separate from any later analysis set. The scripts and logs are in `diag/`.

| # | Candidate (turn 1) | Intermediate | Result | Gate failed |
|---|---|---|---|---|
| 0 | Two-hop "Which continent is the country containing {city} in?" (v1) | Bridge country after the city | v1 pilot: readout calibration 0.12. Diagnosis: no country readable after the city (3/12 vs 12/12). v2: SAME WINDOW (306 cities and 200 pairs) | PG2 |
| 1 | "What is {a} + {b}? Reply only with whether the sum is even or odd." (single digits) | Sum | Answered "Even" to everything (8/16). The sum barely readable after the operands (5/16 vs prior 1/16) | PG1, PG2 |
| 2 | "Compute: {a} + {b} = ? Is the result greater than {c}? Answer yes or no." | Sum | Always "Yes" (3/6 on the first items; 21/40 by forced choice) | PG1 |
| 3 | First digit, last digit or tens of {a} + {b}, forced choice | Sum | 10/40, 2/40 and 11/40; the model copies an operand digit | PG1 |
| 4 | Above or below 50 | Sum | 29/40, borderline; the exact sum is likely not used | not pursued |
| 5 | Carry route ("Compute: {a} + {b}." with prefill "{a} + {b} = ") | Whether a one was carried | Sums 24/24 right, but "did this addition require carrying?" got 12/24 (always "No") even with the operands visible | PG2 |
| 6 | Paper 8 multiplication ("What is {a} * {b}?" with prefill "{a} * {b} = "), least-sure digit | Per-digit confidence | Degenerate: the least-sure digit was always the second (20/20); no swap can move it | PG2 |
| **7** | **Same multiplication; confidence in the second digit (quartile bins of its log-probability)** | **The model's own probability of its second digit, at the first answer position** | **Accuracy 41/48. The probability and entropy labels agree 47/48. Same-product swaps of the answer positions move the label to the donor's 10/10 (layers 0, 8, 14, 21; digits kept 27/32 including donors that were wrong). The untrained report said "sure" 24/24** | **passed (23:26)** |

**Why 7 qualifies.** The confidence in the second digit is produced at the answer positions, after the operands, so swapping those positions leaves the input untouched. The ground truth is the computation's own output probability, not a lens. The answer, all four digits, stays fixed under same-product swaps.

**Caveats.**
- This is a confidence (metacognitive) intermediate, not a value like the bridge country.
- A paraphrase of the same question may predict it closely, which would make T3 uninformative under G5.

Candidate 7 was preregistered as v3 (`PREDICTIONS_V3.md`).

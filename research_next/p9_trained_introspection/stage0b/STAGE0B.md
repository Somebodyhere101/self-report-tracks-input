# Stage 0b: is the swapped-in confidence readable at the new report positions, and does a reporter learn it?

This is an exploratory feasibility check for paper 9 v4. It was not preregistered and was run before the v4 freeze. It used the base model Qwen3-1.7B on 2026-10-22, on **pilot products only**. It follows Stage 0a (`stage0/STAGE0.md`, run by the coordinating session) and the coordinator's decision to make the report immediate.

**Setup common to all parts:**
- Turn 1 ends with the four answer digits.
- Region R is the four digit positions.
- Swaps copy the donor's block outputs at R from layer 16.
- Labels are two bins at the train-median lp2 (−0.001326).

## 1. Probe check at the new report positions (`s0b_extract.py`; analysis by Stage 0a's `s0a_probe.py`, unchanged)

**Rows.** Stage 0a's rows: 590 natural pilot items and 474 same-product swaps (256 with a different two-bin label).
- Swapped runs reproduce the donor's lp2 exactly (max |diff| = 0).
- Digits are kept in 100% of swapped rows.

**Report positions** (option order 0):
- **imm:** the last token of the immediate tail, 27 tokens after the digits, in the same assistant message;
- **del:** the last token of the delayed turn-2 tail, 45 tokens after the digits.

**B2.** A probe trained on natural rows is tested on held-out products' swapped rows whose label differs. The table gives its agreement with the donor bin, then with the recipient bin.

| Position | Layer 8 | Layer 12 | Layer 16 | Layer 20 | Layer 24 | Layer 27 |
|---|---|---|---|---|---|---|
| imm | .35 / .65 | .36 / .64 | .37 / .63 | .40 / .60 | .37 / .63 | .35 / .65 |
| del | .38 / .62 | .39 / .61 | .38 / .62 | .41 / .59 | .39 / .61 | .36 / .64 |
| R0 (first digit, inside R) | .34 / .66 | .38 / .62 | .64 / .36 | .66 / .34 | .61 / .39 | .66 / .34 |

**Reading.**
- The coordinator's bar (donor > 0.60) **fails at both report positions, at every layer.** Both positions carry the recipient's input-implied confidence, as Stage 0a found for v3's turn-2 report.
- Moving the report into the answer, 27 tokens after the digits, does not change this.
- The information is present at R from layer 16 on (0.61–0.66 donor). With four bins, both report positions sit at the majority baseline.
- The untrained model's immediate report always answers A ("Very sure" in order 0); its delayed report answers A 28% of the time.

## 2. Tiny training runs (`s0b_train.py`)

**Data.**
- Pilot **half A only**, v4's seeded halves, so half B stays untouched for v4's frozen learnability gate.
- Half A is split again by product: A1 has 190 items (training) and A2 has 124 items (held out).
- A2 rows: 24 different-bin swaps, 62 same-bin swaps and 124 natural items.
- Input-only predictors on A2's natural items:
  - para: 0.871;
  - recal_para: 0.895;
  - learned: 0.895;
  - majority: 0.573.

**Training.** v4's trainer: two-bin labels, rank r LoRA on the report tail only, Adam with warm-up, batch 8, one random option order per example.

**Scores on A2.** "Balanced" is the mean of the different-bin and same-bin swap accuracies; chance is 0.5, and a reporter that ignores the swap also scores about 0.5. "Very sure" share is the share of natural items answered "Very sure", with scores averaged over both option orders. Values near 0.00 or 1.00 mean a constant answer.

| Run | Steps | Different-bin swaps | Same-bin swaps | Balanced | Natural | "Very sure" share (natural) | Final loss |
|---|---|---|---|---|---|---|---|
| untrained, imm | 0 | .625 | .387 | .506 | .347 | .53 | |
| untrained, del | 0 | .667 | .306 | .487 | .419 | .96 | |
| I, imm, r8, lr 1e-3 | 150 / 300 | .375 / .583 | .645 / .419 | .510 / .501 | .556 / .452 | .03 / .93 | .665 / .707 |
| I, imm, r8, lr 1e-4 | 150 / 300 | .375 / .625 | .661 / .323 | .518 / .474 | .573 / .427 | .00 / .97 | .623 / .699 |
| I, del, r8, lr 1e-3 | 150 / 300 | .458 / .625 | .468 / .371 | .463 / .498 | .532 / .435 | .43 / .88 | .683 / .781 |
| I, del, r8, lr 1e-4 | 150 / 300 | .375 / .625 | .661 / .339 | .518 / .482 | .573 / .427 | .00 / 1.00 | .649 / .697 |
| O, imm, r8, lr 1e-3 | 150 / 300 | .375 / .375 | .548 / .661 | .462 / .518 | .516 / .573 | .31 / .00 | .580 / .735 |
| I, imm, r32, lr 3e-4 | 300 / 600 / 900 | .583 / .542 / .583 | .419 / .435 / .355 | .501 / .489 / .469 | .460 / .452 / .419 | .90 / .67 / .99 | .764 / .723 / .669 |
| O, imm, r32, lr 3e-4 | 300 / 600 / 900 | .375 / .375 / .500 | .661 / .661 / .452 | .518 / .518 / .476 | .573 / .573 / .500 | .00 / .00 / .57 | .740 / .694 / .700 |

**Reading.** No reporter learns anything: within 300 steps at rank 8 in either format at either learning rate, nor within 900 steps at rank 32.
- 12 of 16 checkpoints answer almost constantly (the same answer on at least 87% of natural items); the rest mix the two answers at about chance.
- Training losses settle near ln 2 = 0.693 within 50–150 steps.
- Balanced swap accuracy is at chance.
- O's natural accuracy (0.573) equals the majority answer, against 0.895 for the question-only predictors.

This repeats v3's failure with two bins and both report formats.

## 3. Where along the immediate tail does the information fade? (`s0b_tailpos.py`)

The same natural-trained probe (B2; 256 swaps with a differing two-bin label) was run at the last answer digit and at each of the 27 positions of the immediate tail (`tailpos.json`, `tailpos.log`; GPU 0.5 min, CPU probe 2.1 min). Donor agreement:
- **Last digit (inside R):** 0.63–0.66 at layers 16–27; 0.36 at layer 12, before the swap.
- **First token after the digits:** 0.33–0.36 at layers 16–27.
- **First three positions:** 0.30–0.39.
- **All 27 tail positions:** 0.30–0.42 at layers 16–27, and 0.32–0.43 at layer 12.

**Reading.** The swapped-in confidence is lost at the very first token after the digits, even though, from block 17 on, those positions attend to the donor's digit states. The coordinator's rule asked for > 0.60 at one of the first three positions to propose a one-token tag. It was not met, so the trained-report line stops.

## 4. What this means for v4

- **The coordinator's pre-freeze probe condition for the immediate format is not met.** Under the coordinator's rule (Stage 0 message), the v4 budget ladder is therefore **not started**: v4 is not frozen, and no ladder or main data exist.
- **Both report positions follow the question, not the swapped state.** In the untrained model, the state 27 or 45 tokens after the digits follows the confidence the question implies, not the swapped-in one (linear probes). The design cannot tell recomputation from the question from reading the unswapped lower layers of the answer positions.
- **Short training does not open a route to the digits.** Up to 900 steps at rank 32, the reporters stay at chance, mostly with near-constant answers, and the ordinary reporter does not reach even the majority level of the question-only predictors.
- **Still open:** whether much longer training (the ladder's 3,000 steps) would open a route. Section 3 shows that no position after the digits carries the swapped-in value to begin with.

## Caveats

1. **Exploratory and pilot-only.** The A2 held-out set is small (24 different-bin rows); the runs are short and use one seed.
2. **The probes test the untrained model's state at the report position.** They do not test what a trained adapter can learn; the training runs do that, but only up to the budgets listed.
3. **The cached turn-1 trainer suggested by the coordinator was not built.** v4's tested trainer runs 300 steps in about one minute here.

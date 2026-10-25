# Stage 0a: can the swapped-in confidence be read linearly at the report position?

Exploratory feasibility check for paper 9 v4. This was not preregistered. Run on 2026-10-22, base Qwen3-1.7B, no adapter, PILOT products only.

- Scripts: `s0a_extract.py` (GPU, run through the gpu_lock, 0.5 min) and `s0a_probe.py` (CPU).
- Outputs: `rows.json` and `probe.json`.
- Features (2 positions x 6 layers x 1064 rows, fp16) are stored outside P9 at `C:\Users\<user>\AppData\Local\Temp\p6\stage0\feats.pt`.

## Setup

**Rows.**
- Natural rows: all 590 correct pilot items (374 products).
- Swapped rows: v3 `unit_swap`, which copies R at layers 16 to 27 from a same-product donor. There are 474 ordered pairs, which is the whole pilot pool of donors whose 4-bin differs from the recipient's. Of these, 256 also differ in their 2-bin.

**Swap check.** After the swap, the recipient's measured lp2 equals the donor's lp2 exactly (max |diff| = 0). The four answer digits are kept in 100% of swapped rows.

**Bins.**
- 2-bin: split at the median lp2 of correct TRAIN items, -0.001326. This is res_v3/gen.json th[1], and I re-checked it as the median of the 1111 train items in screen.json. Only lp2 values were read; no test items were touched.
- 4-bin: the v3 train quartiles.

**Turn 2.** The v4 two-option report prompt, rotation 0 (A) Very sure, B) Less sure), run on the turn-1 cache.

**Positions.**
- rep: the last prompt token "(", where the letter is predicted.
- R0: the first answer-digit position, where lp2 is computed. This is the ceiling.

**Probes.** L2 multinomial logistic regression on standardised features, fitted with torch LBFGS. sklearn's compiled extensions are blocked by Windows Application Control on this machine. The regularisation strength was picked on natural rows from the grid {0.5, 0.05, 0.005}. Cross-validation is 5-fold grouped by product. The CI is a 95% cluster bootstrap over products.

**Analyses.**
- A: all rows; the label is the effective bin (the donor's bin for swapped rows).
- B1: swapped rows only, where the donor bin differs from the recipient's own bin. The label is the donor bin.
- B2: a probe trained on natural rows only, tested on the B1 rows of held-out products. I report how often it agrees with the donor bin and how often with the recipient's bin.
- C (input-only probe): the same rows and labels as B1, but the features are the recipient's UNSWAPPED features.

**Baselines.**
- Chance: 0.50 / 0.25.
- Majority: 0.53 for A and 0.50 for B with 2 bins; 0.30 for A and 0.31 for B with 4 bins.
- Input-only label baseline: the recipient's own bin equals the label in 0.00 of B rows, by construction.

## Results (accuracy, 95% CI in brackets for B1)

### 2 bins

| pos | layer | A | B1 swapped-only | B2: agrees with donor / recipient | C input-only probe |
|---|---|---|---|---|---|
| rep | 8  | .61 | .74 [.66,.81] | .38 / .62 | .70 |
| rep | 12 | .63 | .71 [.62,.80] | .39 / .61 | .77 |
| rep | 16 | .60 | .71 [.61,.79] | .36 / .64 | .77 |
| rep | 20 | .64 | .68 [.59,.76] | .40 / .60 | .69 |
| rep | 24 | .65 | .71 [.63,.78] | .38 / .62 | .71 |
| rep | 27 | .64 | .71 [.63,.79] | .40 / .60 | .69 |
| R0 | 8  | .64 | .65 | .35 / .65 | .64 |
| R0 | 12 | .64 | .70 | .37 / .63 | .70 |
| R0 | 16 | .71 | .68 | .64 / .36 | .67 |
| R0 | 20 | .69 | .66 | .66 / .34 | .67 |
| R0 | 24 | .69 | .67 | .61 / .39 | .68 |
| R0 | 27 | .74 | .71 | .65 / .35 | .72 |

### 4 bins

| pos | layer | A | B1 swapped-only | B2: agrees with donor / recipient | C input-only probe |
|---|---|---|---|---|---|
| rep | 8  | .33 | .29 | .31 / .34 | .31 |
| rep | 12 | .33 | .34 | .30 / .36 | .31 |
| rep | 16 | .30 | .30 | .28 / .32 | .31 |
| rep | 20 | .34 | .31 | .26 / .34 | .32 |
| rep | 24 | .33 | .32 | .27 / .35 | .32 |
| rep | 27 | .33 | .33 [.28,.38] | .28 / .35 | .31 |
| R0 | 8  | .34 | .30 | .27 / .41 | .30 |
| R0 | 12 | .34 | .33 | .27 / .37 | .31 |
| R0 | 16 | .40 | .35 | .42 / .26 | .30 |
| R0 | 20 | .40 | .32 | .40 / .25 | .34 |
| R0 | 24 | .38 | .41 | .43 / .24 | .35 |
| R0 | 27 | .44 | .47 [.40,.53] | .41 / .27 | .32 |

The base model's own report has no signal. It picks A 26% of the time, with accuracy 0.47 on natural rows and 0.50 on B rows.

## Verdict

**The swapped-in confidence is not linearly readable at the report position, in any of the six layers.**

**2 bins.** The swapped-only B1 probe passes the nominal 0.60 bar at the report position (about 0.71). However, the input-only probe on the recipient's unswapped features does just as well (0.69 to 0.77). With 2 bins the label is always 1 minus the recipient's own bin, so B1 is explained by reading the recipient's input-determined confidence and inverting it. The decisive test is B2, the natural-trained probe. At the report position it agrees with the RECIPIENT on 60 to 64% of swapped rows and with the donor on only 36 to 40%. What the report token carries is the input's confidence, not the swapped-in state.

**4 bins.** Everything at the report position is at the majority baseline (about 0.30 to 0.34), including natural rows.

**Ceiling at R0.** The ceiling behaves as it should. From layer 16 on, the natural-trained probe follows the donor (2 bins: 0.61 to 0.66; 4 bins: 0.40 to 0.43). Before the swap layer (layers 8 and 12) it follows the recipient. But the ceiling is itself weak: at layer 27, where lp2 is literally computed, 2-bin A is 0.74 and 4-bin A is 0.44. So the information exists at R but was not moved to the report token. Even at R, it is only modestly linear.

## Caveats

1. **This tests the base model's turn-2 state, not what a trained reporter could learn.** A LoRA reporter changes the turn-2 computation and can learn to attend to R, whose K/V at layers 17 and up are the donor's. Stage 0a shows the information is not already sitting at the report token. It does not show that the information is unlearnable.
2. **One report tail only.** I used rotation 0 of the v4 prompt; there is no rotation averaging. Other option orders could shift the report-token state.
3. **Small samples, wide CIs.** Pilot only: 590 natural rows, 256 swapped rows for 2 bins and 474 for 4 bins. CIs are about ±0.08 on B rows.
4. **The 2-bin B1 test is confounded by construction**, because the label equals 1 minus the recipient's bin. B2 and C are the interpretable comparisons.
5. **Regularisation was chosen on natural rows**, which also appear in A's test folds, so A is slightly optimistic. The strongest value (0.5) was picked almost everywhere, which is the edge of the grid. Stronger regularisation was not tried.
6. **Different probe implementation than requested.** sklearn could not be loaded (Application Control), so this is a torch LBFGS logistic regression with my own size-balanced group folds.
7. **Labels are near saturation and numerically noisy.** The quartile cut points span lp2 from -0.018 to -7e-5, so the confidence differences are tiny. bf16 batch-shape effects move some items' lp2 by up to 0.16 nats compared with screen.json, and the measured bin disagrees with the label bin on 0.8% (2 bins) and 2.5% (4 bins) of rows. Labels come from screen.json lp2, as in v3.
8. **The R0 ceiling at layers 16 and up is the donor's own copied state**, so it equals natural readability at R rather than a test of transfer.
9. **Linear probes only.** Nonlinear readability was not tested.
10. **Exploratory and pilot-only.** Nothing here touches the train/test split or the preregistered files.

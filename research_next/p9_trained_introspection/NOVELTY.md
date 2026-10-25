# Paper 9 novelty check (2026-10-21)

The full evidence (citation table, four-criteria table for 40+ papers, search log) is in `novelty_work/NOVELTY_REPORT.md`. A subagent wrote it from web pages only. I re-opened the two items that narrow the claim most: arXiv 2610.17495 and OpenReview 5P7dVpGNPO. Both exist and match the description.

## Verdict in plain words

The core study has not been done. I found nothing up to 2026-10-21 that does all of the following:
- trains a model's own self-report on labels from an activation change that keeps the input and the answer fixed;
- applies that change to a step of a task (here, which country a two-hop question goes through);
- then checks the trained report on ordinary, unpatched questions against a causal measure of that step and against a predictor that sees only the input.

Two pieces have recently been done on their own, so the claim must be narrower than `research_next/P9_PRIOR_ART.md` says:

1. **Interventional versus ordinary training has been compared**, but for low-level signals. An anonymous ACL ARR October 2026 submission, "Interventional Training Reduces Causal Bypass in In-Context Introspection" (OpenReview 5P7dVpGNPO, posted 2026-10-13), trains models to report single neuron values. Training on counterfactually perturbed values beat output-only training, including on natural signals. It covers no task intermediate, holds nothing fixed and, from its abstract, uses no input-only predictor.
2. **A model reading its own running computation in place** has been done with natural labels: Geiping, "A Sense of Proprioception in Language Models", arXiv:2610.17495 (12 Oct 2026). The same weights report routing, entropy and logit-lens values, against a token-lookup baseline. There is no interventional training and no task intermediate.

## The narrow claim paper 9 can make

To our knowledge, as of 2026-10-21, no work trains a model's self-report of a **task intermediate** on **answer-fixed interventional labels** and then tests it on the model's **natural computation** against an **input-only predictor**, **head to head** with observational, injection-trained and untrained reporters of the same model under one budget. Observational and interventional training have been compared only for low-level internal signals (5P7dVpGNPO).

Not claimed as new:
- in-place self-reading (Geiping 2610.17495; also IFT and 5P7dVpGNPO);
- the input-only or third-party bar (Song et al. 2508.14802; Singh et al. 2605.26242; Ferrara 2608.20569; the ICLR 2027 submission GqNWne0GxX; Li et al.'s "– activation" ablation);
- KL- or impact-matched random perturbations (Ferrara 2608.20569);
- interventional training as such (5P7dVpGNPO).

## Corrections to `research_next/P9_PRIOR_ART.md` (not edited; recorded here)

- **Cheah et al. 2607.21090** trains self-explanation faithfulness with RL on edits to the input *text*, not detection of activation interventions. Its cross-type transfer was *weak and inconsistent*, not absent.
- **Singh, Linzen & Ravfogel 2605.26242** does not suggest interventional training. It warns (App. C.5) that fine-tuning may install a task-specific mechanism. That warning is a risk for paper 9 and is cited as one.
- **Pepper et al. 2602.10352**:
  - The author list omits Graziano.
  - The 91% bridge-naming figure is pooled over layers and positions; the best single layer gives 73%.
  - Its adapter reads the vector in a separate prompt.
- **Aoki et al. 2609.00904**: in the table, its control targets *are* independent of the input.
- **Duplicate entry:** the introspection-adapters blog post and arXiv 2604.16812 are the same work.
- **Human precedent:** "no human study trains self-report on decoded labels" is wrong. Frederick (2012, Consciousness and Cognition 21:1345) trained people to report EEG alpha state with feedback, with labels from natural fluctuations. That is the human analogue of paper 9's observational arm. I found no human study that trains report on *induced* states.
- **Bhan et al. 2506.09277** (48–65% bridge decoding) measures decodability with Patchscopes. It does not show that natural bridges differ from the input-implied one. Paper 9 therefore measures natural divergence itself and gates the decisive test on it.

## What this changes in the design (my decisions)

- **Arm X, "external observer", added.** X is the same Qwen3-1.7B with the same LoRA budget, trained on the same observational examples as arm O. It reads the conversation as plain text in a fresh chat, so it cannot read the original computation. It is the strongest input-only predictor in the T3 bar, following GqNWne0GxX's external reporters and Binder et al.'s cross-prediction.
- **Pre-registered meaning of an O ≈ I result**, in PREDICTIONS.md. Geiping's observationally trained reporters predicted routing edits above chance, so O may come close to I.
- **Natural-variation gate kept.** The causal readout is new to this study, so how often the natural bridge diverges from what the input implies is measured, and T3 is decisive only if there is enough divergence.
- **Concurrent-work risk:** a SPAR Fall 2026 project (mentor Belinda Li) plans to train verbal reports on patching and ablation ground truth. Re-check it and any arXiv version of 5P7dVpGNPO before any write-up.

## Must-cite list for a write-up

5P7dVpGNPO (ARR Oct 2026); Geiping 2610.17495; Li et al. 2511.08579; Hahami et al. 2607.14111 (IFT); Fonseca Rivera & Africa 2511.21399; Pepper et al. 2602.10352; Plunkett et al. 2505.17120; Singh et al. 2605.26242; Cheah et al. 2607.21090 (as corrected); Ferrara 2608.20569; GqNWne0GxX (ICLR 2027 submission); Karvonen et al. 2608.16747; Nguyen et al. 2606.23671 (trained introspection did not transfer to a tampering probe); Morris & Plunkett, "causal bypassing" (LessWrong, Nov 2025); Bhan et al. 2506.09277; Biran et al. 2406.12775; Song et al. 2508.14802; Shibata et al. 2011; Cortese et al. 2016; Frederick 2012.

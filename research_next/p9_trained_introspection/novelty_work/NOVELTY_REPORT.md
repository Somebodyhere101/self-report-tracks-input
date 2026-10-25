# Paper 9 novelty and citation check: training a model's self-report on answer-fixed interventions

Checked 2026-10-21. I worked only from web pages: arXiv abstract, HTML and listing pages, the arXiv API, the OpenReview search API, DOI, Crossref and Europe PMC records, LessWrong and Alignment Forum posts, and lab or program pages. I downloaded no PDFs and opened none. I did not touch any folder named `sealed` or anything under `p6\replayA` or `p6\replayB`. Titles, authors and dates below are exact as shown on the pages. The page reader would not quote abstracts word for word, so abstracts are paraphrased and only short phrases are quoted.

---

## (A) Verdict

I found no paper, preprint, OpenReview submission or blog post, up to 2026-10-21, that does all four parts together. That means training a model's own self-report on labels from answer-fixed interventions on a task intermediate, then scoring the report on unpatched items against a causal measure of the natural intermediate and against an input-only predictor. However, the sentence "nobody has compared training on ordinary examples with training on intervened ones" is no longer true in general. An anonymous ACL ARR October 2026 submission, "Interventional Training Reduces Causal Bypass in In-Context Introspection" ([OpenReview 5P7dVpGNPO](https://openreview.net/forum?id=5P7dVpGNPO), posted 2026-10-13), compares output-only fine-tuning with training on counterfactually perturbed values. Its signals are low-level, such as one neuron's activation value, and it reports better introspection of both natural (endogenous) and injected signals. In-place self-reading of running computation (part 1) has also been done on its own, most closely by Geiping, [arXiv:2610.17495](https://arxiv.org/abs/2610.17495) (12 Oct 2026). That paper trains a model to report its own routing, entropy and logit-lens values, compares it with a token-lookup baseline, and tests it on routing edits. Paper 9 is still new only in its narrow form: the first head-to-head comparison of label sources (answer-fixed interventional, observational, injection, none) for a task intermediate, judged on natural computation against an input-only predictor. Several citations in P9_PRIOR_ART.md need fixing:
- Cheah et al. is overstated: cross-type transfer was weak and inconsistent, not absent.
- Singh et al. does not suggest interventional training.
- The author list given for Pepper et al. leaves out Graziano.
- The line that no human study trains self-report on decoded labels is contradicted by EEG alpha-state discrimination training (Frederick 2012, replicating Kamiya 1962).

---

## (B) Citation verification (every paper cited in P9_PRIOR_ART.md)

Status codes:
- **VERIFIED**: the page exists and the claim matches it.
- **VERIFIED\***: verified, with a caveat that should go into the text.
- **PARTLY**: part of the claim is unsupported.
- **MISMATCH**: the claim is wrong or overstated.

| # | Cited as | URL | Exact title | First author et al. | Date (v1; latest) | Status | Note |
|---|---|---|---|---|---|---|---|
| 1 | Li et al. 2511.08579 | https://arxiv.org/abs/2511.08579 | Training Language Models to Explain Their Own Computations | Belinda Z. Li, Guo, Huang, Steinhardt, Andreas | 11 Nov 2025; v3 9 Feb 2026 | VERIFIED\* | **Label:** whether the output changes after the patch, plus what the new output is, so the answer is not held fixed. **How the explainer sees the patch:** the vector goes into a `[s]v[e]` placeholder in the explainer's own text prompt. That prompt names the layer, the token and the original text, and asks how the output would change. That is a separate forward pass, not an in-place read. **Ablation:** the "– activation" row of Table 5 (App. D.2) gives exact match **64.0±0.4 → 59.9±0.4** (Qwen3-8B explaining Qwen3-8B); for Llama-3.1-8B it is 48.6 → 45.2. **Held-out intervention type for patching:** none. **Caveat:** "nothing tested on natural computation" holds for the patching task only. The feature-description task does use unpatched FineWeb activations. |
| 2 | Hahami et al. (IFT) 2607.14111 | https://arxiv.org/abs/2607.14111 | Introspection Fine-Tuning (IFT): Training Small LLMs to Introspect | Ely Hahami, Sinha, Jain | v1 8 May 2026 (ID is 2607, i.e. announced July 2026) | VERIFIED | **Injections:** steering vectors (Gaussian, semantic, or semantic plus concept name) added to one of 10 sentences in the same prompt. **Label:** the index of the steered sentence; the Sem+Rsn variant adds the concept name. **Main result:** Llama-1B localization 9.6% → 60.6%; transfers zero-shot to strength comparison (30.2% → 52.2%). **Missing:** no test on natural computation or task intermediates, and no input-only baseline. Both statements in the doc hold. |
| 3 | Fonseca Rivera & Africa 2511.21399 | https://arxiv.org/abs/2511.21399 | Steering Awareness: Detecting Activation Steering from Within | Joshua Fonseca Rivera, David Demitri Africa | 26 Nov 2025; v3 19 Mar 2026 | VERIFIED | **Setup:** CAA concept vectors injected at about two-thirds depth on the last prompt token; LoRA trained to report whether an injection happened and to name the concept. **Results:** best model 95.5% detection and 71.2% identification, with 0 false positives. **Transfer:** only to vectors built in geometrically similar ways (PCA/ICA/LDA vectors fail). **History:** v1 was single-author, titled "Training Introspective Behavior: Fine-Tuning Induces Reliable Internal State Detection in a 7B Model". Accepted at COLM 2026 (OpenReview decision 2026-07-08). |
| 4 | Pepper et al. 2602.10352 | https://arxiv.org/abs/2602.10352 | Learning Self-Interpretation from Interpretability Artifacts: Training Lightweight Adapters on Vector-Label Pairs | Keenan Pepper, McKenzie, Pop, Servaes, Leitgab, Vaiana, Rosenblatt, **Graziano**, de Lucena | 10 Feb 2026; v2 2 Jun 2026 | VERIFIED\* | **Result:** 455/500 (**91.0%±1.3**) against 282/500 (56.4%) untrained. **Caveat:** the 91.0% is pooled over all layers and token positions, with 10 samples per cell; the best single layer gives **73.0%**. **Setup:** Llama-3.1-8B-Instruct; prompts filtered to those answered correctly without CoT; the vector enters a placeholder token in a separate SelfIE-style prompt. **Training labels:** observational (SAE auto-interp labels and contrastive vectors). **Baselines:** no input-only baseline. **Table fix:** the comparison-table author list leaves out Michael S. A. Graziano, who is listed on both v1 and v2. |
| 5 | Plunkett et al. 2505.17120 | https://arxiv.org/abs/2505.17120 | Self-Interpretability: LLMs Can Describe Complex Internal Processes that Drive Their Decisions | Dillon Plunkett, Morris, Reddy, Morales | 21 May 2025; v2 10 Nov 2025 | VERIFIED | On native preferences, r rose from .46 to .71 for GPT-4o and from .40 to .70 for GPT-4o-mini (Experiment 3). Training labels were the instilled weights. There is no input-only or third-party predictor for native preferences. |
| 6 | Singh, Linzen & Ravfogel 2605.26242 | https://arxiv.org/abs/2605.26242 | Can LLMs Introspect? A Reality Check | Shashwat Singh, Linzen, Ravfogel | 25 May 2026; v2 21 Aug 2026 | PARTLY | **Supported:** layer-0 input probes match or beat the prompted, untrained reports, and the paper trains no reporter. **Not supported:** "only suggests interventional training". The only related remark (§4.3.2) is that models could probably be trained to tell activation interventions from input interventions. App. C.5 deliberately avoids fine-tuned models, because fine-tuning might solve the task by adding a task-specific mechanism instead of revealing a general capacity. Paper 9 should cite that point as a risk, not as support. |
| 7 | Cheah et al. 2607.21090 | https://arxiv.org/abs/2607.21090 | Training Large Language Models for Self-Explanation Faithfulness | Yeoktatt Cheah, Pérez-Ortiz, Siegel, Camburu | 23 Jul 2026 | **MISMATCH** (overstated) | **What is trained:** RL (GRPO) on self-explanation faithfulness, not "detection". The reward is 1 when the explanation mentions an inserted text edit exactly when that edit changed the decision. **Interventions:** edits to the input text (random-word insertion, user-bias phrases). **Transfer:** the authors call cross-type transfer limited, with small gains. Llama3.1-8B trained on random insertions reached 0.178 Phi-CCT on user bias (the row labels in Table 4 disagree with the text about direction). The reverse direction and Qwen3-8B did not replicate. **Fix:** write "weak, inconsistent transfer", not "did not transfer". |
| 8 | Bhan et al. 2506.09277 | https://arxiv.org/abs/2506.09277 | NeuroFaith: Evaluating LLM Self-Explanation Faithfulness via Internal Representation Alignment | Milan Bhan, Vittaut, Chesneau, Chandar, Lesot | 10 Jun 2025; v4 29 Jan 2026 | VERIFIED\* | **Quote:** on correctly answered Wikidata-2-hop items, models "identify the correct bridge object in their internal representations only 48-65% of the time". Decoding used Patchscopes; Gemma-2-2B gives 48.3% and Gemma-2-27B 64.9%, and Mistral-7B was also tested. **Caveat:** the authors attribute the misses to shortcuts and alternative pathways and do not consider decoder failure. This is decodability evidence, not proof that natural bridges differ from the one the input implies. |
| 9 | Guo, Ruis, Andreas, Li 2606.32038 (table) | https://arxiv.org/abs/2606.32038 | Introspective Coupling: Self-Explanation Training Tracks Behavioral Change Despite Fixed Supervision | Zifan Carl Guo et al. | 30 Jun 2026 | VERIFIED | Supervised by counterfactual behaviour on modified inputs. The trained explanations track the model's current behaviour more than their fixed targets. The row classification fits. |
| 10 | Ji-An et al. 2505.13763 | https://arxiv.org/abs/2505.13763 | Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations | Li Ji-An, Xiong, Wilson, Mattar, Benna | 19 May 2025; v2 24 Oct 2025 | VERIFIED | In-context neurofeedback only, with no fine-tuning. "In-context only" is correct. |
| 11 | Aoki et al. 2609.00904 (table) | https://arxiv.org/abs/2609.00904 | In-Context Neurofeedback: Can LLMs Control Their Internal Representations through Privileged Access? | Koshiro Aoki et al. | 1 Sep 2026 | VERIFIED (row slightly off) | The redesign makes the control target privileged: a third party cannot infer it from the prompt. So "labels independent of input" should read yes or partly, not "no". Models showed no reliable control. |
| 12 | Shenoy et al. 2604.16812 (table) | https://arxiv.org/abs/2604.16812 | Introspection Adapters: Training LLMs to Report Their Learned Behaviors | Keshav Shenoy, Yang, Sheshadri, Mindermann, Lindsey, Marks, Wang | 18 Apr 2026; v2 28 Apr 2026 | VERIFIED | One shared LoRA adapter, trained across fine-tunes with implanted behaviours, generalizes to unseen fine-tunes and gives state-of-the-art AuditBench results. |
| 13 | Binder et al. 2410.13787 (table) | https://arxiv.org/abs/2410.13787 | Looking Inward: Language Models Can Learn About Themselves by Introspection | Felix J Binder et al. | 17 Oct 2024 | VERIFIED | Tests self-prediction against a cross-trained second model, a third-party predictor. |
| 14 | Ferrara 2608.20569 (table) | https://arxiv.org/abs/2608.20569 | Open-Weight Masked Introspection: Measuring What Language Models Can Report About Their Own Computation | Emilio Ferrara | 20 Aug 2026 | VERIFIED | **Undercited.** **Controls:** sham runs, impact-matched random perturbations and a text-only observer, the closest precedent for T4 and the input-only bar. **Fine-tuned arm:** a LoRA on Qwen2.5-7B with labels "detected/not detected" over 400 random directions, scoring 100/100 on held-out directions; no held-out families. **Self-report:** AUROC about 0.5007 across 78k measurements. |
| 15 | Karvonen et al. 2512.15674 (table) | https://arxiv.org/abs/2512.15674 | Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers | Adam Karvonen et al. (11 authors) | 17 Dec 2025; v2 6 Jan 2026 | VERIFIED | Row fits. |
| 16 | Pan, Chen, Steinhardt 2412.08686 (table) | https://arxiv.org/abs/2412.08686 | LatentQA: Teaching LLMs to Decode Activations Into Natural Language | Alexander Pan et al. | 11 Dec 2024; v2 23 Mar 2026 | VERIFIED | Row fits. |
| 17 | Anthropic alignment blog, introspection adapters | https://alignment.anthropic.com/2026/introspection-adapters/ | Introspection Adapters: Training LLMs to Report Their Learned Behaviors | Keshav Shenoy et al. | 28 Apr 2026 | VERIFIED | Same work as row 12, so the comparison table counts it twice. **What the post adds:** SFT plus DPO; AuditBench 59% against 53% for the next-best method and 44% for Activation Oracles. **Missing:** no test on activation interventions or natural computation. |
| 18 | Steinmetz Yalon et al. 2602.02467 (table) | https://arxiv.org/abs/2602.02467 | Indications of Belief-Guided Agency and Meta-Cognitive Monitoring in Large Language Models | Noam Steinmetz Yalon et al. | 2 Feb 2026 | VERIFIED | Row fits. Singh et al. show layer-0 probes match these reports. |
| 19 | Baldelli et al. 2610.20179 (table) | https://arxiv.org/abs/2610.20179 | Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs | Davide Baldelli et al. | 14 Oct 2026 | VERIFIED | Prompt-only, no training. A "writer" model serves as the third-party comparison; 3 of 10 models show small control effects. |
| 20 | Song et al. 2508.14802 (table) | https://arxiv.org/abs/2508.14802 | Privileged Self-Access Matters for Introspection in AI | Siyuan Song et al. | 20 Aug 2025 | VERIFIED | Gives a third-party reliability definition of introspection. |
| 21 | Biran et al. 2406.12775 (table) | https://arxiv.org/abs/2406.12775 | Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries | Eden Biran et al. | 18 Jun 2024; v2 14 Oct 2024 | VERIFIED | Finds the bridge resolved in early layers; back-patching fixes up to 66% of previously incorrect cases. |
| 22 | Shibata 2011, Science | https://doi.org/10.1126/science.1212003 | Perceptual Learning Incepted by Decoded fMRI Neurofeedback Without Stimulus Presentation | Kazuhisa Shibata, Watanabe, Sasaki, Kawato | Science 334(6061):1413–1415, 9 Dec 2011 | VERIFIED | science.org returned 403, so I verified through Crossref and Europe PMC (PMID 22158821, PMC3297423). The orientation-specific learning claim matches. |
| 23 | Cortese 2016, Nat Commun | https://pmc.ncbi.nlm.nih.gov/articles/PMC5171844 | Multivoxel neurofeedback selectively modulates confidence without changing perceptual performance | Aurelio Cortese, Amano, Koizumi, Kawato, Lau | Nat Commun 7:13669, 15 Dec 2016 (doi 10.1038/ncomms13669) | VERIFIED | Confidence moved both up and down while accuracy stayed unchanged. |

**Statements elsewhere in P9_PRIOR_ART.md that are now wrong or need narrowing:**
- "Nobody has directly compared training on ordinary examples with training on intervened ones either" (Plain answer) and "none compares observational with interventional training" (What is genuinely open). Both are **contradicted in general** by the anonymous ARR October 2026 submission (see C1). Narrow both to task intermediates under answer-fixed patches.
- "I found no verified human study that trains self-report on decoded labels" is **contradicted**. Frederick (2012), *Psychophysics of EEG alpha state discrimination*, Consciousness and Cognition 21(3):1345–1354, doi 10.1016/j.concog.2012.06.009, verified through Europe PMC. Participants reported "high" or "low" alpha whenever EEG-detected alpha events triggered a prompt, and got correctness feedback. 75% of 106 reached significant discrimination within nine sessions. It replicates Kamiya (1962), which I did not open. The labels are observational (natural fluctuations), so this is the human analogue of Paper 9's O arm. Decoded neurofeedback (Shibata, Cortese) remains the analogue of producing an induced state. I found no human study that trains report on induced (interventional) states.
- "Warning: … trained detection of text interventions did not transfer across intervention types". Change this to "transferred only weakly and inconsistently", per row 7.

---

## (C) New or missed work, classified on the four criteria

The four criteria are:
1. **In-place:** the same model reads its own running computation, with no separate decoder model or separate prompt that receives the activation.
2. **Answer-fixed interventional labels on a task intermediate.**
3. **Natural transfer against an input-only predictor.**
4. **Observational versus interventional training compared head to head.**

Each cell gives yes, partly, no or unclear, with a short reason. Papers are ordered from closest to Paper 9 down to background.

### C-i. Closest (must cite; affect the novelty wording)

| ID | Paper | URL | Date | (1) In-place | (2) Answer-fixed interventional labels, task intermediate | (3) Natural transfer vs input-only | (4) Obs vs int training | One-sentence summary |
|---|---|---|---|---|---|---|---|---|
| C1 | Anonymous, **"Interventional Training Reduces Causal Bypass in In-Context Introspection"** (ACL ARR Oct 2026, Submission 8979) | https://openreview.net/forum?id=5P7dVpGNPO (I read the abstract through the OpenReview search API; the forum page is behind a browser check, which I did not bypass) | 2026-10-13 | **Likely yes.** "In-context introspection" of its own signals, but the abstract does not say how the signal is read. | **Partly.** Labels are counterfactually perturbed values of internal signals, but the signals are low-level (e.g., one neuron's activation), not a task intermediate, and nothing is held fixed. | **Partly.** Tests endogenous (natural) and injected signals, including out-of-distribution, but no input-only predictor is mentioned. | **YES.** Output-only fine-tuning against interventional training, head to head. | Trains hundreds of single-locale introspectors (the paper's term) across three low-level signal types and four LMs. Shows output-only fine-tuned reporters show causal bypass, and reports that interventional training improves introspection of natural and injected signals, generalizes across token positions and scales with data. |
| C2 | Geiping, **"A Sense of Proprioception in Language Models"** | https://arxiv.org/abs/2610.17495 (arXiv HTML returned 404, so I read the full text through https://www.alphaxiv.org/abs/2610.17495.md) | 12 Oct 2026 | **Yes.** The same weights answer questions about routing, entropy, logit-lens and decision depth at earlier token positions in the conversation. | **No.** Labels are the model's live, unintervened signals. Routing edits (remove or replace the top-1 expert, router noise) appear only at evaluation, where on-policy fine-tuned models predict the post-edit expert above chance, except when the top expert is swapped for a random one. | **Partly.** A token-identity lookup baseline (routing 19.0% against about 62–65% trained; entropy 21.8% against 43–54%), but the test is in-distribution on natural signals and not about task intermediates. | **No.** It compares live labels with stale base-model labels, not with interventional training. | Main model LFM2.5-8B-A1B, plus OLMoE, Qwen3.5/3.6-35B-A3B and GLM-4.7-Flash. Training is SFT and on-policy SFT with labels read from the same forward pass, and capabilities are largely preserved. |
| C3 | Anonymous, **"Introspection or Post-Hoc Rationalization? A Cross-Model and Cross-Domain Study of LLM Metacognition"** (ICLR 2027 submission) | https://openreview.net/forum?id=GqNWne0GxX (abstract read through the search API) | 2026-09-18 | **Partly.** Accurate self-reports on introspection tasks, but no activation reading is described. | **No.** | **Partly.** External reporters get exactly the self-reporter's inputs (a third-party, input-only bar) and match its accuracy. Fine-tuning gains across 3 task families and 16 datasets mostly fail to transfer between tasks. | **No.** | 8 LLMs (7B–70B). Concludes that introspection training builds task-specific rationalization rather than privileged access. |
| C4 | Karvonen, Ong, Kantamneni, Marks, **"Would this change your answer? Evaluating Explanations of LLM Behavior In The Wild with Counterfactual Experiments"**, plus LessWrong follow-up **"Training Models to Predict and Explain Their In-the-Wild Behavior"** | https://arxiv.org/abs/2608.16747 ; https://www.lesswrong.com/posts/YyAMz52wDxnLhwvWL/training-models-to-predict-and-explain-their-in-the-wild | 17 Aug 2026; LW 4 Sep 2026 | **Partly.** Self-prediction of behaviour, with no activation reading. | **No.** Labels come from counterfactual edits to the input prompt. | **Partly.** In the Binder-style test, a cross-trained other model (Llama-3.1-8B against Qwen3-8B) scored within error bars of the self-trained one, so no self-advantage. Gains did transfer to a held-out hint setting. | **No.** | CHIVE finds and tests unusual behaviours in the wild. Training on counterfactual outcomes generalizes, but open-ended self-explanation gains were weak. |
| C5 | SPAR Fall 2026 project **"Introspection Training for Verbalizing Activations"** (mentor Belinda Li, Anthropic) | https://www.sparai.org/projects/f26/recNKpeygLfUGyGiz | Fall 2026, running to mid-December | Unclear | **Planned.** Ground truth for verbal reports would come from probe readouts, feature activations, and the measured effects of ablation and patching. | Not mentioned | Not mentioned | A planned project, not a publication. It is the most direct concurrent-work (scooping) risk for part 2, and results could appear around Dec 2026 to early 2027. |
| C6 | Atkinson, Plunkett, Bau, **"Identifying Introspection From the Inside"** (COLM 2026) | https://arxiv.org/abs/2610.07186 | 5 Oct 2026 | **Yes.** Self-report of LoRA-learned preferences. | **No.** Training does not supervise self-report; weight ablations and attribution patching are used only for analysis. | **Partly.** Separates faithful from confabulating models mechanistically, but no input-only predictor is described. | **No.** | Faithful self-report emerges late in training and leaves structural signatures. |
| C7 | Zeng, Assis, Wang, **"Evaluating and Improving LLM Self-Modeling"** (EMNLP 2026) | https://arxiv.org/abs/2608.30980 | 31 Aug 2026 | **Partly.** Verifiable questions about the model's own behaviour. | **No.** | **Partly.** Some transfer to held-out tasks, but the authors doubt the gains reflect privileged access. | **No.** | A benchmark plus RL on synthetic self-modeling data. |

### C-ii. Relevant, should be cited

| Paper | URL | Date | (1) | (2) | (3) | (4) | One-sentence summary |
|---|---|---|---|---|---|---|---|
| Yoshida et al., Introspecting Alignment Shifts Beyond Behaviors Implanted Through Fine-Tuning | https://arxiv.org/abs/2608.04347 | 5 Aug 2026; v2 26 Sep 2026 | Partly (the adapter takes activation differences) | No | Partly (unseen fine-tunes and safety categories) | No | Delta-Aware Introspection Adapter describes unintended alignment shifts and beats Introspection Adapters. |
| Kutsyk & Zieliński, Revealing Hidden Model Behaviors with Task-Specific Self-Reports | https://arxiv.org/abs/2607.03640 | 3 Jul 2026; v2 28 Jul 2026 | Partly | No | Partly (7 implanted behaviours, compared with Introspection Adapters) | No | A LoRA adapter (SAR) makes a fine-tuned model describe its hidden behaviour and halves hallucinated reports compared with Introspection Adapters. |
| Pres, Ruis, Ghebreselassie, Li, Andreas, Self-CTRL: Self-Consistency Training with Reinforcement Learning | https://arxiv.org/abs/2606.18327 | 16 Jun 2026 | Partly (self-explanations of biases and refusal) | No | Partly (held-out distributions; a third-party auditor's accuracy rises from 36% to 92%) | Partly (compares a self-consistency signal with direct ground-truth supervision) | Consistency RL raises R² between reported and measured bias from 0.24 to 0.64, matching ground-truth supervision. |
| Zou, Sun, Kong, Wang, A mechanistic study of language model introspection | https://arxiv.org/abs/2609.35108 | 28 Sep 2026 | Yes (localizes an injection among 10 positions in the same pass) | Partly (concept injection, analysis only, no training) | No | No | Middle-layer "gate" heads and later "router" heads support detecting and localizing injections. |
| Konrad, Tanyel, Ayvaz, Self-Reports Do Not Identify Self-Models: An Identifiability Test for Counterfactual Reports | https://arxiv.org/abs/2609.32449 | 26 Sep 2026 | Yes | Partly (activation interventions on affect-like states; no training mentioned) | Partly (environment-shift tests, no input-only predictor) | No | Self-reports follow the demonstration environment, so benchmarks need shift-invariance tests under a fixed intervention. |
| Macar, Yang, Wang, Wallich, Ameisen, Lindsey, Mechanisms of Introspective Awareness | https://arxiv.org/abs/2603.21396 | 22 Mar 2026; v5 10 Jun 2026 | Yes | Partly (DPO and a trained bias vector raise detection of injections; no task intermediate) | No | No | A two-stage circuit (evidence carriers, then gates). DPO can elicit detection where SFT does not, and a trained bias vector adds 75% on held-out concepts. |
| Nguyen, Ahmed, Kim, Can LLMs Reliably Self-Report Adversarial Prefills, and How? | https://arxiv.org/abs/2606.23671 | 22 Jun 2026; v5 1 Sep 2026 | Yes | No | Partly (prefilled against natural outputs) | No | Introspection training raises accuracy but **does not transfer** to the tampering probe, which is a risk for T2/T3. |
| Hahami et al., Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs | https://arxiv.org/abs/2512.12411 | 13 Dec 2025; v2 1 Mar 2026 | Yes | No (untrained) | No | No | Earlier work by the IFT team: binary yes/no detection is a logit-shift artifact, while localization works only for early-layer injections. |
| Li, Ceballos Arroyo, Rogers, Saphra, Wallace, Do Activation Verbalization Methods Convey Privileged Information? | https://arxiv.org/abs/2509.13316 | 16 Sep 2025; v4 13 May 2026 | No (a separate verbalizer) | No | Yes-ish (verbalizer benchmarks are solvable without internals) | No | Verbalizations often reflect the input or the verbalizer's own knowledge. This is the input-only argument applied to verbalizers. |
| Fraser-Taliente, Kantamneni, Ong et al. (Anthropic), Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations | https://transformer-circuits.pub/2026/nla/index.html | 7 May 2026 | No (activation inserted into a separate prompt's token) | No (unsupervised RL reconstruction) | No input-only baseline described | No | A verbalizer and reconstructor trained to round-trip activations. Edits to explanations steer the model. |
| Anonymous, Training Language Models for Introspective Oversight (ICLR 2027, desk rejected) | https://openreview.net/forum?id=M7bhrIY7b9 (abstract read through the search API) | 2026-09-19 | Unclear (described only as a separate observer channel) | No (rewards checked against actual responses) | Partly (AUROC 0.907 against transcript monitoring 0.828 and Activation Oracles 0.788 on held-out oversight evaluations) | No | A trained observer answers verifiable questions about the assistant's behaviour. |
| Martorell & Bianchi, Don't Ask: Understanding What Numeric Self-Reports Measure in Language Models (v1/v2 title: "Quantitative Introspection in Language Models…") | https://arxiv.org/abs/2603.18893 | 19 Mar 2026; v3 10 Oct 2026 | Yes | Partly (steering, no training) | Partly (compared with probe scores) | No | Numeric affect self-reports track the state after the question more than before it, and are not concept-specific. |
| Zhao, Cao, Wang, Gui, Think in Latent, Explain in Language: Self-Explainable Latent Reasoning | https://arxiv.org/abs/2608.13570 | 1 Jul 2026 | Yes (the same model decodes its own latent states) | No (CoT-style supervision) | No | No | Trains latent reasoning and its verbal decoding jointly, with no faithfulness test. |
| Goel, Kim, Shavit, Wang, Learning to Interpret Weight Differences in Language Models (DIT) | https://arxiv.org/abs/2510.05092 | 6 Oct 2025; v4 21 Mar 2026 | Partly | Partly (labels from synthetic, planted weight diffs, not activation patches of a task step) | No | No | An adapter makes a fine-tuned model describe its own weight changes. |
| Turpin et al., Teaching Models to Verbalize Reward Hacking in Chain-of-Thought Reasoning (VFT) | https://arxiv.org/abs/2506.22777 | 28 Jun 2025 | Partly | Partly (labels from the influence of input cues) | Partly (held-out cues) | No | Verbalization fine-tuning raises admission of cue influence; after RL only 6% of hacks go undetected, against 88% without it. |
| Li, Phuong, Tan, Spilling the Beans: Teaching LLMs to Self-Report Their Hidden Objectives | https://arxiv.org/abs/2511.06626 | 10 Nov 2025; v5 22 Mar 2026 | No | No | Partly (out-of-distribution transfer) | No | Training models to admit factual errors transfers to confessing hidden objectives (F1 0.98). |
| Betley et al., Tell me about yourself: LLMs are aware of their learned behaviors | https://arxiv.org/abs/2501.11120 | 19 Jan 2025 | Partly | No | No | No | Fine-tuned models describe their trained behaviours without being trained to. |
| Bozoukov et al., Minimal and Mechanistic Conditions for Behavioral Self-Awareness in LLMs | https://arxiv.org/abs/2511.04875 | 6 Nov 2025 | Partly | No | No | No | A rank-1 LoRA is enough for behavioural self-awareness, and one steering vector captures most of the effect. |
| Lindsey, Emergent Introspective Awareness in Large Language Models | https://arxiv.org/abs/2601.01828 | 5 Jan 2026 | Yes | No (untrained) | No | No | arXiv version of the concept-injection work. The author describes the ability as very unreliable and dependent on context. |
| Pearson-Vogel, Vanek, Douglas, Kulveit, Latent Introspection: Models Can Detect Prior Concept Injections | https://arxiv.org/abs/2602.20031 | 23 Feb 2026 | Yes | No (untrained) | No | No | The residual stream detects injections that the sampled text denies. |
| Morris & Plunkett, Tests of LLM introspection need to rule out causal bypassing (LessWrong/AF) | https://www.lesswrong.com/posts/LD8yupMtE6btAE3R9/tests-of-llm-introspection-need-to-rule-out-causal-bypassing | 28 Nov 2025 | n/a (conceptual) | n/a | n/a | n/a | Defines "causal bypassing", where an accurate report comes through a route that skips the internal state. Paper 9's answer-fixed patch is a direct answer to this, so cite it as motivation. |

### C-iii. Context (checked; not close on the four criteria)

| Paper | URL | Date | Note |
|---|---|---|---|
| Berg & Kaiser, Language Models Act on Hidden Valence | https://arxiv.org/abs/2609.35591 | 28 Sep 2026 | Changes the hidden KV cache while **visible text is held identical**. This precedents Paper 9's answer-fixed idea, but for behaviour, not trained report. |
| Kowalski, Fonseca Rivera, Macar, Africa, Measuring Activation Control in Large Language Models | https://arxiv.org/abs/2608.21664 | 21 Aug 2026 | Neurofeedback-style control benchmark; models can partly evade probes and activation oracles. |
| Ashuach et al., Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness | https://arxiv.org/abs/2604.12373 | 14 Apr 2026 | Probes on the model's own states beat probes on peer models only on items where models disagree, which is a useful design idea for T3. |
| Naphade et al., Me, Myself, and π: Evaluating and Explaining LLM Introspection | https://arxiv.org/abs/2603.20276 | 17 Mar 2026 | Models predict their own policies better than peer models do; untrained. |
| Sobotka, Karabag, Topcu, Why Do LLMs Struggle in Strategic Play? Broken Links Between Observations, Beliefs, and Actions | https://arxiv.org/abs/2605.00226 | 30 Apr 2026 | Decoded internal beliefs are more accurate than verbal reports, and accuracy drops with multi-hop reasoning. |
| Kalyvas et al., An Empirical Study of Counterfactual Self-Explanations in LLMs | https://arxiv.org/abs/2609.17119 | 15 Sep 2026 | Behavioural counterfactual self-explanations; untrained. |
| Kirin, Operational Proto-Introspection in Looped Language Models… | https://arxiv.org/abs/2607.18553 | 20 Jul 2026 | An external probe reads computation quality. The readout helps choose outputs but cannot be used to steer generation. |
| Xu et al., When Words Fall Short: Iterative Synergy Between Verbalized Reasoning and Hidden Features for LLM Confidence Estimation | https://arxiv.org/abs/2609.34454 | 28 Sep 2026 | Training of confidence verbalization coupled with a hidden-feature estimator. |
| Anonymous, WaXup: Weight Averaging of Experts as Data Augmentation for Robust Introspection (ICLR 2027 submission) | https://openreview.net/forum?id=voVokAGQjL (search API) | 2026-09-17 | Augmentation by weight interpolation for introspection-adapter-style training. |
| Dinoraptor101, LLM introspection research is missing Nisbett & Wilson's control group (LessWrong; the page shows an automated rejection notice) | https://www.lesswrong.com/posts/6d3ffDFt9tBYdiaoe/llm-introspection-research-is-missing-nisbett-and-wilson-s | 21 Aug 2026 | Argues for outside-observer arms (the Nisbett & Wilson design) in LLM introspection tests. Weak source, but useful framing for the input-only bar. |
| Frederick, Psychophysics of EEG alpha state discrimination (human) | https://pmc.ncbi.nlm.nih.gov/articles/PMC3424312 (read through Europe PMC) | Sep 2012 | Training self-report on decoded EEG labels, with observational labels. See the fix in (B). |

**Search conclusion.** No item does **(2) and (3) together**. The only item with **(4) = yes** is C1, and it concerns low-level signals rather than a task intermediate. Nothing reports a trained self-report of a **bridge entity** or other two-hop intermediate. Pepper et al. decode bridges with an observationally trained adapter, and Bhan et al. analyse bridges without training a reporter.

---

## (D) The narrow open claim, and the strongest threat to each part

**Claim, restated precisely.** In Qwen3-1.7B, a LoRA-trained forced-choice self-report uses the same weights in a follow-up turn. It names the bridge country of the two-hop prompt "Which continent is the country containing {city} in?". It is trained on labels from **answer-fixed activation patches** at the city/bridge positions. The donor city is in another country on the same continent, so the input text and the final answer are unchanged. The claim is that this reporter:
- (a) transfers to an unseen intervention family (a different layer or position, and additive steering);
- (b) on **unpatched** items, predicts the **causally measured natural bridge** better than an **input-only predictor**.

The comparison runs head to head against observational-label (O), injection-trained (J) and untrained (N) reporters, with identical LoRA budget, example count and format. KL-matched random perturbations serve as the tamper control (T4), and T5 checks capability drift.

**Wording I would defend.** "To our knowledge, as of 2026-10-21, no prior or concurrent work trains a model's self-report of a *task intermediate* on answer-fixed interventional labels and evaluates it on the model's *natural* computation against an input-only predictor. Observational and interventional training have been compared for low-level internal signals (Anonymous, ARR Oct 2026), but not for task intermediates."

| Part | Strongest threat | What to do |
|---|---|---|
| (1) Same model reads its own running computation in place | **Geiping 2610.17495**: the same weights report routing, entropy and logit-lens values for earlier tokens in context, with labels taken from the same forward pass. C1 ("in-context introspection"), IFT (localization within the same pass) and Ferrara (probes in the same context) do this too. | Do not claim (1) as new. Present it as a design property and cite Geiping, IFT and C1. |
| (2) Training labels from an answer-fixed intervention on a task intermediate | **C1** trains on counterfactually perturbed internal values, though they are low-level and nothing is held fixed. **Li et al. 2511.08579** uses labels derived from patches, but the answer changes and the explainer reads the patch in a separate prompt. **IFT and Steering Awareness** use injection labels, not task steps. **Berg & Kaiser 2609.35591** manipulates hidden state with visible text held fixed, but for behaviour, not trained report. Concurrent risk: the **SPAR project (C5)** plans to train verbalizations on ablation and patching ground truth. | Claim only the conjunction "answer-fixed AND task intermediate (the bridge)". Move fast, and check C5's outputs and any arXiv version of C1 before submission. |
| (3) Natural transfer, scored against an input-only predictor | Input-only and third-party bars are **established**: Singh et al. (layer-0 probes match the reports); C3 (external reporters with identical inputs match self-accuracy, and introspection fine-tuning fails across tasks); C4 (a cross-trained other model is within error of the self-trained one); Li et al.'s "– activation" ablation (64.0 → 59.9); Geiping's lookup baseline; Ferrara's text-only observer and impact-matched perturbations. **Empirical threat:** if natural bridges almost always equal the country the input implies, T3 cannot separate reading from reconstruction. Bhan's 48–65% is Patchscopes decodability, possibly decoder misses, not a causal measure. | Present the novelty as the target, the *causally measured natural bridge* on unpatched items, combined with the training-label manipulation, not as the existence of a baseline. Report how often the natural bridge diverges from the input-implied one under Paper 9's own causal measure before relying on T3. Credit Singh, C3, Ferrara and Li. |
| (4) Observational vs interventional training, head to head | **C1 (ARR Oct 2026)** compares output-only fine-tuning with interventional training directly and reports better transfer to natural (endogenous) signals. This is the **main threat**. Also **Geiping**: an observationally trained reporter predicts routing edits above chance, so the O arm may come close to I on T1/T2, which shrinks the expected I-vs-O gap. | Cite C1 as concurrent. Narrow (4) to a task intermediate under answer-fixed patches, plus the J and N arms, all judged against the input-only bar on natural items. Pre-register what an O ≈ I result would mean. |

**Risks to the outcome rather than to novelty:**
- **Cross-type transfer is weak:** Cheah et al.; Nguyen et al. 2606.23671 (no transfer to the tampering probe); C3 (fails across tasks); Fonseca Rivera & Africa (only geometrically similar vectors). This bears on T2.
- **Fine-tuning may install a task-specific mechanism** (Singh et al., App. C.5). This bears on the interpretation of T3.
- **Privileged access may not show:** cross-trained predictors matched self-trained ones in C4 and Binder-style tests. This argues for running a cross-model reporter as an extra input-only-style control.

---

## (E) Search log

All results were read on 2026-10-21. Where results are sorted by date, the newest relevant item found is listed.

**arXiv API** (`export.arxiv.org/api/query`, sorted by submittedDate, newest first):
1. `all:introspection AND all:fine-tuning` (50): 36 results, newest 2026-10-09; found Atkinson 2610.07186 and Yoshida 2608.04347.
2. `ti:introspection OR ti:introspective OR ti:introspect` (60): 175 total, read down to about 2025-05; found Zou 2609.35108, Kirin, Macar, Lederman, Pearson-Vogel and Hahami 2512.12411.
3. `abs:introspection AND abs:activation` (50): 78 total, read 41; found Geiping 2610.17495, Berg, Kowalski and Dadfar.
4. `abs:"self-report" AND abs:activation` (50): 2026-06 to 2026-10.
5. `abs:"privileged access" AND (language OR LLM)` (50).
6. `abs:"self-interpretation" OR "self-interpret" OR "explain their own"` (50).
7. `abs:"bridge entity"` (50): 19 results, all years.
8. `(steering AND awareness) OR "injected thought" OR "concept injection" OR "injected concept"` (50): 690 total; the first 39 (Oct 2026) were irrelevant.
9. `abs:neurofeedback AND (language OR LLM OR LLMs)` (40): 6 results.
10. `LatentQA OR "activation oracle(s)" OR "activation verbalizer/verbalization"` (50): 15 results.
11. `"interchange intervention" AND (explanation(s) OR verbal OR language)` (40): 19 results.
12. `(introspection|introspective|metacognitive|self-report) AND (patching|counterfactual|interventional) AND (train|training|fine-tune|finetune)` (50): 127 total, read about 38.
13. `("multi-hop"|"two-hop"|"2-hop") AND (introspection|"self-report"|verbalize|introspective|"self-explanation")` (40): 20 results.
14. `(introspection|introspective|self-report|metacognition) AND "ground truth" AND LLM` (50): 46 total, read 38; found Self-CTRL 2606.18327.
15. `(SelfIE|Patchscopes|Patchscope|"self-interpretation"|"self-explaining") AND (train…)` (40): found SELR 2608.13570.
16. `(steering|injection|injected|perturbation) AND (introspection|introspective|self-report|awareness) AND LLM` (60): 1,343 total; the first 40 (Oct 2026) were irrelevant.
17. `(introspection|introspective|"self-report(s)"|verbalize|verbalization) AND ("activation patching"|"causal intervention(s)"|interchange)` (50): 17 results, none trains report.
18. `("latent reasoning"|"implicit reasoning"|"latent multi-hop") AND (verbalize|report|introspection|explain)` (40).
19. `"causal bypass(ing)"` (30): 1 irrelevant result.
20. `abs:introspection OR introspective OR introspect` (40): 598 total, read 40 (2026-07-30 to 2026-10-13).
21. `("self-knowledge"|"self-awareness"|metacognition|metacognitive) AND (activation(s)|"internal state(s)"|"hidden state") AND LLM` (50): 56 total, read 42.
22. `introspectors OR "in-context introspection" OR "interventional training"` failed (HTTP 429, then 503); I used the arXiv search page instead.

**arXiv search pages** (`arxiv.org/search`, newest first):
- "introspectors": 4 results, none relevant.
- "interventional training": 31 results, none about LLM introspection; no arXiv version of C1.
- "introspection input-only".
- "self-report activation patching language model": 2 results.
- "introspection language models" (all fields): newest relevant 2026-10-12, Geiping.
- "self-report internal language model": 2026 results to 2026-10-19; found Nguyen 2606.23671 and IPoET 2609.34454.

**OpenReview** (`api2.openreview.net/notes/search`; forum pages and `notes?id=` are behind a browser challenge, which I did not attempt to bypass):
- "introspection" (2 pages): found C1 (ARR Oct 2026), C3 (ICLR 2027), WaXup (ICLR 2027) and "One Concept Per Activation" (NeurIPS 2026 workshop; abstract only skimmed).
- "causal bypass introspection".
- "Introspection or Post-Hoc Rationalization metacognition".
- "introspective self-report activation": found "Training Language Models for Introspective Oversight".
- "privileged access introspection language model".
- "self-interpretation activations trained own hidden states".
- "introspection fine-tuning counterfactual perturbation self-report".
- "latent multi-hop reasoning bridge entity self-report verbalize": NeuroFaith's ICLR 2026 version "Did You Faithfully Say What You Thought?…" was rejected.
- "WaXup…", "steering awareness injected concept detection fine-tuning" (Steering Awareness accepted at COLM 2026), "Interventional Training Reduces Causal Bypass", "Training Language Models for Introspective Oversight".

**WebSearch:**
- training self-report activation-patching labels natural transfer 2026 (extended)
- introspection fine-tuning input-only baseline 2026 (extended)
- LLM bridge-entity self-report "which country"
- LessWrong/AF training introspection with activation patching 2026 (extended)
- "privileged access" training third-party predictor 2026 (extended)
- self-explanation faithfulness training with causal hidden-state interventions 2026
- introspection Sep–Oct 2026, trained, intervention generalization (extended)
- verbalizing intermediate steps with activation ground truth, unpatched generalization (extended)
- "observational" vs "interventional" labels in introspection
- "transient single-token injections" 85% (led to the v1 title of 2511.21399 and the LessWrong steering-detection post)
- "causal bypass" introspection (extended)
- alignmentforum introspection training 2026 (extended)
- CoT faithfulness training with causal intervention 2026 (found CASE 2607.18820, LIFT and counterfactual simulation training; these train CoT, not self-report of internals, and were not opened)
- the Geiping title
- the DIT, Spilling the Beans and VFT titles
- injected-concept training generalizing to endogenous states 2026 (extended)
- patching "answer unchanged" bridge self-report 2026
- Qwen3 LoRA same-model reporter with intervention labels 2026
- self-simulation / external-reporter cross-model transfer 2026
- alignment.anthropic.com 2026 introspection verbalizer (found the SPAR project)
- transluce.org 2026 self-explanation (found 2609.17119 and the Karvonen et al. LessWrong post)
- "beyond the input" input-only predictor 2026
- "natural language autoencoders" 2026
- Frederick 2012 EEG alpha discrimination

**Date coverage.** Focus was 2025-01 to 2026-10-21, with older work checked where it was cited. The newest relevant items found were C1 (2026-10-13), Baldelli (2026-10-14, already cited) and Geiping (2026-10-12). No new relevant arXiv item dated after 2026-10-14 turned up in the listings I checked.

**Access notes:**
- arXiv HTML for 2610.17495 returned 404, so I read the text through alphaxiv's web rendering. I fetched no PDFs.
- science.org returned 403 for the Shibata DOI, so I used Crossref and Europe PMC instead.
- PMC pages showed a browser check, so I used the Europe PMC REST API instead.
- OpenReview forum pages showed a browser challenge. I did not bypass it and read the notes through the search API instead.
- The transformer-circuits NLA page contained an embedded instruction aimed at AI summarizers. I ignored it.
- The page reader would not reproduce abstracts verbatim, so abstracts are paraphrased.
- C1, C3, WaXup and the oversight paper are anonymous submissions under review. Their contents could change, and their details come from abstracts only.

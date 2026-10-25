**Verdict: NOT READY.** `main.tex` was edited at 14:40:04, after `main.pdf` was built at 14:14:47. The PDF and the staged release bundle therefore still carry the author-approval text that the 14:40 DEVIATIONS entry withdraws as overstated. The fix is a rebuild: the current source builds cleanly (my test build in `audit/paper_audit_final/build_check/`: 18 pages, no warnings). After that, four should-fix wording items remain. Every number checks out.

**Who and when.** Independent auditor (Claude, Opus 5.5), 2026-10-25, about 14:20–14:45 (+04:00).
- I changed no existing file. I wrote only this report and the files in `audit/paper_audit_final/`.
- CPU only, at most 4 threads, nothing downloaded.
- I did not open any `sealed` folder, `labkeys` or `research_loop/`.
- **Concurrent edit.** `main.tex`, `RESULTS.md` and `DEVIATIONS.md` changed at 14:40:04, while this audit was running. I diffed them against the 14:24 snapshot in `paper9_ms/repo/` (identical to the version I first read) and audited the four changed manuscript lines too (l.61, l.415, l.469, l.471).
- Line numbers below refer to `main.tex` as of 14:40:04. The edit replaced lines in place, so the numbering is unchanged.

**What I checked.**
- **Numbers.** `audit/paper_audit_final/make_numbers_copy.py` is `make_numbers.py` with its output redirected to `regen/`. It reproduces `paper/numbers.json` byte for byte, and all four figure content streams are identical. `recompute_paper_copy.py` (output also redirected) reproduces `audit/paper_audit/recomputed_paper.json` byte for byte.
- **Figures.** `check_pdf_figs.py`: the four figure XObjects embedded in `main.pdf` (pages 6, 9, 10, 11) are byte-identical to the current `fig_*.pdf`. Re-check issue 11 is resolved.
- **Freeze hashes.** `check_hashes.py` checked all 67 lines in the three hash files, in the study folder and in the release repo:
  - one line fails, `README.md` in `PREDICTIONS.sha256`;
  - the two v3 lines superseded by the `.amend` file fail as expected, and the amend lines match;
  - all 18 `WS:` lines match.
- **Attribution.** I checked the attribution and approval sentences against the session transcripts (2026-10-21/22). Details are under Task 2.
- **Build.** The BibTeX and LaTeX logs are clean. Every `\ref` resolves.

---

## Task 1. Status of the re-check's open items

| Item | Status | Current text (line) |
|---|---|---|
| Issue 1 (title) | resolved (nit N1) | l.26 "Self-Report Positions in Qwen3-1.7B Track the Input, Not the Swapped-In Computation: Answer-Fixed Swaps, Probes and Null Training Results". It is scoped to one model and to positions, and the subtitle names probes. |
| Issue 3 (probe coverage) | resolved | l.375 "not linearly readable at any position probed after the digits: every position of the immediate report, and the final position of the delayed report." |
| Issue 4 (two routes) | **partly** | l.368 "**Multiplication.** They could be told apart." This contradicts l.370 ("Two routes remain open ... or read it from the lower layers of the answer positions"): one open route *is* reading the recipient's own computation. l.383 "rather than read from the process" has the same problem. See S2. |
| Issue 11 (stale Fig. 2) | resolved | The embedded Figure 2 equals the current `fig_probe_layers.pdf`. But the PDF as a whole is now stale; see B1. |
| Issue 19 ("at least 88%") | resolved | l.317 "at least 87\% of natural items". The minimum is 109/124 = 87.9%, and 12 of 16 checkpoints are at 87.9–100%: recomputed. |
| Issue 28 (ties) | resolved | l.249 "arms O, I and J exactly on 49--63\% ... (31\% and 63\% for the two X seeds)". Values: O, I, J 48.5–63.4%; X0 30.7%; X1 63.4%. |
| Issue 30 (human bullet) | resolved | l.385 "when a swap changes the confidence at the answer positions, the change is not linearly detectable at the positions that report it." |
| Issue 31 (records) | **partly** | Fixed: RESULTS.md l.272, STAGE0B.md l.49 and l.64–65. Still wrong: RESULTS.md l.273 "is lost at the first token ... at layers 16–27", and STAGE0B.md l.84 "a report ... computes its confidence again from the question". See S4. |
| New 1 (stale Fig. 2) | resolved | as issue 11 |
| New 2 (88%) | resolved | l.317 |
| New 3 (ties scope) | resolved | l.249 |
| New 4 (l.374 coverage) | resolved | l.375 |
| New 5 (l.364 Discussion) | **partly** | The Discussion now separates the two tasks (l.367–368). But l.368 "They could be told apart" is new and contradicts l.370. l.368 also says the positions "carried ... not the swapped-in confidence" without "linearly readable". See S2. |
| New 6 (l.384) | resolved | l.385 |
| New 7 ("lost", "do not carry") | resolved in the manuscript | l.43 "no longer linearly readable"; l.292 heading; l.300; l.311 "do not carry the swapped-in state in a linearly readable form". The records still say "lost" (RESULTS.md l.273). |
| New 8 (DEVIATIONS count) | **unresolved** | The 14:45 entry still says "4 major and 27 further issues ... revised for all of them". The 14:56 entry does not correct it. The audit listed 32 issues (4/11/17), and issue 32 is not addressed. See S4. |
| New 9a (l.323 Stage 0a only) | unresolved (optional) | l.324 "At Stage~0a's report position, the recipient's own label is linearly readable ... at 0.69--0.77". Arm O was trained on the immediate format, where the value is 0.67–0.72. See N2. |
| New 9b (issue 32, plain summary) | partly (optional) | l.39 opens plainly. Items (1)–(5) still use "placeholder readout", "rank-32 adapter" and "balanced accuracy" without definition. |
| New 9c (`references.bib` comments) | unresolved (cosmetic) | The comment `% ---- language models` (bib l.429) sits above `fleming2017self`, a human-metacognition entry. Not visible in the PDF. |

## Task 2. Sentences changed after the re-check, compared with the raw files

**Changes of 2026-10-22 14:56.** These are the items in the table above. Every changed number was recomputed:
- 87% / 109 of 124;
- 49–63%, 31% and 63%;
- coverage of the six-layer and five-layer probes;
- 0.33–0.36, 0.30–0.42 and 0.63–0.66 (`tailpos.json`);
- 0.35–0.41 over the three report cells: 0a 0.359–0.402, immediate 0.348–0.398, delayed 0.355–0.414.

All match.

**l.61 (intro, question 1; changed 2026-10-25 and again at 14:40).** The current text reads "The question, and the idea of training the self-report on the model's computation rather than its input, are the author's. The specific method, proposed by the AI system ..., is to train the self-report on examples where an answer-fixed intervention has changed an internal step ...".
- Supported by the coordinating-session transcript (2026-10-21 06:21:36Z). The author asked "why don't we just train them on their computation in some way instead, is that a paper?".
- In the reply (06:22:10Z), Claude proposed:
  - training on examples whose computation was changed and whose input was not;
  - the four-arm comparison;
  - the test on natural computation against an input-only predictor.
- The neurofeedback analogy came at 08:55:18Z, in reply to "do humans already do it?". **Match.**

**l.426 ("Files.")** This sentence is inaccurate; see S3. I checked it against the staged release (`paper9_ms/repo/DATA.md`, `strip_report.json`):
- Git already contains the per-item results. Zenodo adds only the eight JSON files larger than 2 MB: `res_v2/premise.json` and the seven `res_v3/eval_*.json`.
- So "the Zenodo archive, which adds the per-item results" overstates what Zenodo adds.
- No Zenodo DOI is given; the release README still has the placeholder `<ZENODO-DOI>`.
- I could not check the GitHub URL offline.

**l.415, l.469, l.471 (approvals and contributions; rewritten at 14:40).**
- Supported by the transcripts:
  - "Could you do everything tonight?" (2026-10-21 19:10:36Z).
  - No author message between 13:32Z and 19:10Z on 10-21, which spans the v2 freeze at 18:40Z. So "the second study ran ... without a separate approval" holds.
  - "Can we actually do our tests significantly faster? ... small scale?" / "Then do it" (10-22 07:53–07:54Z).
  - The first rule was announced to the author after it was sent (08:06:12Z, "I've sent the paper 9 session a design change ... Before anything is frozen, the same 10-minute probe has to confirm ...").
  - "Go ahead" (09:20:37Z) answered a plan that said the line stops "unless the swapped-in confidence survives a token or two after the digits". It gave no threshold, which matches "without being shown its threshold".
  - The S4′ item is on `research_next/VERIFIED_SHORTLIST.md` (2026-10-20 13:10): "Huginn ... self-report trained on input-independent natural variation".
- The superseded text, "The author approved each stage" and "the author approved them", was not supported. It is still in `main.pdf`: see B1.

**"Claude Opus 5.5 in every session" (l.459).**
- Every transcript that touches this study folder ran on `claude-opus-5-5`.
- The only other models found in those files belong to other studies' sub-agents (a Sonnet recompute of another folder; a Haiku loop agent). No P9 work used them.
- **Match**, as far as the local transcripts show.

**DEVIATIONS 2026-10-25 claims.** All verified:
- `numbers.json` and `recomputed_paper.json` regenerate byte for byte;
- the README hash line fails;
- every other hashed file matches.

## Task 3. Hostile read: other findings

1. **Stale build** (B1). `main.tex` 14:40:04 is newer than `main.pdf` 14:14:47 (the figures are 14:13:41). The PDF text still contains "The author approved each stage" and lacks the S4′ sentence. The staged release copies were built at 14:24, before the 14:40 edits: `paper9_ms/repo/`, `zenodo_upload/self-report-tracks-input.pdf` (byte-identical to the current `main.pdf`), the source zip and `code-and-data.zip`. They hold the old `main.tex`, `RESULTS.md` and `DEVIATIONS.md`. Only `release_files/README.md` and `zenodo_files/METADATA.md` were updated at 14:40.
2. **Discussion versus the two-routes caveat** (S2). l.368 "They could be told apart"; l.369 "This extends paper~8's \verdict{RECONSTRUCTION FROM INPUT} ... to the report position's internal state"; l.383 "the report is assembled from available cues, here the question, rather than read from the process". Each of these picks the recompute route, which l.45, l.370 and l.392 say the design cannot establish.
3. **The freeze statement omits the failing hash** (S3). l.404–410 state the freezes without noting that one hashed file no longer verifies. The release README notes it. `DATA.md` says the opposite: "every freeze hash verifies from this repository" (`release_files/DATA.md` l.22), which is false (README line).
4. **Two question-only figures without their item sets** (N3). l.376 "0.867 on two-label test items" is `recal_para` on the 1,148 v3 test items. l.324 "0.895 on the same items" is the 124 Stage 0b held-out items. Both are right, but l.376 sits in an argument about arm O, which was trained and tested on the 124.
5. **The provenance sentence is still loose** (N4). l.423 says the numbers were "typed into it, either by `paper/make\_numbers.py`". The script types nothing, and its docstring still says "nothing typed by hand".
6. **l.317 "mixed the two answers at about chance"** (N5). The other four checkpoints answered "Very sure" on 31–67% of natural items, with natural accuracy 0.45–0.53. "About chance" fits the accuracy, not the mix.
7. **Unchanged and fine.** All captions and cross-references agree with the text and the files:
   - Fig. 1 (11/12 boundary, dotted R-only line);
   - Fig. 2 (256/474 swaps; the 0.60 bar);
   - Fig. 3 (layer 12 omitted);
   - Fig. 4;
   - Tables 1 and 2;
   - the counts 24/62/124, 190/124 and 1,111/1,148.

   I found no internal numeric contradictions.
8. **Process nit.** The 2026-10-25 session rewrote `audit/paper_audit/recomputed_paper.json` inside the earlier auditor's output folder. The content is byte-identical, but such reproductions should go to a new folder.

## Task 4. Ranked items with replacement wording

### Blocking

**B1. Rebuild the PDF and the release bundle from the 14:40 source.** Run pdflatex, then bibtex, then pdflatex twice in `paper/`. Then re-run `paper9_ms/build_release.py`, so that the repo, the Zenodo PDF, the source zip and `code-and-data.zip` carry the current `main.tex`, `RESULTS.md` and `DEVIATIONS.md`. Then confirm that `main.pdf` is newer than `main.tex` and contains "looped model" and not "approved each stage". The current source builds cleanly (18 pages, no warnings) in `audit/paper_audit_final/build_check/`.

### Should fix

**S1. l.426, "Files."** Replace with:
> \paragraph{Files.} The study folder holds the preregistrations, code, deviation logs, per-item results and audits. They are released in the repository (\url{https://github.com/Somebodyhere101/self-report-tracks-input}). The eight result files larger than 2\,MB (\texttt{res\_v2/premise.json} and the seven \texttt{res\_v3/eval\_*.json}) are only in the Zenodo record (doi:\,[DOI]), which also archives the whole repository. Model weights, the probes' stored hidden-state features, job logs and the lab's GPU-lock wrapper are withheld, and the model's free-text answers in the item screens were removed; the repository's \texttt{DATA.md} lists each and how to rebuild it.

Fill in the DOI, or write "this Zenodo record" if the PDF goes only to Zenodo. Check that the GitHub URL resolves before posting.

**S2. Discussion, l.368, l.369 and l.383.** These still pick one of the two open routes.
- l.368:
  > \item \textbf{Multiplication.} Here the swap separates what the input implies from the part of the computation that set the output. The report positions carried, in linearly readable form, the confidence that the recipient's own input implies, not the swapped-in confidence that determined the model's actual output. This held at every layer probed, whether the question came 27 or 45 tokens after the answer.
- l.369:
  > \item \textbf{Relation to earlier results.} This is consistent with paper~8's \verdict{RECONSTRUCTION FROM INPUT}, now at the level of the report position's internal state, for a naturally occurring quantity whose input and answer are held fixed. It matches the injection literature \citep{ferrara2026owmi,hatua2026verbalization}.
- l.383:
  > \item \textbf{The inference view.} The result is consistent with it: the report positions carry what the question implies, not the swapped-in part of the process \citep{nisbett1977telling,carruthers2011opacity}. The design cannot rule out that they read the unswapped lower layers of the answer positions instead.

**S3. The failing freeze hash.**
- In the manuscript, add after l.410:
  > One hashed file no longer matches: the study's \texttt{README.md}, a file index with no decision rule, which was extended after the v1 freeze without a log entry at the time (logged in \texttt{DEVIATIONS.md}, 2026-10-25). Every other hashed file matches.
- In `release_files/DATA.md` l.22, replace with:
  > None of the withheld files is named in a `PREDICTIONS*.sha256` file. Every freeze hash verifies from this repository except the study README's line in `PREDICTIONS.sha256`; see the README, "Verifying the freezes".

**S4. Records** (they ship in the repository).
- New DEVIATIONS correction entry:
  > The 2026-10-22 14:45 entry's count is wrong: the manuscript audit listed 32 issues (4 major, 11 moderate, 17 minor), not 4 and 27, and issue 32 (a plain summary before the abstract's numbered items) was not addressed.
- RESULTS.md l.273:
  > The swapped-in confidence is no longer linearly readable from the first token after the digits. Donor agreement is 0.63–0.66 at the last digit, but 0.33–0.36 one token later, and 0.30–0.42 across the whole immediate report, at layers 16, 20, 24 and 27.
- STAGE0B.md l.84:
  > **Both report positions follow the question, not the swapped state.** In the untrained model, the state 27 or 45 tokens after the digits follows the confidence the question implies, not the swapped-in one (linear probes). The design cannot tell recomputation from the question from reading the unswapped lower layers of the answer positions.

### Nits

- **N1. l.26** (optional):
  > Self-Report Positions in Qwen3-1.7B Linearly Track the Input, Not the Swapped-In Computation: ...
- **N2. l.324:**
  > At the report positions, the recipient's own label is linearly readable from its unswapped state: 0.67--0.72 at the immediate report, the format arm~O was trained on, and 0.69--0.77 at Stage~0a's delayed report.

  Raw values: 0b immediate C 0.672–0.723; 0a C 0.688–0.773.
- **N3. l.376:**
  > A question-only predictor reaches 0.895 on Stage~0b's 124 held-out items (0.867 on the 1,148 v3 test items with two labels), and the input-implied label is linearly readable at the report position.
- **N4. l.423:**
  > The numbers in this manuscript were typed in from \texttt{numbers.json}, which \texttt{paper/make\_numbers.py} generates from the result files, or from the files named in the text.

  Also change the script docstring "nothing typed by hand" to "every number the manuscript takes from it".
- **N5. l.317:**
  > Twelve of the 16 checkpoints answered almost constantly, giving the same answer on at least 87\% of natural items; the other four answered ``Very sure'' on 31--67\% of natural items, with natural accuracy 0.45--0.53.
- **N6. l.45:** "the absence of a readable route" → "the absence of a linearly readable route".
- **N7. l.300:** "A trained route would have to be built, not tapped." → "A trained route would probably have to be built rather than tapped."
- **N8. Abstract (issue 32, optional).** Before "(1)", add:
  > In short: in the task where the two could be separated, the positions where the model reports its confidence follow what the question implies, not the confidence the model's computation was made to produce, and training did not change that.
- **N9. `references.bib`.** Move `% ---- language models` (l.429) and the Carruthers note (l.455–456) next to the entries they describe.
- **N10. Process.** Reproduce audit outputs into a new folder, not into an earlier auditor's folder.

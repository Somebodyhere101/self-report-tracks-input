"""Paper 9 manuscript audit: build the number table (markdown) from the recomputed values.
Status: match (printed value = raw value rounded half-up at the printed precision); rounding (printed value within half a
unit of the raw value, e.g. 'about 8'); MISMATCH (printed digit wrong); CLAIM (number right or not a number, but the
accompanying claim is wrong or overstated: see the issue list in AUDIT_paper.md).
Inputs: recomputed_paper.json, extra_checks_paper.json (this folder) and two earlier audit files.
Prints the table to stdout and the counts to stderr."""
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
P9 = os.path.dirname(os.path.dirname(HERE))
R = json.load(open(os.path.join(HERE, "recomputed_paper.json"), encoding="utf-8"))
X = json.load(open(os.path.join(HERE, "extra_checks_paper.json"), encoding="utf-8"))
CHK = json.load(open(os.path.join(P9, "audit", "v3_checks.json"), encoding="utf-8"))
SIG = json.load(open(os.path.join(P9, "audit", "v3_arm_I_signal.json"), encoding="utf-8"))
v1, v2, v3, s0, t0, tp = R["v1"], R["v2"], R["v3"], R["stage0"], R["stage0b_train"], R["tailpos"]
S = s0["summary"]
rows = []


def rhu(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def fmt(x):
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in x) + "]"
    if isinstance(x, float):
        return f"{x:.6g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.3g}"
    return str(x)


def add(loc, paper, raw, src, status=None, note=""):
    """paper: printed value(s) as strings (or a descriptive string); raw: matching raw value(s)."""
    if status is None:
        st = "match"
        pv = paper if isinstance(paper, list) else [paper]
        rv = raw if isinstance(raw, list) else [raw]
        for p, r in zip(pv, rv):
            s = str(p)
            d = len(s.split(".")[1]) if "." in s else 0
            if rhu(r, d) == float(s):
                continue
            if abs(float(r) - float(s)) <= 0.5 * 10 ** (-d) + 1e-12:
                st = "rounding" if st == "match" else st
            else:
                st = "MISMATCH"
        status = st
    shown = paper if isinstance(paper, str) else " / ".join(str(p) for p in paper)
    rows.append((loc, shown, fmt(raw), src, status, note))


cur = lambda k, B: v2["curves"][k][str(B)]["donor"]  # noqa: E731
finals = [v["final"] for v in v3["trainlogs"].values()]
step25 = [v["step25"] for v in v3["trainlogs"].values()]
# ------------------------------------------------------------------ abstract
add("Abstract (1)", ["0.840", "0.795"], [cur("report", 11), cur("readout", 11)], "res_v2/premise.json B rows at B=11 (= analysis_v2.json B.*.11.donor)")
add("Abstract (1)", "200", v2["n_pairs"], "premise.json pairs")
add("Abstract (1)", ["0.025", "0.025"], [cur("report", 12), cur("readout", 12)], "premise.json B rows at B=12")
add("Abstract (1)", ["0.039", "-0.007", "0.085"], [v2["A"]["R_minus_prior"]["diff"]] + v2["A"]["R_minus_prior"]["ci"], "premise.json + res/screen.json; bootstrap seed 11 (analysis_v2 method)")
add("Abstract (1) 'the bridge country is held at the city's own tokens'", "claim", "-", "DEVIATIONS 23:56 correction 3; RESULTS §3 'The instrument'", "CLAIM", "bridge not measured (issue 8)")
add("Abstract (2)", "1148", v3["setup"]["n_test_correct"], "res_v3/screen.json correct test items (= gen.json n_test_items)")
add("Abstract (2) 'training loss at ln 4'", "ln 4", [min(finals), max(finals)], "res_v3/trainlog_*.json hist[-1].loss", "match")
add("Abstract (3) swap layer", "16", 16, "gen.json cfg.l1; stage0/rows.json l1")
add("Abstract (3)", ["0.61", "0.66"], S["R0_0b_L16plus"], "stage0b/probe_imm/stage0/probe.json R0_L{16..27}_K2 B2.acc (re-run exact)", note="ceiling by construction (issue 17)")
add("Abstract (3)", ["27", "45"], [X["tokens"]["imm_len"], X["tokens"]["del_len"]], "token count of v4lib.tail_imm / tail_del (cached tokenizer)")
add("Abstract (3) 'at every layer from 8 to 27'", "20 layers", "6 probed (8, 12, 16, 20, 24, 27)", "probe.json keys", "CLAIM", "issue 3")
add("Abstract (3)", ["0.35", "0.41"], S["rep_all_three_formats"], "union of rep_L*_K2 B2.acc over stage0, probe_imm, probe_del (re-run exact)")
add("Abstract (4)", ["900", "32"], [900, 32], "stage0b/s0b_train.json runs imm_*_r32_3e-4")
add("Abstract (4)", ["0.46", "0.52"], t0["ranges"]["balanced_all_runs"], "s0b_train.json acc_swap_balanced, all 16 checkpoints", note="immediate arm-I runs only: 0.469-0.518 (issue 16)")
add("Abstract (5) 'At layers 16-27'", "16-27", "16, 20, 24, 27 probed", "tailpos.json layers", "CLAIM", "issue 3")
add("Abstract (5)", ["0.63", "0.66"], tp["d4_L16plus"], "stage0b/tailpos.json B2_donor L{16..27}_p0 (re-run exact)")
add("Abstract (5)", ["0.33", "0.36"], tp["first_L16plus"], "tailpos.json L{16..27}_p1")
add("Abstract (5)", ["0.30", "0.42"], tp["all_tail_L16plus"], "tailpos.json L{16..27}_p1..27")
add("Abstract (5)", "27", tp["n_tail"], "tailpos.json positions minus d4")
add("Abstract, last sentence 'recompute confidence from the question'", "claim", "-", "engine: blocks 0-16 of later positions see the recipient's digit states", "CLAIM", "issue 4")
# ------------------------------------------------------------------ introduction
add("Intro, paper 8", "0.90", 0.90, "paper8_ms/main.tex l.201, l.213 (depth tau of (iv)@21 in the self-report rows)")
add("Intro, paper 8 'two forced-choice self-reports'", "2", 2, "paper8_ms/main.tex self-report table (arms 1 and 2)")
add("§2 Related work 'eight open models' (ferrara2026owmi)", "8", 8, "paper8_ms/main.tex l.81 (same description, from paper 8's checked bibliography)")
# ------------------------------------------------------------------ setting
add("Setting, blocks", "28", 28, "HF cache Qwen3-1.7B config.json num_hidden_layers")
add("Setting, '8 GB laptop GPU'", "8", 8, "PREDICTIONS.md l.40 (laptop RTX 5070, 8 GB)")
add("Setting, 'LoRA ... all 28 blocks'", "28", 28, "code/p9lib.py install_lora (every block, s2lib.LORA_TARGETS)")
# ------------------------------------------------------------------ 3.1
add("§3.1 v1 calibration", "0.116", v1["cal_frozen"], "res_pilot/pilot.json P1.per_item + res/screen.json (19/164)", "CLAIM", "metric misdescribed; 0.354 = 58/164 without the agreement filter (issue 6)")
add("§3.1 v1", "164", v1["n_known_like"], "known-like usable pilot cities")
add("§3.1 v1 threshold", "0.80", 0.80, "code/analyze.py GATE min_calib")
add("§3.1 v1 reliability", "0.309", v1["reliability"], "pilot.json per_item reliable (60/194)")
add("§3.1 v1", "194", v1["n_usable_pilot"], "pilot.json per_item")
add("§3.1 v1 threshold", "0.70", 0.70, "GATE min_reliab")
add("§3.1 v2", "306", v2["n_known"], "premise.json known")
add("§3.1 v2 R", ["0.353", "0.301", "0.408"], [v2["A"]["R"]["cal"]] + v2["A"]["R"]["ci"], "premise.json A.R picks; bootstrap seed 11")
add("§3.1 v2 prior", ["0.314", "0.261", "0.366"], [v2["A"]["prior"]["cal"]] + v2["A"]["prior"]["ci"], "premise.json A.prior")
add("§3.1 v2 R - prior", ["0.039", "-0.007", "0.085"], [v2["A"]["R_minus_prior"]["diff"]] + v2["A"]["R_minus_prior"]["ci"], "premise.json")
add("§3.1 v2 'up to about 8 points'", "8", 100 * v2["A"]["R_minus_prior"]["ci"][1], "upper CI bound x 100", "rounding", "8.5 points")
add("§3.1 v2 city+R", ["0.961", "0.938", "0.980"], [v2["A"]["city+R"]["cal"]] + v2["A"]["city+R"]["ci"], "premise.json A.city+R")
add("§3.1 v2 pairs", "200", v2["n_pairs"], "premise.json pairs", note="recipients drawn from 363 usable test cities (issue 23)")
add("§3.1 v2 report B=11 / B=12", ["0.840", "0.025"], [cur("report", 11), cur("report", 12)], "premise.json B rows")
add("§3.1 v2 readout B=11 / B=12", ["0.795", "0.025"], [cur("readout", 11), cur("readout", 12)], "premise.json B rows")
add("§3.1 v2 agreement", "0.9645", v2["agreement"], "premise.json report vs readout picks, 2,000 rows")
add("§3.1 v2 pair-resampled CI", ["0.951", "0.977"], v2["pair_cluster_ci_my_seed"], "my pair-cluster bootstrap (seed 2026); audit/recomputed_v1v2.json ci_pair_cluster", note="audit CI, not the preregistered row CI [0.956, 0.973]")
add("§3.1 v2 kappa", "0.741", X["kappa"]["kappa_role"], "chance = 0.863 from per-B donor/recipient/other rates", note="option-level kappa = 0.952 (issue 21)")
add("§3.1 v2 R-only swap", ["0.020", "0.945"], [v2["C"]["donor"], v2["C"]["recipient"]], "premise.json report_Rswap rows (n = 200)")
add("§3.1 'up to about layer 11'", "11", 11, "transition at B = 12 (premise.json)")
add("Fig. 1 caption", ["200", "11", "12"], [200, 11, 12], "fig_v2_window.pdf; make_numbers.py fig. 1 (analysis_v2 donor, donor_ci; C donor line)")
# ------------------------------------------------------------------ 3.2
add("§3.2 candidates", "7", 7, "TASK_SEARCH.md rows 1-7")
add("§3.2 items", "3339", v3["setup"]["n_items_formula"], "count of 11<=a<=b<=99 with a 4-digit product; screen.json has 3,339")
add("§3.2 cut points p2", ["0.982", "0.9987", "0.99993"], v3["setup"]["p2_cuts"], "exp(quartiles of lp2 over correct train items) = exp(gen.json th)")
add("§3.2 swap layer", "16", v3["cfg"]["l1"], "gen.json cfg.l1")
add("§3.2 different-bin test swaps", "412", v3["gates"]["G4_n"], "gen.json T1 diff rows; donor bin carried on 1.00, digits kept 1.00")
add("§3.2 split", ["1111", "1148"], [v3["setup"]["n_train_correct"], v3["setup"]["n_test_correct"]], "screen.json correct items by split", note="of 1,327 / 1,329 items (issue 24)")
add("§3.2 training", ["8", "300", "8", "0.0001", "2"], [8, v3["cfg"]["steps"], v3["cfg"]["bs"], v3["cfg"]["lr"], len(v3["cfg"]["seeds"])], "code_v3/config_base.json lora_rank; gen.json cfg")
add("§3.2 'within about 25 steps'", "25", [min(step25), max(step25)], "trainlog step-25 losses (1.27-1.58)", "match")
add("§3.2 final losses", ["1.35", "1.44"], [min(finals), max(finals)], "trainlog_*.json hist[-1].loss (X1 1.3548, I0 1.4380)", note="numbers.json 1.355 would double-round to 1.36; text is right")
add("§3.2 ln 4", "1.386", v3["ln4"], "math")
add("§3.2 median top-letter probability", ["0.26", "0.31"], [min(v for k, v in v3["median_top_letter_prob"].items() if k != "N"), max(v for k, v in v3["median_top_letter_prob"].items() if k != "N")], "eval_*.json T3 lp: median of max letter probability (trained arms incl. X)")
# ------------------------------------------------------------------ Table 1
T1P = {"N": (["0.272"], ["-0.325", "-0.368", "-0.281"], ["0.277", "0.233", "0.318", "0.217"], ["0.236", "0.202", "0.272"]),
       "O": (["0.233", "0.234"], ["-0.363", "-0.397", "-0.328"], ["0.242", "0.210", "0.274", "0.226"], ["0.263", "0.237", "0.289"]),
       "I": (["0.249", "0.229"], ["-0.358", "-0.393", "-0.322"], ["0.301", "0.267", "0.335", "0.229"], ["0.280", "0.254", "0.306"]),
       "J": (["0.235", "0.251"], ["-0.354", "-0.389", "-0.318"], ["0.212", "0.181", "0.245", "0.217"], ["0.282", "0.255", "0.309"])}
for arm, (acc, dl, t1, t2) in T1P.items():
    T3, T1, T2 = v3["T3"][arm], v3["T1"][arm], v3["T2"][arm]
    add(f"Table 1 {arm} T3 acc", acc, T3["acc_per_seed"], "eval_*.json T3 picks vs gen.json labels")
    add(f"Table 1 {arm} T3 Δ [CI]", dl, [T3["delta"]] + T3["ci"], "re-implemented code/analyze.boot_ci (seed 9; P* re-chosen per resample)")
    add(f"Table 1 {arm} T1 diff [CI], same", t1, [T1["acc_diff"]] + T1["ci_diff"] + [T1["acc_same"]], "eval_*.json T1 (412 diff, 345 same rows), seed-averaged")
    add(f"Table 1 {arm} T2 pooled [CI]", t2, [T2["pooled"]] + T2["pooled_ci"], "eval_*.json T2a + T2c changed rows; recipient-clustered bootstrap")
add("Table 1 input-only", ["0.597", "0.597", "0.245", "0.238"], [v3["baselines"][k] for k in ("para", "learned", "external_X", "majority")], "gen.json para_bin; own conditional-logit fit (= para on every item); X seeds' mean scores; train majority")
add("Table 1 caption", ["0.25", "1148", "0.597", "22", "335", "18", "312"], [0.25, 1148, v3["baselines"]["para"], 22, v3["T2"]["I"]["n"]["T2a"], 18, v3["T2"]["I"]["n"]["T2c"]], "gen.json cfg ls/la; T2 changed-row counts")
add("Table 1 caption 'two untrained families'; 'averaged over two seeds'", "claim", "T2c = arm J's training family; T3 column is per seed", "gen.json train.J (bin vectors at la = 18, s_add = 2) = tests.T2c", "CLAIM", "issue 9")
# ------------------------------------------------------------------ exploratory signal
pm = v3["permutation"]
add("§3.2 permutations", "5000", 5000, "audit/v3_checks.py (5,000 label permutations)")
add("§3.2 signal, T1 diff", ["0.301", "0.253"], [pm["T1 diff:I"]["acc"], pm["T1 diff:I"]["independence"]], "eval_I_s*.json; independence = sum_b P(pick=b) P(label=b)")
add("§3.2 signal, T1 diff p", "0.0006", CHK["independence"]["T1 diff:I"]["perm_p"], "audit/v3_checks.json (Monte Carlo); my re-run 0.0004", "match")
add("§3.2 signal, n", ["412", "312"], [pm["T1 diff:I"]["n"], pm["T2c changed:I"]["n"]], "row counts")
add("§3.2 signal, bin vector", ["0.322", "0.270"], [pm["T2c changed:I"]["acc"], pm["T2c changed:I"]["independence"]], "eval_I_s*.json T2c changed rows")
add("§3.2 signal, bin vector p", "0.0012", CHK["independence"]["T2c changed:I"]["perm_p"], "audit/v3_checks.json; my re-run 0.0006", "match")
add("§3.2 'Bonferroni over the 20 ... tests that were run'", "20", "20 pooled + 36 per-seed tests were run", "v3_checks.json (20); v3_arm_I_signal.json (36)", "CLAIM", "issue 12")
add("§3.2 per seed (seed 1)", ["0.311", "0.0006"], [pm["T1 diff:I1"]["acc"], SIG["I1 T1 diff"][2]], "audit/v3_arm_I_signal.json; my re-run p 0.0002")
add("§3.2 per seed (seed 0)", ["0.291", "0.11"], [pm["T1 diff:I0"]["acc"], SIG["I0 T1 diff"][2]], "audit/v3_arm_I_signal.json; my re-run p 0.110")
add("§3.2 rank correlation", "0.91", v3["spearman_lp2_para"], "screen.json lp2 vs para_lp2 on the 1,148 T3 items (Spearman)")
add("§3.2 recalibrated paraphrase", "0.718", v3["recal_para_acc"], "screen.json para_lp2 binned at its own train quartiles, vs T3 labels")
# ------------------------------------------------------------------ 3.3
r0 = s0["rows_0a"]
add("§3.3 rows", ["590", "474", "256"], [r0["n_nat"], r0["n_swap"], r0["n_diff2"]], "stage0/rows.json (same in probe_imm and probe_del rows.json)")
add("§3.3 median p2", "0.9987", r0["p2_med"], "exp(rows.json med)")
add("§3.3 exact swap; digits kept", "exact; all", f"max abs(lp2 - donor) = {r0['swap_lp2_eq_donor_maxabs']}; kept {r0['digits_kept']}", "rows.json lp2_meas vs lp2_y; correct_meas (all three extractions)", "match")
add("§3.3 first digit, layers 16+", ["0.61", "0.66"], S["R0_0b_L16plus"], "probe_imm R0_L{16..27}_K2 B2 (Stage 0a: also 0.609-0.660)")
add("§3.3 first digit, layers 8, 12", ["0.34", "0.38"], S["R0_0b_L8_12"], "probe_imm R0_L8/L12_K2 B2 (Stage 0a: 0.348-0.367)")
add("§3.3 first digit, layer 27, all rows", ["0.74", "0.43"], [S["R0_0b_L27_A_K2"], S["R0_0b_L27_A_K4"]], "probe_imm R0_L27 A.acc (re-run exact)", note="Stage 0a: 0.737 / 0.441; STAGE0.md and PREDICTIONS_V4 say 0.44")
add("§3.3 report positions", ["0.35", "0.41"], S["rep_all_three_formats"], "rep_L*_K2 B2 over the three formats")
add("§3.3 'at every layer from 8 to 27'", "20 layers", "6 probed", "probe.json", "CLAIM", "issue 3")
add("§3.3 'the turn-2 report used in v3 (Stage 0a)'", "claim", "v4 two-option prompt, one option order", "stage0/s0a_extract.py", "CLAIM", "issue 10")
add("§3.3 token distances", ["45", "27"], [X["tokens"]["del_len"], X["tokens"]["imm_len"]], "token count (cached tokenizer)")
add("§3.3 'with four labels ... majority level'", "majority", [S["rep_K4_A_imm"], S["rep_K4_A_del"], S["rep_K4_A_majority"]], "probe_imm/del rep_L*_K4 A.acc vs A.majority 0.295", "match", "0.31-0.33 against 0.30: 'near'")
add("§3.3 pitfall, swapped-only probe", ["0.68", "0.74"], S["rep_0a_B1"], "stage0/probe.json rep_L*_K2 B1.acc (re-run exact)")
add("§3.3 pitfall, input-only probe", ["0.69", "0.77"], S["rep_0a_C"], "stage0/probe.json rep_L*_K2 C_input_probe.acc (re-run exact)")
add("Fig. 2 caption", ["256", "474", "0.60"], [r0["n_diff2"], r0["n_swap"], 0.60], "rows.json; DEVIATIONS 12:33 bar")
add("Fig. 2 caption 'Below the dotted chance line the probe follows the recipient'", "claim", "four-label report curves 0.255-0.310, above 0.25", "probe_*.json rep_L*_K4 B2", "CLAIM", "issue 11")
# ------------------------------------------------------------------ 3.4
add("§3.4 training / held-out items", ["190", "124"], [t0["Pstar_A2_recomputed"]["n_A1"], t0["Pstar_A2_recomputed"]["n_A2"]], "re-derived from screen.json with seeded('v4-half') and seeded('s0b-split')")
add("§3.4 arm I settings", ["8", "0.0001", "0.001", "300", "32", "0.0003", "900"], [8, 0.0001, 0.001, 300, 32, 0.0003, 900], "s0b_train.json runs", "CLAIM", "rank-32 run was immediate only (issue 7)")
add("§3.4 'Arm O, immediate format: the same settings'", "claim", "O: rank 8 at lr 1e-3 only, plus rank 32", "s0b_train.json runs", "CLAIM", "issue 7")
add("§3.4 'Every one collapsed to a constant answer or flipped'", "claim", "4 of 16 checkpoints mixed (share 0.31-0.67)", "s0b_train.json frac_pick0_nat", "CLAIM", "issue 19")
add("§3.4 'loss near ln 2'", "0.693", [min(e["final_loss"] for r in t0["runs"].values() for e in r.values()), max(e["final_loss"] for r in t0["runs"].values() for e in r.values())], "s0b_train.json loss_hist (final losses 0.580-0.781)", "match")
add("§3.4 balanced accuracy", ["0.46", "0.52"], t0["ranges"]["balanced_all_runs"], "s0b_train.json acc_swap_balanced, all checkpoints")
add("§3.4 arm O natural, maximum", "0.573", t0["ranges"]["O_nat_max"], "s0b_train.json acc_nat (arm O)")
add("§3.4 majority answer", "0.573", t0["Pstar_A2_recomputed"]["majority"], "re-derived from screen.json (A1 majority bin 1, scored on A2)")
add("§3.4 question-only predictors", "0.895", t0["Pstar_A2_recomputed"]["recal_para"], "re-derived recal_para on A2 (file: learned 0.895 too)")
add("§3.4 largest budget", ["32", "900", "8"], [32, 900, 8], "s0b_train.json; trainlib_v4 batch 8")
# ------------------------------------------------------------------ Table 2
N0 = t0["N"]
add("Table 2 untrained balanced imm / del", ["0.506", "0.487"], [N0["imm"]["acc_swap_balanced"], N0["del"]["acc_swap_balanced"]], "s0b_train.json N")
add("Table 2 untrained natural imm / del", ["0.347", "0.419"], [N0["imm"]["acc_nat"], N0["del"]["acc_nat"]], "s0b_train.json N")
add("Table 2 untrained first-option share", "--", [N0["imm"]["frac_pick0_nat"], N0["del"]["frac_pick0_nat"]], "s0b_train.json N frac_pick0_nat", "CLAIM", "values exist, 0.53 / 0.96 (issue 13)")
T2P = {"imm_I_r8_1e-3": (["0.510", "0.501"], ["0.556", "0.452"], ["0.03", "0.93"], ["0.665", "0.707"]),
       "imm_I_r8_1e-4": (["0.518", "0.474"], ["0.573", "0.427"], ["0.00", "0.97"], ["0.623", "0.699"]),
       "del_I_r8_1e-3": (["0.463", "0.498"], ["0.532", "0.435"], ["0.43", "0.88"], ["0.683", "0.781"]),
       "del_I_r8_1e-4": (["0.518", "0.482"], ["0.573", "0.427"], ["0.00", "1.00"], ["0.649", "0.697"]),
       "imm_O_r8_1e-3": (["0.462", "0.518"], ["0.516", "0.573"], ["0.32", "0.00"], ["0.580", "0.735"]),
       "imm_I_r32_3e-4": (["0.501", "0.489", "0.469"], ["0.460", "0.452", "0.419"], ["0.90", "0.67", "0.99"], ["0.764", "0.723", "0.669"]),
       "imm_O_r32_3e-4": (["0.518", "0.518", "0.476"], ["0.573", "0.573", "0.500"], ["0.00", "0.00", "0.57"], ["0.740", "0.694", "0.700"])}
for name, (bal, nat, fo, fl) in T2P.items():
    ev = t0["runs"][name]
    steps = sorted(ev, key=int)
    add(f"Table 2 {name} steps", [str(s) for s in steps], [int(s) for s in steps], "s0b_train.json evals keys")
    add(f"Table 2 {name} balanced", bal, [ev[s]["acc_swap_balanced"] for s in steps], "acc_swap_balanced")
    add(f"Table 2 {name} natural", nat, [ev[s]["acc_nat"] for s in steps], "acc_nat")
    add(f"Table 2 {name} first-option share", fo, [ev[s]["frac_pick0_nat"] for s in steps], "frac_pick0_nat (share answered 'Very sure', both orders averaged)",
        note=("issue 5: 39/124 = 0.3145 prints as 0.31; 0.32 is a double rounding of numbers.json 0.315" if name == "imm_O_r8_1e-3" else ""))
    add(f"Table 2 {name} final loss", fl, [ev[s]["final_loss"] for s in steps], "loss_hist[-1].loss")
add("Table 2 caption", ["24", "62", "0.50", "124", "0.573", "0.895"], [24, 62, 0.50, 124, t0["Pstar_A2_recomputed"]["majority"], t0["Pstar_A2_recomputed"]["recal_para"]], "s0b_train.json n_diff / n_same / n_nat; re-derived P*")
add("Table 2 caption 'First-option share: ... answered with the first option'", "claim", "metric = share answered 'Very sure' (bin 0), both orders averaged", "stage0b/s0b_train.py score(); code_v4/trainlib_v4.eval_reports", "CLAIM", "issue 13")
add("Fig. 3 caption", ["8", "4", "2", "300", "32", "900"], [8, 4, 2, 300, 32, 900], "trainlog_*.json (8 runs); s0b_train.json")
add("Fig. 3 caption 'every run stays near ln 2'", "claim", "delayed runs start at 4.36; logged peaks up to 1.12", "s0b_train.json loss_hist", "CLAIM", "issue 19")
# ------------------------------------------------------------------ 3.5
add("§3.5 positions; swaps", ["27", "256"], [tp["n_tail"], tp["n_B"]], "tailpos.json")
add("§3.5 last digit, layers 16+", ["0.63", "0.66"], tp["d4_L16plus"], "tailpos.json L*_p0")
add("§3.5 last digit, layer 12", "0.36", tp["d4_L12"], "tailpos.json L12_p0 = 0.3594")
add("§3.5 first token after the digits", ["0.33", "0.36"], tp["first_L16plus"], "tailpos.json L*_p1 (0.3320-0.3555)", note="RESULTS, STAGE0B and DEVIATIONS say 0.33-0.35 (wrong there, issue 31)")
add("§3.5 first three positions", "0.39", tp["first3_L16plus"][1], "tailpos.json L*_p1..3 maximum (incl. layer 12: still 0.3906)")
add("§3.5 all 27 positions", ["0.30", "0.42"], tp["all_tail_L16plus"], "tailpos.json L16..27, p1..27")
add("§3.5 'at layers 16-27'", "16-27", "16, 20, 24, 27", "tailpos.json layers", "CLAIM", "issue 3")
add("§3.5 swap from layer 16; block 17", ["16", "17"], [16, 17], "code/p9lib.py (layer l = output of block l); paper8 patchlib2 forward hook on block outputs")
add("§3.5 'no position after the answer'", "claim", "immediate tail only (plus the last delayed token)", "tailpos.json; probe_del", "CLAIM", "issue 3")
add("Fig. 4 caption", ["256", "0.60"], [tp["n_B"], 0.60], "tailpos.json; rule text in Temp\\p9\\results_v4.py at 13:34:23", note="layer 12 not plotted (issue 20)")
# ------------------------------------------------------------------ discussion, limitations, preregistration, appendix
add("§4 token distances", ["27", "45"], [X["tokens"]["imm_len"], X["tokens"]["del_len"]], "token count")
add("§4 'at every layer'", "claim", "6 probed layers", "probe.json", "CLAIM", "issue 3")
add("§4 question-only predictors, two labels", "0.867", R["two_bin_design"]["recal_para_test"], "res_v4_design/design.json recal_para; re-derived from screen.json", note="an external predictor, not 'the report's own estimate' (issue 22)")
add("§4 'modestly linear'", "0.74", S["R0_0b_L27_A_K2"], "probe_imm R0_L27_K2 A")
add("§4 limitations, budgets", ["32", "900", "190", "8", "300", "3000"], [32, 900, 190, 8, 300, 3000], "s0b_train.json; gen.json cfg; PREDICTIONS_V4.md ladder (3,000 steps)")
add("§4 limitations, different-label swaps", "24", t0["runs"]["imm_I_r8_1e-3"]["150"]["n_diff"], "s0b_train.json n_diff")
add("§5 freezes", "3", 3, "PREDICTIONS.sha256, PREDICTIONS_V2.sha256, PREDICTIONS_V3.sha256 (+ .amend)", note="hashes cover documents and code, not data inputs (issue 25)")
add("§5 bars", ["0.60", "0.60", "3"], [0.60, 0.60, 3], "DEVIATIONS 12:33; RESULTS §9")
add("§5 Stage 0b result", ["0.35", "0.40", "0.39"], S["rep_B2_donor_imm"] + [tp["first3_all_layers_max"]], "probe_imm rep_L*_K2 B2; tailpos first three positions")
add("App. A screens", ["7", "6", "48"], [7, 6, 48], "TASK_SEARCH.md (item counts 6-48)", note="'six that failed' includes one 'not pursued' (issue 26)")
add("App. A candidate 7", "41/48; 47/48; 10/10", "41/48; 47/48; 10/10", "TASK_SEARCH.md row 7; diag/ rerun logs", "match")
add("AI-use section model id", "claude-opus-5-5", "claude-opus-5-5", "session model id", "match")

esc = lambda x: str(x).replace("|", "\\|")  # noqa: E731
numtok = re.compile(r"^-?\d[\d,]*(\.\d+)?$")
nnum = {}
print("| # | Location | Paper value | Recomputed (raw) | Source (path: field) | Status |")
print("|---|---|---|---|---|---|")
for i, (loc, p, r, s, st, note) in enumerate(rows, 1):
    st2 = st if not note else f"{st} ({note})"
    print(f"| {i} | {esc(loc)} | {esc(p)} | {esc(r)} | {esc(s)} | {esc(st2)} |")
    toks = [t for t in re.split(r"\s*/\s*|;\s*", str(p)) if numtok.match(t.strip())]
    nnum[st] = nnum.get(st, 0) + len(toks)
by = {}
for row in rows:
    by[row[4]] = by.get(row[4], 0) + 1
print(json.dumps({"rows": len(rows), "rows_by_status": by, "numeric_values_by_row_status": nnum, "numeric_values_total": sum(nnum.values())}), file=sys.stderr)

# Audit copy of paper/make_numbers.py with output redirected to audit/paper_audit_final/regen (study files untouched).
"""Paper 9 manuscript: every number and figure, read from the study's result files (nothing typed by hand).
Writes paper/numbers.json and paper/fig_*.pdf. CPU only. Run from anywhere: python paper/make_numbers.py"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regen")
P9 = 'C:/Users/<user>/AppData/Roaming/Claude/scratch-workspaces/2898f83d-79d0-4cb9-a017-7d323d549974/0c4963d8-51ed-4ceb-9445-28b8639f6fa1/scratch-2026-10-13-464f6a/research_next/p9_trained_introspection'


def load(*p):
    return json.load(open(os.path.join(P9, *p), encoding="utf-8"))


def r3(x):
    return None if x is None else round(float(x), 3)


N = {}
# ------------------------------------------------------------------ v1 bed check
g1 = load("res_pilot", "pilot_gates.json")["checks"]
N["v1"] = {"PG1_acc2": r3(g1["PG1_task"]["acc2_all"]), "PG1_n_test": g1["PG1_task"]["n_test_usable"],
           "PG2_calibration": r3(g1["PG2_readout"]["calibration_known"]), "PG2_reliability": r3(g1["PG2_readout"]["reliability"]),
           "PG3_transfer_l1_0": r3(g1["PG3_manipulation"]["transfer"])}
# ------------------------------------------------------------------ v2 premise test
v2 = load("res_v2", "analysis_v2.json")
Bs = ["4", "6", "8", "9", "10", "11", "12", "14", "16", "20"]
N["v2"] = {"n_known": v2["n_known"], "n_pairs": v2["n_pairs"],
           "A": {k: {"cal": r3(v2["A"][k]["calibration"]), "ci": [r3(c) for c in v2["A"][k]["ci"]]} for k in ("prior", "R", "city+R")},
           "A_R_minus_prior": {"diff": r3(v2["A"]["R_minus_prior"]["diff"]), "ci": [r3(c) for c in v2["A"]["R_minus_prior"]["ci"]]},
           "A_cityR_minus_R": {"diff": r3(v2["A"]["cityR_minus_R"]["diff"]), "ci": [r3(c) for c in v2["A"]["cityR_minus_R"]["ci"]]},
           "B_report_donor": {b: r3(v2["B"]["report"][b]["donor"]) for b in Bs},
           "B_readout_donor": {b: r3(v2["B"]["readout"][b]["donor"]) for b in Bs},
           "B_agreement": {"mean": round(v2["B_agreement"]["mean"], 4), "ci_rows": v2["B_agreement"]["ci"]},
           "transition": v2["transition"], "C": v2["C"], "verdict": v2["verdict"]}
ag = load("audit", "recomputed_v1v2.json")["v2"]["agreement"]  # the v1/v2 audit's recomputation
N["v2"]["B_agreement"].update({"ci_pair_cluster": ag["ci_pair_cluster"], "expected_if_independent": r3(ag["expected_if_independent_pooled"]),
                               "kappa_pooled": r3(ag["kappa_pooled"])})
# ------------------------------------------------------------------ v3 training study
v3 = load("res_v3", "analysis.json")
chk = load("audit", "v3_checks.json")
sig = load("audit", "v3_arm_I_signal.json")
N["v3"] = {"verdict": v3["verdict"], "reasons": v3["reasons"],
           "gates": {"G1_acc": r3(v3["gate_values"]["G1"]["accuracy_test"]), "n_test": v3["gate_values"]["G1"]["n_test_usable"],
                     "G2_kept": r3(v3["gate_values"]["G2"]["answer_kept_T1"]), "G3": r3(v3["gate_values"]["G3"]["bin_prob_vs_entropy_agreement"]),
                     "G4": r3(v3["gate_values"]["G4"]["T1_diff_bin_equals_donor"]), "G4_n": v3["gate_values"]["G4"]["n"],
                     "G5_n_div": v3["gate_values"]["G5"]["n_divergent"], "G5_power": v3["gate_values"]["G5"]["power_strong_reader"]},
           "T3_baselines": {k: r3(v) for k, v in v3["T3_baselines"].items()},
           "T3": {a: {"acc_per_seed": [r3(x) for x in v3["T3"][a]["acc_per_seed"]], "delta": r3(v3["T3"][a]["delta_vs_best_baseline"]),
                      "ci": [r3(c) for c in v3["T3"][a]["ci"]]} for a in ("N", "O", "I", "J")},
           "T1": {a: {"acc_diff": r3(v3["T1"][a]["acc_diff"]), "acc_same": r3(v3["T1"][a]["acc_same"]),
                      "ci_diff": [r3(c) for c in v3["T1"][a]["ci_diff"]]} for a in ("N", "O", "I", "J")},
           "T2": {a: {"T2a": r3(v3["T2"][a]["acc"]["T2a"]), "T2c": r3(v3["T2"][a]["acc"]["T2c"]), "pooled": r3(v3["T2"][a]["pooled_acc"]),
                      "ci": [r3(c) for c in v3["T2"][a]["pooled_ci"]], "n": v3["T2"][a]["n_changed"]} for a in ("N", "O", "I", "J")},
           "T4_I_kl": {"inc": r3(v3["T4"]["I"]["kl"]["switch_increase"]), "ci": [r3(c) for c in v3["T4"]["I"]["kl"]["ci"]]},
           "T5": {a: {k: r3(v) for k, v in d.items() if k != "drift_flag"} for a, d in v3["T5"].items()},
           "exploratory_I_signal_per_seed": {k: [r3(x) for x in v] for k, v in sig.items() if k.startswith("I") and ("diff" in k or "T2c changed" in k)},
           "recalibrated_para": chk.get("exploratory_recalibrated_para"),
           "exploratory_permutation_pooled": {k: {kk: ((round(vv, 4) if kk == "perm_p" else r3(vv)) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                              for k, v in chk["independence"].items()},
           "train_label_entropy": {k: r3(v["entropy"]) for k, v in chk["train_label_entropy"].items()},
           "max_letter_prob_median": {k: r3(v["max_letter_prob_median"]) for k, v in chk["spread"].items()}}
pg3 = load("res_pilot_v3", "pilot_gates.json")["checks"]
gen3 = load("res_v3", "gen.json")
scr3 = load("res_v3", "screen.json")["items"]
N["v3"]["setup"] = {"n_items": len(scr3), "screen_accuracy": r3(pg3["PG1_task"]["accuracy_all"]),
                    "n_train_items": gen3["n_train_items"], "n_test_items": gen3["n_test_items"],
                    "quartile_cuts_lp2": gen3["th"], "quartile_cuts_p2": [round(math.exp(t), 5) for t in gen3["th"]],
                    "PG3_transfer_l16": pg3["PG3_swap"]["transfer"], "PG3_digits_kept": pg3["PG3_swap"]["digits_kept"],
                    "test_bin_counts": [sum(1 for e in gen3["tests"]["T3"] if e["label"] == k) for k in range(4)]}
from scipy.stats import spearmanr  # noqa: E402
te3 = [scr3[e["x"]] for e in gen3["tests"]["T3"]]
N["v3"]["setup"]["spearman_lp2_vs_paraphrase_T3"] = r3(spearmanr([i["lp2"] for i in te3], [i["para_lp2"] for i in te3])[0])
losses = {}
for arm in "OIJX":
    for s in (0, 1):
        h = load("res_v3", f"trainlog_{arm}_s{s}.json")["hist"]
        losses[f"{arm}{s}"] = [(x["step"], x["loss"]) for x in h]
N["v3"]["train_loss_final"] = {k: r3(v[-1][1]) for k, v in losses.items()}
N["v3"]["ln4"] = r3(math.log(4))
# ------------------------------------------------------------------ Stage 0a (coordinating session) and 0b probes
def probe_rows(P, pos, K):
    out = {}
    for l in (8, 12, 16, 20, 24, 27):
        r = P["res"][f"{pos}_L{l}_K{K}"]
        out[l] = {"A": r3(r["A"]["acc"]), "B1": r3(r["B1"]["acc"]), "B2_donor": r3(r["B2"]["acc"]),
                  "B2_donor_ci": [r3(c) for c in r["B2"]["ci"]], "B2_recipient": r3(r["B2"]["agree_recipient"]), "C": r3(r["C_input_probe"]["acc"])}
    return out


p0a = load("stage0", "probe.json")
pim = load("stage0b", "probe_imm", "stage0", "probe.json")
pde = load("stage0b", "probe_del", "stage0", "probe.json")
N["stage0"] = {"0a_checks": {k: (r3(v) if isinstance(v, float) else v) for k, v in p0a["checks"].items()},
               "0b_checks_imm": {k: (r3(v) if isinstance(v, float) else v) for k, v in pim["checks"].items()},
               "0b_checks_del": {k: (r3(v) if isinstance(v, float) else v) for k, v in pde["checks"].items()}}
for K in (2, 4):
    N["stage0"][f"K{K}"] = {"0a_rep_delayed_v3R": probe_rows(p0a, "rep", K), "0a_R0": probe_rows(p0a, "R0", K),
                            "0b_rep_immediate": probe_rows(pim, "rep", K), "0b_rep_delayed": probe_rows(pde, "rep", K),
                            "0b_R0": probe_rows(pim, "R0", K)}


def span(d, key):
    v = [x[key] for x in d.values()]
    return [min(v), max(v)]


s2 = N["stage0"]["K2"]
N["stage0"]["summary_K2"] = {
    "0a_rep_B2_donor": span(s2["0a_rep_delayed_v3R"], "B2_donor"), "0a_rep_B2_recipient": span(s2["0a_rep_delayed_v3R"], "B2_recipient"),
    "0a_rep_C": span(s2["0a_rep_delayed_v3R"], "C"), "0a_rep_B1": span(s2["0a_rep_delayed_v3R"], "B1"),
    "imm_B2_donor": span(s2["0b_rep_immediate"], "B2_donor"), "imm_B2_recipient": span(s2["0b_rep_immediate"], "B2_recipient"),
    "del_B2_donor": span(s2["0b_rep_delayed"], "B2_donor"), "del_B2_recipient": span(s2["0b_rep_delayed"], "B2_recipient"),
    "R0_L16plus_B2_donor": span({l: s2["0b_R0"][l] for l in (16, 20, 24, 27)}, "B2_donor"),
    "R0_L8_12_B2_donor": span({l: s2["0b_R0"][l] for l in (8, 12)}, "B2_donor"),
    "R0_L27_A": s2["0b_R0"][27]["A"], "R0_L27_A_K4": N["stage0"]["K4"]["0b_R0"][27]["A"]}
# ------------------------------------------------------------------ Stage 0b training runs
t0b = load("stage0b", "s0b_train.json")
N["stage0b_train"] = {"n_A1": t0b["n_A1"], "n_A2": t0b["n_A2"], "Pstar_A2": {k: r3(v) for k, v in t0b["Pstar_A2_acc"].items()},
                      "N": {f: {k: r3(v) if isinstance(v, float) else v for k, v in d.items()} for f, d in t0b["N"].items()},
                      "runs": {}}
for name, run in t0b["runs"].items():
    N["stage0b_train"]["runs"][name] = {"fmt": run["fmt"], "arm": run["arm"], "rank": run["rank"], "lr": run["lr"],
                                        "evals": {S: {k: r3(ev[k]) for k in ("acc_diff", "acc_same", "acc_swap_balanced", "acc_nat", "frac_pick0_nat")}
                                                  | {"final_loss": r3(ev["loss_hist"][-1]["loss"])} for S, ev in run["evals"].items()}}
bal = [ev["acc_swap_balanced"] for run in N["stage0b_train"]["runs"].values() for ev in run["evals"].values() if run["arm"] == "I"]
nat_O = [ev["acc_nat"] for run in N["stage0b_train"]["runs"].values() for ev in run["evals"].values() if run["arm"] == "O"]
N["stage0b_train"]["summary"] = {"I_balanced_range": [min(bal), max(bal)], "O_nat_max": max(nat_O), "ln2": r3(math.log(2)),
                                 "largest": "rank 32, lr 3e-4, 900 steps x 8 examples (7,200 examples)"}
design = load("res_v4_design", "design.json")
N["two_bin_design"] = {"med": design["medians"]["med"], "acc": {k: r3(v) for k, v in design["T3_input_only"]["acc"].items()}}
# ------------------------------------------------------------------ tail-position decay (if present)
tp_path = os.path.join(P9, "stage0b", "tailpos.json")
if os.path.exists(tp_path):
    tp = json.load(open(tp_path, encoding="utf-8"))
    N["tailpos"] = {"positions": tp["positions"], "layers": tp["layers"], "n_B": tp["n_B"],
                    "B2_donor": {k: r3(v) for k, v in tp["B2_donor"].items()}, "B2_recipient": {k: r3(v) for k, v in tp["B2_recipient"].items()}}
    late = [l for l in tp["layers"] if l >= 16]
    npos = len(tp["positions"])
    d = lambda l, p: tp["B2_donor"][f"L{l}_p{p}"]  # noqa: E731
    N["tailpos"]["summary"] = {
        "d4_L12": r3(d(12, 0)), "d4_L16plus": [r3(min(d(l, 0) for l in late)), r3(max(d(l, 0) for l in late))],
        "first_after_digits_L16plus": [r3(min(d(l, 1) for l in late)), r3(max(d(l, 1) for l in late))],
        "first3_L16plus": [r3(min(d(l, p) for l in late for p in (1, 2, 3))), r3(max(d(l, p) for l in late for p in (1, 2, 3)))],
        "all_tail_L16plus": [r3(min(d(l, p) for l in late for p in range(1, npos))), r3(max(d(l, p) for l in late for p in range(1, npos)))],
        "all_tail_L12": [r3(min(d(12, p) for p in range(1, npos))), r3(max(d(12, p) for p in range(1, npos)))],
        "n_tail_positions": npos - 1}
json.dump(N, open(os.path.join(HERE, "numbers.json"), "w", encoding="utf-8"), indent=1)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
# Figure 1: v2 Part B
fig, ax = plt.subplots(figsize=(3.4, 2.3))
x = [int(b) for b in Bs]
for key, lab, mk in (("report", "untrained self-report", "o"), ("readout", "placeholder readout", "s")):
    y = [v2["B"][key][b]["donor"] for b in Bs]
    lo = [v2["B"][key][b]["donor_ci"][0] for b in Bs]
    hi = [v2["B"][key][b]["donor_ci"][1] for b in Bs]
    ax.plot(x, y, marker=mk, ms=3, lw=1, label=lab)
    ax.fill_between(x, lo, hi, alpha=0.2)
ax.axhline(v2["C"]["report_names_donor"], color="grey", ls=":", lw=1)
ax.text(19.8, v2["C"]["report_names_donor"] + 0.04, "R-only swap (report)", ha="right", fontsize=6.5, color="grey")
ax.set_xlabel("first swapped layer B")
ax.set_ylabel("share naming the donor's country")
ax.set_ylim(-0.03, 1.03)
ax.legend(frameon=False, fontsize=6.5, loc="center right")
ax.set_xticks([4, 6, 8, 10, 12, 14, 16, 20])
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_v2_window.pdf"))
plt.close(fig)
# Figure 2: probe agreement with the donor, by layer
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4))
L = [8, 12, 16, 20, 24, 27]
for ax, K in zip(axs, (2, 4)):
    S = N["stage0"][f"K{K}"]
    for key, lab, st in (("0b_R0", "first answer digit (inside R)", "-o"), ("0b_rep_immediate", "immediate report (27 tokens later)", "-s"),
                         ("0b_rep_delayed", "delayed report (45 tokens later)", "-^"), ("0a_rep_delayed_v3R", "delayed report, Stage 0a", "--x")):
        y = [S[key][l]["B2_donor"] for l in L]
        ax.plot(L, y, st, ms=3, lw=1, label=lab)
    ax.axhline(1 / K, color="grey", ls=":", lw=1)
    if K == 2:
        ax.axhline(0.6, color="red", ls="--", lw=0.7)
        ax.text(8, 0.612, "pre-freeze bar 0.60", ha="left", fontsize=6, color="red")
    ax.set_title(f"{K} bins", fontsize=8)
    ax.set_xlabel("layer")
    ax.set_ylabel("probe agrees with donor")
    ax.set_xticks(L)
hd, lb = axs[0].get_legend_handles_labels()
fig.legend(hd, lb, loc="lower center", ncol=4, fontsize=6, frameon=False)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(os.path.join(HERE, "fig_probe_layers.pdf"))
plt.close(fig)
# Figure 3: training losses (v3, four options; Stage 0b, two options)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.3))
for k, v in losses.items():
    axs[0].plot([s for s, _ in v], [l for _, l in v], lw=0.8, alpha=0.8)
axs[0].axhline(math.log(4), color="black", ls=":", lw=1)
axs[0].text(300, math.log(4) + 0.3, "ln 4 (uniform guess)", ha="right", fontsize=6.5)
axs[0].set_ylim(0, 8)
axs[0].set_title("v3: four bins, rank 8, 300 steps (8 runs)", fontsize=8)
for name, run in t0b["runs"].items():
    last = run["evals"][max(run["evals"], key=int)]
    h = last["loss_hist"]
    axs[1].plot([x["step"] for x in h], [x["loss"] for x in h], lw=0.8, alpha=0.8, label=name.replace("_", " "))
axs[1].axhline(math.log(2), color="black", ls=":", lw=1)
axs[1].text(880, math.log(2) + 0.15, "ln 2 (uniform guess)", ha="right", fontsize=6.5)
axs[1].set_ylim(0, 4.6)
axs[1].set_title("Stage 0b: two bins, up to rank 32 and 900 steps", fontsize=8)
for ax in axs:
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
axs[1].legend(frameon=False, fontsize=5.5, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_training.pdf"))
plt.close(fig)
# Figure 4: tail-position decay
if "tailpos" in N:
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    pos = N["tailpos"]["positions"]
    for l in N["tailpos"]["layers"]:
        if l < 16:
            continue
        y = [N["tailpos"]["B2_donor"][f"L{l}_p{p}"] for p in range(len(pos))]
        ax.plot(range(len(pos)), y, "-o", ms=2, lw=0.9, label=f"layer {l}")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.hlines(0.6, 0.6, 3.4, color="red", ls="--", lw=0.9)  # the bar applied to the first three positions only
    ax.text(3.6, 0.6, "bar 0.60 (positions 1-3)", va="center", fontsize=6, color="red")
    ax.set_xticks(range(len(pos)))
    ax.set_xticklabels([repr(t)[1:-1] if i else "d4" for i, t in enumerate(pos)], rotation=90, fontsize=5)
    ax.set_ylabel("probe agrees with donor")
    ax.set_xlabel("position: last answer digit (inside R), then each token of the immediate report")
    ax.legend(frameon=False, fontsize=6, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_tailpos.pdf"))
    plt.close(fig)
print(json.dumps({"v2_B_report": N["v2"]["B_report_donor"], "stage0_summary_K2": N["stage0"]["summary_K2"],
                  "stage0b_summary": N["stage0b_train"]["summary"], "tailpos": "tailpos" in N}, indent=1))

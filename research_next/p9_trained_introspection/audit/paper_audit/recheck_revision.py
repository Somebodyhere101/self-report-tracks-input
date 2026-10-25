"""Paper 9 manuscript audit, re-check of the revision (2026-10-22 14:44 build): recompute every number that is new or
changed in the revised main.tex from the raw result files. CPU, 1 thread, numpy/scipy only, no study module imported.
Writes audit/paper_audit/recheck_revision.json and prints a summary."""
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
P9 = os.path.dirname(os.path.dirname(HERE))


def load(*p):
    with open(os.path.join(P9, *p), encoding="utf-8") as f:
        return json.load(f)


OUT = {}
# ---------------------------------------------------------------- v1: 0.354 and PG3 0.754
pil = load("res_pilot", "pilot.json")
scr1 = {i["city"]: i for i in load("res", "screen.json")["items"]}
per = pil["P1"]["per_item"]
known = [c for c in per if scr1[c]["p1"] == scr1[c]["country"] and scr1[c].get("free1") == scr1[c]["country"]]
OUT["v1"] = {"cal_without_agreement": sum(per[c]["cb_country"] == scr1[c]["country"] for c in known) / len(known),
             "n_known": len(known), "PG3_from0": pil["P2"]["from0"], "PG3_gates": load("res_pilot", "pilot_gates.json")["checks"]["PG3_manipulation"]}
# ---------------------------------------------------------------- v2: 363, R-prior upper bound, kappa chance
prem = load("res_v2", "premise.json")
OUT["v2"] = {"n_test_usable": prem["n_test"], "R_minus_prior_ci_upper": load("res_v2", "analysis_v2.json")["A"]["R_minus_prior"]["ci"][1],
             "kappa": json.load(open(os.path.join(HERE, "extra_checks_paper.json"), encoding="utf-8"))["kappa"]}
# recipients: how many of the 200 recipients are known-like, and are all donors known-like
kn = set(prem["known"])
OUT["v2"]["recipients_known_like"] = sum(x in kn for x, y in prem["pairs"])
OUT["v2"]["donors_known_like"] = sum(y in kn for x, y in prem["pairs"])
# ---------------------------------------------------------------- v3: ties, Bonferroni, per-seed
ties = {}
for a in "OIJX":
    for s in (0, 1):
        ev = load("res_v3", f"eval_{a}_s{s}.json")
        sc = np.array([r["scores"] for r in ev["T3"]])
        mx = sc.max(1, keepdims=True)
        tied = (sc == mx).sum(1) > 1
        pick = np.array([r["pick"] for r in ev["T3"]])
        first_max = np.argmax(sc == mx, axis=1)
        ties[f"{a}{s}"] = {"tied_share": float(tied.mean()), "pick_is_first_max_on_ties": bool((pick[tied] == first_max[tied]).all())}
OUT["v3_ties"] = ties
OUT["v3_ties_range_OIJ"] = [min(v["tied_share"] for k, v in ties.items() if k[0] in "OIJ"), max(v["tied_share"] for k, v in ties.items() if k[0] in "OIJ")]
OUT["v3_ties_range_OIJX"] = [min(v["tied_share"] for v in ties.values()), max(v["tied_share"] for v in ties.values())]
chk = load("audit", "v3_checks.json")["independence"]
sig = load("audit", "v3_arm_I_signal.json")
OUT["bonferroni"] = {"n_pooled": len(chk), "n_per_seed": len(sig), "n_total": len(chk) + len(sig),
                     "thr_20": 0.05 / len(chk), "thr_56": 0.05 / (len(chk) + len(sig)),
                     "p_T1": chk["T1 diff:I"]["perm_p"], "p_T2c": chk["T2c changed:I"]["perm_p"],
                     "T1_survives_56": chk["T1 diff:I"]["perm_p"] < 0.05 / (len(chk) + len(sig)),
                     "T2c_survives_56": chk["T2c changed:I"]["perm_p"] < 0.05 / (len(chk) + len(sig))}
OUT["per_seed"] = {k: sig[k] for k in ("I0 T1 diff", "I1 T1 diff", "I0 T2c changed", "I1 T2c changed")}
# ---------------------------------------------------------------- Stage 0a / 0b probe values cited anew
p0a = load("stage0", "probe.json")["res"]
pim = load("stage0b", "probe_imm", "stage0", "probe.json")["res"]
pde = load("stage0b", "probe_del", "stage0", "probe.json")["res"]
OUT["stage0a_R0_L27_A"] = {"K2": p0a["R0_L27_K2"]["A"]["acc"], "K4": p0a["R0_L27_K4"]["A"]["acc"]}
cells = []
for name, P in (("0a", p0a), ("imm", pim), ("del", pde)):
    for l in (8, 12, 16, 20, 24, 27):
        b = P[f"rep_L{l}_K4"]["B2"]
        cells.append(b["agree_recipient"] - b["acc"])
OUT["K4_report_cells_recipient_minus_donor"] = {"n": len(cells), "all_positive": all(c > 0 for c in cells), "min": min(cells), "max": max(cells)}
OUT["C_input_probe_rep_K2"] = {n: [min(P[f"rep_L{l}_K2"]["C_input_probe"]["acc"] for l in (8, 12, 16, 20, 24, 27)),
                                   max(P[f"rep_L{l}_K2"]["C_input_probe"]["acc"] for l in (8, 12, 16, 20, 24, 27))] for n, P in (("0a", p0a), ("imm", pim), ("del", pde))}
# ---------------------------------------------------------------- Stage 0b training: ranges, near-constant checkpoints, losses
t0 = load("stage0b", "s0b_train.json")
ck, immI = [], []
for name, run in t0["runs"].items():
    for S, ev in run["evals"].items():
        share = ev["frac_pick0_nat"]
        ck.append({"run": name, "step": int(S), "very_sure_share": share, "same_answer_share": max(share, 1 - share), "acc_nat": ev["acc_nat"],
                   "n_same_answer": round(max(share, 1 - share) * ev["n_nat"]), "n_nat": ev["n_nat"]})
        if run["arm"] == "I" and run["fmt"] == "imm":
            immI.append(ev["acc_swap_balanced"])
ck.sort(key=lambda c: -c["same_answer_share"])
OUT["checkpoints"] = ck
near = [c for c in ck if c["same_answer_share"] >= 0.85]
mixed = [c for c in ck if c["same_answer_share"] < 0.85]
OUT["near_constant"] = {"n": len(near), "min_same_answer_share": min(c["same_answer_share"] for c in near),
                        "min_count": f"{min(c['n_same_answer'] for c in near)}/{near[0]['n_nat']}",
                        "n_at_least_0.88": sum(c["same_answer_share"] >= 0.88 for c in ck)}
OUT["mixed"] = {"n": len(mixed), "acc_nat": [c["acc_nat"] for c in mixed], "shares": [c["very_sure_share"] for c in mixed]}
OUT["balanced_immediate_I"] = [min(immI), max(immI)]
loss = {}
for name, run in t0["runs"].items():
    h = run["evals"][max(run["evals"], key=int)]["loss_hist"]
    loss[name] = [(x["step"], round(x["loss"], 3)) for x in h]
    # first logged step after which every logged loss is within 0.15 of ln 2
    ok = None
    for i in range(len(h)):
        if all(abs(x["loss"] - math.log(2)) <= 0.15 for x in h[i:]):
            ok = h[i]["step"]
            break
    loss[name + "_settled_within_0.15_from_step"] = ok
OUT["s0b_loss"] = loss
# ---------------------------------------------------------------- untrained immediate report: 19/53 and 47/71
N = t0["N"]["imm"]
labels = np.array([r["label"] for r in t0["rowsO"]])
n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
n = len(labels)
picks_vs = round(N["frac_pick0_nat"] * n)
correct = round(N["acc_nat"] * n)
# correct = a + (n1 - (picks_vs - a))  ->  a = (correct - n1 + picks_vs) / 2
a = (correct - n1 + picks_vs) / 2
table = [[int(a), n0 - int(a)], [picks_vs - int(a), n1 - (picks_vs - int(a))]]  # rows: label VS, label LS; cols: pick VS, pick LS
OUT["untrained_imm"] = {"n": n, "label_VS": n0, "label_LS": n1, "picks_VS": picks_vs, "correct": correct,
                        "VS_items_answered_VS": table[0][0], "LS_items_answered_VS": table[1][0], "acc_nat": N["acc_nat"],
                        "independence_level": (picks_vs / n) * (n0 / n) + (1 - picks_vs / n) * (n1 / n),
                        "fisher_two_sided_p": float(fisher_exact(table)[1]), "frac_pick0_nat": N["frac_pick0_nat"],
                        "consistency": abs(N["frac_pick0_nat"] * n - picks_vs) < 1e-9 and abs(N["acc_nat"] * n - correct) < 1e-9}
OUT["untrained_rows"] = {f: {k: t0["N"][f][k] for k in ("acc_swap_balanced", "acc_nat", "frac_pick0_nat")} for f in ("imm", "del")}
with open(os.path.join(HERE, "recheck_revision.json"), "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps({k: v for k, v in OUT.items() if k not in ("checkpoints", "s0b_loss", "v3_ties")}, indent=1, default=str))
print("ties", {k: round(v["tied_share"], 3) for k, v in OUT["v3_ties"].items()}, "first-max rule", all(v["pick_is_first_max_on_ties"] for v in OUT["v3_ties"].values()))
for c in OUT["checkpoints"]:
    print(c["run"], c["step"], round(c["same_answer_share"], 4), c["n_same_answer"], round(c["acc_nat"], 3))
for k, v in OUT["s0b_loss"].items():
    print(k, v)

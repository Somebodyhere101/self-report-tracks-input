"""Paper 9 manuscript audit (independent): recompute the manuscript's numbers from the raw result files.
CPU only, numpy/scipy, no study module imported (helpers re-implemented here). Reads only P9 result files.
Writes audit/paper_audit/recomputed_paper.json. Run: python recompute_paper.py"""
import hashlib
import json
import math
import os
import random
from decimal import ROUND_HALF_UP, Decimal

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
P9 = os.path.dirname(os.path.dirname(HERE))


def load(*p):
    with open(os.path.join(P9, *p), encoding="utf-8") as f:
        return json.load(f)


def seeded(*keys):  # p9lib.seeded, re-implemented
    h = hashlib.sha256("|".join(str(k) for k in keys).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def rhu(x, d):  # round half up (what a reader expects)
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


OUT = {}

# =============================================================================== v1 bed check
pil = load("res_pilot", "pilot.json")
pg = load("res_pilot", "pilot_gates.json")["checks"]
scr1 = {i["city"]: i for i in load("res", "screen.json")["items"]}
per = pil["P1"]["per_item"]
known = [c for c in per if scr1[c]["p1"] == scr1[c]["country"] and scr1[c].get("free1") == scr1[c]["country"]]
hits_frozen = sum(1 for c in known if per[c]["reliable"] and per[c]["cb_country"] == scr1[c]["country"])
hits_noreli = sum(1 for c in known if per[c]["cb_country"] == scr1[c]["country"])
rel = sum(1 for c in per if per[c]["reliable"])
rel_cc = sum(1 for c in per if per[c]["cb_country"] == per[c]["cb_capital"])
OUT["v1"] = {"n_usable_pilot": len(per), "n_known_like": len(known), "cal_frozen": hits_frozen / len(known),
             "cal_frozen_hits": hits_frozen, "cal_without_reliability": hits_noreli / len(known), "cal_noreli_hits": hits_noreli,
             "reliability": rel / len(per), "reliability_hits": rel, "reliable_equals_country_capital_agreement": rel == rel_cc,
             "gates_file": {"cal": pg["PG2_readout"]["calibration_known"], "rel": pg["PG2_readout"]["reliability"]}}

# =============================================================================== v2 premise test
P = load("res_v2", "premise.json")
scr2 = {i["city"]: i for i in load("res", "screen.json")["items"] if i.get("ok")}


def ci11(stat, n, seed=11, nb=10000):  # analyze_v2.ci, re-implemented
    rng = np.random.default_rng(seed)
    v = np.array([stat(rng.integers(0, n, n)) for _ in range(nb)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


kn = P["known"]
hit = {v: np.array([P["A"][v][c]["pick"] == scr2[c]["country"] for c in kn], float) for v in ("prior", "R", "city+R")}
A = {v: {"cal": float(h.mean()), "ci": ci11(lambda i, h=h: h[i].mean(), len(h))} for v, h in hit.items()}
d2 = hit["R"] - hit["prior"]
A["R_minus_prior"] = {"diff": float(d2.mean()), "ci": ci11(lambda i: d2[i].mean(), len(d2))}
by = {}
for r in P["B"]:
    by.setdefault((r["kind"], r["B"]), []).append(r)
curves = {}
for kind in ("report", "readout"):
    curves[kind] = {}
    for B in P["bounds"]:
        rr = by[(kind, B)]
        don = np.array([r["pick"] == scr2[r["y"]]["country"] for r in rr], float)
        rec = np.array([r["pick"] == scr2[r["x"]]["country"] for r in rr], float)
        curves[kind][B] = {"donor": float(don.mean()), "recipient": float(rec.mean()), "n": len(rr)}
# agreement, pair-cluster CI, kappa (roles: donor / recipient / filler-by-name)
agree, pair_of, pe = [], [], []
for B in P["bounds"]:
    rep = {(r["x"], r["y"]): r["pick"] for r in by[("report", B)]}
    ro = {(r["x"], r["y"]): r["pick"] for r in by[("readout", B)]}
    keys = sorted(rep)
    for k in keys:
        agree.append(rep[k] == ro[k])
        pair_of.append(k)

    def role(r, pick):
        x, y = r
        return "donor" if pick == scr2[y]["country"] else "recipient" if pick == scr2[x]["country"] else "other:" + pick
    # chance agreement from per-B marginals over the candidate shown at each slot index
    cands = {(r["x"], r["y"]): r["cands"] for r in by[("report", B)]}
    ip = np.array([cands[k].index(rep[k]) for k in keys])
    iq = np.array([cands[k].index(ro[k]) for k in keys])
    pe.append(sum((ip == j).mean() * (iq == j).mean() for j in range(4)))
agree = np.array(agree, float)
pairs = sorted(set(pair_of))
mem = {p: [i for i, q in enumerate(pair_of) if q == p] for p in pairs}
rng = np.random.default_rng(2026)
boots = []
for _ in range(10000):
    pk = rng.integers(0, len(pairs), len(pairs))
    idx = np.concatenate([mem[pairs[j]] for j in pk])
    boots.append(agree[idx].mean())
po, pe_m = float(agree.mean()), float(np.mean(pe))
rc = by[("report_Rswap", -1)]
OUT["v2"] = {"n_known": len(kn), "n_pairs": len(P["pairs"]), "A": A, "curves": curves,
             "agreement": po, "n_rows": len(agree), "pair_cluster_ci_my_seed": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
             "expected_chance_slotwise": pe_m, "kappa_slotwise": (po - pe_m) / (1 - pe_m),
             "C": {"donor": float(np.mean([r["pick"] == scr2[r["y"]]["country"] for r in rc])),
                   "recipient": float(np.mean([r["pick"] == scr2[r["x"]]["country"] for r in rc])), "n": len(rc)}}

# =============================================================================== v3 training study
G = load("res_v3", "gen.json")
S3 = load("res_v3", "screen.json")["items"]
T, IT, th = G["tests"], G["items"], G["th"]


def b4(lp2, th=th):
    return 0 if lp2 >= th[2] else 1 if lp2 >= th[1] else 2 if lp2 >= th[0] else 3


def quart(v):
    v = sorted(v)
    return [v[len(v) // 4], v[len(v) // 2], v[(3 * len(v)) // 4]]


n_items = sum(1 for a in range(11, 100) for b in range(a, 100) if 1000 <= a * b <= 9999)
tr = [i for i in S3.values() if i["split"] == "train" and i["correct"]]
te = [i for i in S3.values() if i["split"] == "test" and i["correct"]]
th_re = quart([i["lp2"] for i in tr])
setup = {"n_items_formula": n_items, "n_items_screen": len(S3), "n_train_correct": len(tr), "n_test_correct": len(te),
         "n_train_all": sum(1 for i in S3.values() if i["split"] == "train"), "n_test_all": sum(1 for i in S3.values() if i["split"] == "test"),
         "th_recomputed": th_re, "th_matches_gen": th_re == th, "p2_cuts": [math.exp(t) for t in th_re],
         "screen_acc": float(np.mean([i["correct"] for i in S3.values()]))}
E = {"N": load("res_v3", "eval_N.json")}
for a in "OIJX":
    for s in (0, 1):
        E[f"{a}{s}"] = load("res_v3", f"eval_{a}_s{s}.json")
runs = {"N": ["N"], "O": ["O0", "O1"], "I": ["I0", "I1"], "J": ["J0", "J1"]}
t3 = T["T3"]
truth = np.array([e["label"] for e in t3])
# input-only predictors
tr_keys = sorted({e["x"] for e in G["train"]["O"]})
tr_bins = [IT[k]["bin"] for k in tr_keys]
maj = int(np.bincount(tr_bins, minlength=4).argmax())
para = np.array([IT[e["x"]]["para_bin"] for e in t3])
pl = np.array([S3[k]["para_lp2"] for k in tr_keys])
mu, sd = pl.mean(), pl.std() + 1e-9


def feats3(k):
    pb = IT[k]["para_bin"]
    z = (S3[k]["para_lp2"] - mu) / sd
    out = []
    for b in range(4):
        oh = [float(b == j) for j in (1, 2, 3)]
        out.append([float(b == pb), abs(b - pb)] + oh + [o * z for o in oh])
    return np.array(out)


def clogit(X, y, l2=1e-2):
    n = len(y)

    def f(w):
        z = X @ w
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        p = np.exp(z - lse[:, None])
        return (lse - z[np.arange(n), y]).mean() + 0.5 * l2 * w @ w, ((X * p[..., None]).sum(1) - X[np.arange(n), y]).mean(0) + l2 * w
    r = minimize(f, np.zeros(X.shape[2]), jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-9})
    return r.x


w = clogit(np.array([feats3(k) for k in tr_keys]), np.array(tr_bins))
learned = np.array([int(np.argmax(feats3(e["x"]) @ w)) for e in t3])
xs = np.mean([np.array([r["scores"] for r in E[f"X{s}"]["T3"]]) for s in (0, 1)], 0)
PRED = {"para": para, "majority": np.full(len(t3), maj), "learned": learned, "external_X": xs.argmax(1)}
corr = {k: (v == truth).astype(float) for k, v in PRED.items()}
base_stack = np.stack([corr[k] for k in corr])
best = max(corr, key=lambda k: corr[k].mean())


def boot9(stat, n, groups=None, seed=9, nb=10000):  # code/analyze.boot_ci, re-implemented
    rng = np.random.default_rng(seed)
    vals = []
    if groups is None:
        for _ in range(nb):
            vals.append(stat(rng.integers(0, n, n)))
    else:
        keys = sorted(set(groups))
        members = {k: [i for i, g in enumerate(groups) if g == k] for k in keys}
        for _ in range(nb):
            pick = rng.integers(0, len(keys), len(keys))
            vals.append(stat(np.array([i for p in pick for i in members[keys[p]]])))
    vals = np.array(vals, float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


V3 = {"setup": setup, "baselines": {k: float(v.mean()) for k, v in corr.items()}, "best": best,
      "learned_equals_para_all_items": bool((learned == para).all()), "majority_bin": maj,
      "n_div_best": int(len(t3) - corr[best].sum()), "T3": {}, "T1": {}, "T2": {}}
for arm, names in runs.items():
    cs = [np.array([E[n]["T3"][k]["pick"] == e["label"] for k, e in enumerate(t3)], float) for n in names]
    V3["T3"][arm] = {"acc_per_seed": [float(c.mean()) for c in cs],
                     "delta": float(np.mean([c.mean() for c in cs]) - corr[best].mean()),
                     "ci": boot9(lambda idx, cs=cs: np.mean([c[idx].mean() for c in cs]) - base_stack[:, idx].mean(1).max(), len(t3))}
    d = [np.mean([E[n]["T1"][k]["pick"] == e["label"] for n in names]) for k, e in enumerate(T["T1"]) if e["kind"] == "diff"]
    s = [np.mean([E[n]["T1"][k]["pick"] == e["label"] for n in names]) for k, e in enumerate(T["T1"]) if e["kind"] == "same"]
    V3["T1"][arm] = {"acc_diff": float(np.mean(d)), "acc_same": float(np.mean(s)), "n_diff": len(d), "n_same": len(s),
                     "ci_diff": boot9(lambda idx, d=np.array(d): d[idx].mean(), len(d))}
    fam = {}
    for name in ("T2a", "T2c"):
        rows = [(k, e) for k, e in enumerate(T[name]) if e["cb_int"] != IT[e["x"]]["bin"] and e["cb_int"] != IT[e["x"]]["para_bin"]]
        fam[name] = [(e["x"], np.mean([E[n][name][k]["pick"] == e["cb_int"] for n in names])) for k, e in rows]
    pooled = fam["T2a"] + fam["T2c"]
    vals = np.array([v for _, v in pooled], float)
    V3["T2"][arm] = {"T2a": float(np.mean([v for _, v in fam["T2a"]])), "T2c": float(np.mean([v for _, v in fam["T2c"]])),
                     "n": {k: len(v) for k, v in fam.items()}, "pooled": float(vals.mean()),
                     "pooled_ci": boot9(lambda idx: vals[idx].mean(), len(vals), groups=[x for x, _ in pooled])}
V3["gates"] = {"G2_kept_all_T1": float(np.mean([e["answer_kept"] for e in T["T1"]])),
               "G4_diff_bin_eq_donor": float(np.mean([e["cb_int"] == e["label"] for e in T["T1"] if e["kind"] == "diff"])),
               "G4_n": sum(1 for e in T["T1"] if e["kind"] == "diff"),
               "kept_on_diff_rows": float(np.mean([e["answer_kept"] for e in T["T1"] if e["kind"] == "diff"]))}
V3["T2_layers"] = {"T2a": sorted({tuple(e["layers"]) for e in T["T2a"]}), "la": G["cfg"]["la"], "s_add": G["cfg"]["s_add"]}
V3["J_train_uses_bin_vectors_at_la_scale_s_add"] = all("c" in e and "layer" not in e and "scale" not in e for e in G["train"]["J"])
V3["T2c_rows_are_bin_vector_at_la"] = all(e["arm"] == "add" and "layer" not in e and e["scale"] == G["cfg"]["s_add"] for e in T["T2c"])
V3["cfg"] = {k: G["cfg"][k] for k in ("steps", "bs", "seeds", "lr", "l1", "ls", "la", "s_add")}
# spearman, recalibrated paraphrase, bin counts
te3 = [S3[e["x"]] for e in t3]
V3["spearman_lp2_para"] = float(spearmanr([i["lp2"] for i in te3], [i["para_lp2"] for i in te3])[0])
thp = quart([i["para_lp2"] for i in tr])
recal = np.array([b4(S3[e["x"]]["para_lp2"], thp) for e in t3])
V3["recal_para_acc"] = float((recal == truth).mean())
V3["test_bin_counts"] = np.bincount(truth, minlength=4).tolist()
# exploratory permutation tests (pooled over seeds; own RNG)
prng = np.random.default_rng(777)


def subsets():
    d = [k for k, e in enumerate(T["T1"]) if e["kind"] == "diff"]
    s = [k for k, e in enumerate(T["T1"]) if e["kind"] == "same"]
    out = {"T1 diff": ("T1", d, np.array([T["T1"][k]["label"] for k in d])),
           "T1 same": ("T1", s, np.array([T["T1"][k]["label"] for k in s])),
           "T3": ("T3", list(range(len(t3))), truth)}
    for name in ("T2a", "T2c"):
        rows = [k for k, e in enumerate(T[name]) if e["cb_int"] != IT[e["x"]]["bin"] and e["cb_int"] != IT[e["x"]]["para_bin"]]
        out[f"{name} changed"] = (name, rows, np.array([T[name][k]["cb_int"] for k in rows]))
    return out


perm = {}
for sub, (tn, idx, lab) in subsets().items():
    for arm, names in runs.items():
        pk = np.array([[E[n][tn][k]["pick"] for k in idx] for n in names])
        acc = float((pk == lab[None]).mean())
        ind = float(np.mean([sum((p == b).mean() * (lab == b).mean() for b in range(4)) for p in pk]))
        pv = None
        if arm in ("I",) or sub in ("T1 diff", "T2c changed"):
            ps = np.array([(pk == prng.permutation(lab)[None]).mean() for _ in range(5000)])
            pv = float((np.sum(ps >= acc) + 1) / 5001)
        perm[f"{sub}:{arm}"] = {"acc": acc, "independence": ind, "perm_p_my_rng": pv, "n": len(idx)}
        if arm == "I":
            for n in names:
                p1 = np.array([E[n][tn][k]["pick"] for k in idx])
                a1 = float((p1 == lab).mean())
                ps = np.array([(p1 == prng.permutation(lab)).mean() for _ in range(5000)])
                perm[f"{sub}:{n}"] = {"acc": a1, "independence": float(sum((p1 == b).mean() * (lab == b).mean() for b in range(4))),
                                      "perm_p_my_rng": float((np.sum(ps >= a1) + 1) / 5001)}
V3["permutation"] = perm
# training logs
tl = {}
for a in "OIJX":
    for s in (0, 1):
        h = load("res_v3", f"trainlog_{a}_s{s}.json")["hist"]
        tl[f"{a}{s}"] = {"first": h[0]["loss"], "step25": [x["loss"] for x in h if x["step"] == 25][0],
                         "final": h[-1]["loss"], "final_step": h[-1]["step"],
                         "max_after_25": max(x["loss"] for x in h if x["step"] >= 25), "min_after_25": min(x["loss"] for x in h if x["step"] >= 25)}
V3["trainlogs"] = tl
V3["ln4"] = math.log(4)
spread = {}
for n, ev in E.items():
    lp = np.array([r["lp"] for r in ev["T3"]])
    spread[n] = float(np.median(np.exp(lp).max(2)))
V3["median_top_letter_prob"] = spread
OUT["v3"] = V3

# =============================================================================== Stage 0a / 0b probes (probe.json raw floats)
PRB = {"0a": load("stage0", "probe.json"), "imm": load("stage0b", "probe_imm", "stage0", "probe.json"),
       "del": load("stage0b", "probe_del", "stage0", "probe.json")}
L6 = (8, 12, 16, 20, 24, 27)


def col(src, pos, K, field, sub="acc", layers=L6):
    return {l: PRB[src]["res"][f"{pos}_L{l}_K{K}"][field][sub] for l in layers}


S0 = {}
for src in PRB:
    for pos in ("rep", "R0"):
        for K in (2, 4):
            S0[f"{src}_{pos}_K{K}"] = {"B2_donor": col(src, pos, K, "B2"), "B2_recipient": col(src, pos, K, "B2", "agree_recipient"),
                                       "A": col(src, pos, K, "A"), "B1": col(src, pos, K, "B1"), "C": col(src, pos, K, "C_input_probe"),
                                       "A_majority": col(src, pos, K, "A", "majority"), "lam": {l: PRB[src]["res"][f"{pos}_L{l}_K{K}"]["C"] for l in L6}}
rows0 = load("stage0", "rows.json")["rows"]
sw = [r for r in rows0 if r["kind"] == "swap"]
S0["rows_0a"] = {"n_nat": sum(r["kind"] == "nat" for r in rows0), "n_swap": len(sw), "n_diff2": sum(r["bin2_x"] != r["bin2_y"] for r in sw),
                 "swap_lp2_eq_donor_maxabs": max(abs(r["lp2_meas"] - r["lp2_y"]) for r in sw),
                 "digits_kept": float(np.mean([r["correct_meas"] for r in sw])), "med": load("stage0", "rows.json")["med"],
                 "p2_med": math.exp(load("stage0", "rows.json")["med"])}
for f in ("imm", "del"):
    rr = load("stage0b", f"probe_{f}", "stage0", "rows.json")["rows"]
    s = [r for r in rr if r["kind"] == "swap"]
    S0[f"rows_{f}"] = {"n_nat": sum(r["kind"] == "nat" for r in rr), "n_swap": len(s), "n_diff2": sum(r["bin2_x"] != r["bin2_y"] for r in s),
                       "swap_lp2_eq_donor_maxabs": max(abs(r["lp2_meas"] - r["lp2_y"]) for r in s), "digits_kept": float(np.mean([r["correct_meas"] for r in s]))}


def span(d, layers=L6):
    v = [d[l] for l in layers]
    return [min(v), max(v)]


summ = {"rep_B2_donor_0a": span(S0["0a_rep_K2"]["B2_donor"]), "rep_B2_donor_imm": span(S0["imm_rep_K2"]["B2_donor"]),
        "rep_B2_donor_del": span(S0["del_rep_K2"]["B2_donor"]), "R0_0b_L16plus": span(S0["imm_R0_K2"]["B2_donor"], (16, 20, 24, 27)),
        "R0_0a_L16plus": span(S0["0a_R0_K2"]["B2_donor"], (16, 20, 24, 27)), "R0_0b_L8_12": span(S0["imm_R0_K2"]["B2_donor"], (8, 12)),
        "R0_0a_L8_12": span(S0["0a_R0_K2"]["B2_donor"], (8, 12)), "R0_0b_L27_A_K2": S0["imm_R0_K2"]["A"][27], "R0_0b_L27_A_K4": S0["imm_R0_K4"]["A"][27],
        "R0_0a_L27_A_K2": S0["0a_R0_K2"]["A"][27], "R0_0a_L27_A_K4": S0["0a_R0_K4"]["A"][27],
        "rep_0a_B1": span(S0["0a_rep_K2"]["B1"]), "rep_0a_C": span(S0["0a_rep_K2"]["C"]),
        "rep_K4_A_imm": span(S0["imm_rep_K4"]["A"]), "rep_K4_A_del": span(S0["del_rep_K4"]["A"]), "rep_K4_A_majority": S0["imm_rep_K4"]["A_majority"][8],
        "rep_K4_B2donor_imm": span(S0["imm_rep_K4"]["B2_donor"]), "rep_K4_B2donor_del": span(S0["del_rep_K4"]["B2_donor"]),
        "rep_K4_B2rec_imm": span(S0["imm_rep_K4"]["B2_recipient"]), "rep_K4_B2rec_del": span(S0["del_rep_K4"]["B2_recipient"]),
        "K4_B2_donor_minus_recipient_at_rep": {f"{src}_L{l}": S0[f"{src}_rep_K4"]["B2_donor"][l] - S0[f"{src}_rep_K4"]["B2_recipient"][l]
                                               for src in ("imm", "del", "0a") for l in L6}}
summ["rep_all_three_formats"] = [min(summ["rep_B2_donor_0a"][0], summ["rep_B2_donor_imm"][0], summ["rep_B2_donor_del"][0]),
                                 max(summ["rep_B2_donor_0a"][1], summ["rep_B2_donor_imm"][1], summ["rep_B2_donor_del"][1])]
S0["summary"] = summ
OUT["stage0"] = S0

# =============================================================================== Stage 0b training (s0b_train.json raw floats)
t0b = load("stage0b", "s0b_train.json")
R0b = {"n_A1": t0b["n_A1"], "n_A2": t0b["n_A2"], "Pstar_A2_file": t0b["Pstar_A2_acc"], "N": t0b["N"], "runs": {}}
allbal, Ibal, immbal, natO, firstshare = [], [], [], [], []
for name, run in t0b["runs"].items():
    R0b["runs"][name] = {}
    for S, ev in run["evals"].items():
        h = ev["loss_hist"]
        R0b["runs"][name][S] = {k: ev[k] for k in ("acc_diff", "acc_same", "acc_swap_balanced", "acc_nat", "frac_pick0_nat", "n_diff", "n_same", "n_nat")}
        R0b["runs"][name][S].update({"final_loss": h[-1]["loss"], "final_step": h[-1]["step"], "loss_min": min(x["loss"] for x in h),
                                     "loss_max_after_first": max(x["loss"] for x in h[1:]) if len(h) > 1 else None, "loss_first": h[0]["loss"],
                                     "n_nat_pick0": round(ev["frac_pick0_nat"] * ev["n_nat"])})
        allbal.append(ev["acc_swap_balanced"])
        firstshare.append((name, S, ev["frac_pick0_nat"]))
        if run["arm"] == "I":
            Ibal.append(ev["acc_swap_balanced"])
        if run["fmt"] == "imm":
            immbal.append(ev["acc_swap_balanced"])
        if run["arm"] == "O":
            natO.append(ev["acc_nat"])
R0b["ranges"] = {"balanced_all_runs": [min(allbal), max(allbal)], "balanced_I_only": [min(Ibal), max(Ibal)],
                 "balanced_immediate_only": [min(immbal), max(immbal)], "O_nat_max": max(natO),
                 "n_checkpoints": len(allbal), "mixed_first_option_checkpoints": [x for x in firstshare if 0.1 < x[2] < 0.85]}
R0b["formats_ranks"] = {n: (r["fmt"], r["arm"], r["rank"], r["lr"], sorted(int(s) for s in r["evals"])) for n, r in t0b["runs"].items()}
# recompute P* on A2 (para / recal_para / majority) from the item screen
pilot = sorted([i for i in S3.values() if i["split"] == "pilot" and i["correct"]], key=lambda i: i["key"])
half = {p: ("A" if seeded("v4-half", p).random() < 0.5 else "B") for p in sorted({i["product"] for i in pilot})}
Ah = [i for i in pilot if half[i["product"]] == "A"]
sub = {p: ("A1" if seeded("s0b-split", p).random() < 0.6 else "A2") for p in sorted({i["product"] for i in Ah})}
A1 = [i for i in Ah if sub[i["product"]] == "A1"]
A2 = [i for i in Ah if sub[i["product"]] == "A2"]
trv = sorted(i["lp2"] for i in tr)
trp = sorted(i["para_lp2"] for i in tr)
med, med_para = trv[len(trv) // 2], trp[len(trp) // 2]
b2 = lambda v, m: 0 if v >= m else 1  # noqa: E731
y2 = np.array([b2(i["lp2"], med) for i in A2])
maj2 = int(np.bincount([b2(i["lp2"], med) for i in A1], minlength=2).argmax())
R0b["Pstar_A2_recomputed"] = {"n_A1": len(A1), "n_A2": len(A2), "para": float(np.mean([b2(i["para_lp2"], med) == y for i, y in zip(A2, y2)])),
                              "recal_para": float(np.mean([b2(i["para_lp2"], med_para) == y for i, y in zip(A2, y2)])),
                              "majority": float(np.mean(y2 == maj2)), "majority_bin": maj2, "med": med, "med_para": med_para}
OUT["stage0b_train"] = R0b

# =============================================================================== two-bin design (test items)
yt = np.array([b2(i["lp2"], med) for i in te])
OUT["two_bin_design"] = {"recal_para_test": float(np.mean([b2(i["para_lp2"], med_para) == y for i, y in zip(te, yt)])),
                         "para_test": float(np.mean([b2(i["para_lp2"], med) == y for i, y in zip(te, yt)])),
                         "file": load("res_v4_design", "design.json")["T3_input_only"]["acc"]}

# =============================================================================== tail positions (tailpos.json raw floats)
tp = load("stage0b", "tailpos.json")
dn = lambda l, p: tp["B2_donor"][f"L{l}_p{p}"]  # noqa: E731
late = [l for l in tp["layers"] if l >= 16]
npos = len(tp["positions"])
TP = {"n_B": tp["n_B"], "n_tail": npos - 1, "layers": tp["layers"], "d4_L12": dn(12, 0),
      "d4_L16plus": [min(dn(l, 0) for l in late), max(dn(l, 0) for l in late)],
      "first_L16plus": [min(dn(l, 1) for l in late), max(dn(l, 1) for l in late)],
      "first3_L16plus": [min(dn(l, p) for l in late for p in (1, 2, 3)), max(dn(l, p) for l in late for p in (1, 2, 3))],
      "first3_all_layers_max": max(dn(l, p) for l in tp["layers"] for p in (1, 2, 3)),
      "all_tail_L16plus": [min(dn(l, p) for l in late for p in range(1, npos)), max(dn(l, p) for l in late for p in range(1, npos))],
      "all_tail_L12": [min(dn(12, p) for p in range(1, npos)), max(dn(12, p) for p in range(1, npos))],
      "donor_plus_recipient_eq_1": all(abs(tp["B2_donor"][k] + tp["B2_recipient"][k] - 1) < 1e-9 for k in tp["B2_donor"]),
      "last_tail_vs_probe_imm_rep": {l: (dn(l, npos - 1), PRB["imm"]["res"][f"rep_L{l}_K2"]["B2"]["acc"], PRB["imm"]["res"][f"rep_L{l}_K2"]["C"])
                                     for l in tp["layers"]}}
OUT["tailpos"] = TP

with open(os.path.join(HERE, "recomputed_paper.json"), "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("wrote", os.path.join(HERE, "recomputed_paper.json"))

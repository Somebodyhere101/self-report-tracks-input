"""Paper 9 v3 auditor checks (numpy + standard library; no study module imported).
1. Spot check: 5 rows of eval_I_s0.json T3, scores re-derived from the per-rotation letter log-probs.
2. Leakage and design checks on gen.json (splits, donors, training lists, constraints, option sets).
3. Row-order check: eval rows align with gen.json rows (same-x report log-probs are closer than shuffled ones).
4. Extra diagnostics: independence levels and permutation tests per arm/test, picks of the input-implied bin,
   score spreads, training-label entropies, a recalibrated paraphrase predictor (exploratory, not frozen)."""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "res_v3")
SEEDS = [0, 1]


def load(n):
    with open(os.path.join(RES, n), encoding="utf-8") as f:
        return json.load(f)


def cuts(values):
    v = sorted(values)
    n = len(v)
    return [v[n // 4], v[n // 2], v[(3 * n) // 4]]


def pbin(lp2, th):
    return 0 if lp2 >= th[2] else 1 if lp2 >= th[1] else 2 if lp2 >= th[0] else 3


def scores_from_lp(lp):
    return [sum(lp[r][(k - r) % 4] for r in range(4)) / 4 for k in range(4)]


def first_argmax(s):
    b = 0
    for k in range(1, 4):
        if s[k] > s[b]:
            b = k
    return b


scr = load("screen.json")["items"]
G = load("gen.json")
T = G["tests"]
th = G["th"]
B = {k: pbin(v["lp2"], th) for k, v in scr.items()}
PB = {k: pbin(v["para_lp2"], th) for k, v in scr.items()}
split = {k: v["split"] for k, v in scr.items()}
prod = {k: v["product"] for k, v in scr.items()}
E = {"N": load("eval_N.json")}
for a in "OIJX":
    for s in SEEDS:
        E[f"{a}{s}"] = load(f"eval_{a}_s{s}.json")
OUT = {}

# ------------------------------------------------------------------ 1. spot check (5 rows of I seed 0, T3)
print("=== 1. spot check: eval_I_s0.json T3 ===")
rows = [0, 287, 574, 861, 1147]
spot = []
for i in rows:
    r = E["I0"]["T3"][i]
    s = scores_from_lp(r["lp"])
    p = first_argmax(s)
    m = max(s)
    tied = [k for k in range(4) if s[k] == m]
    spot.append({"row": i, "x": T["T3"][i]["x"], "lp": r["lp"], "scores_recomputed": s, "scores_stored": r["scores"],
                 "scores_equal": s == r["scores"], "pick_recomputed": p, "pick_stored": r["pick"], "top_tied": tied,
                 "truth_bin": T["T3"][i]["label"]})
    print(f"row {i} x={T['T3'][i]['x']} truth={T['T3'][i]['label']}")
    for rr in range(4):
        print(f"   rotation {rr}: letters A-D show bins {[(j + rr) % 4 for j in range(4)]}  lp {[round(v, 4) for v in r['lp'][rr]]}")
    print(f"   scores recomputed {[round(v, 6) for v in s]} stored {[round(v, 6) for v in r['scores']]} "
          f"equal={s == r['scores']} pick {p} stored {r['pick']} tied top {tied}")
OUT["spot_check"] = spot

# ------------------------------------------------------------------ 2. leakage and design
print("\n=== 2. leakage and design ===")
tr_prod = {prod[k] for k in scr if split[k] == "train"}
te_prod = {prod[k] for k in scr if split[k] == "test"}
pi_prod = {prod[k] for k in scr if split[k] == "pilot"}
L = {"split_products_disjoint": not (tr_prod & te_prod) and not (tr_prod & pi_prod) and not (te_prod & pi_prod),
     "every_product_single_split": all(len({split[k] for k in scr if prod[k] == p}) == 1 for p in {prod[k] for k in scr})}
train_keys = set()
for arm, lst in G["train"].items():
    for e in lst:
        train_keys.add(e["x"])
        if "y" in e:
            train_keys.add(e["y"])
L["train_list_keys"] = len(train_keys)
L["train_list_splits"] = sorted({split[k] for k in train_keys})
L["train_list_all_correct"] = all(scr[k]["correct"] for k in train_keys)
L["train_list_test_products"] = len({prod[k] for k in train_keys} & te_prod)
test_keys = set()
for name, lst in T.items():
    for e in lst:
        test_keys.add(e["x"])
        if "y" in e:
            test_keys.add(e["y"])
L["test_row_keys"] = len(test_keys)
L["test_row_splits"] = sorted({split[k] for k in test_keys})
L["test_rows_all_correct"] = all(scr[k]["correct"] for k in test_keys)
L["test_products_in_train_lists"] = len({prod[k] for k in test_keys} & {prod[k] for k in train_keys})
L["key_overlap_train_test"] = len(train_keys & test_keys)
# donors: same product, different key, constraints
chk = {}
I_ = G["train"]["I"]
chk["I_n"] = len(I_)
chk["I_alternates_diff_same"] = all(e["kind"] == ("diff", "same")[t % 2] for t, e in enumerate(I_))
chk["I_same_product"] = all(prod[e["x"]] == prod[e["y"]] and e["x"] != e["y"] for e in I_)
chk["I_label_is_donor_bin"] = all(e["label"] == B[e["y"]] for e in I_)
chk["I_diff_constraint"] = all(B[e["y"]] != B[e["x"]] and B[e["y"]] != PB[e["x"]] for e in I_ if e["kind"] == "diff")
chk["I_same_constraint"] = all(B[e["y"]] == B[e["x"]] for e in I_ if e["kind"] == "same")
J = G["train"]["J"]
chk["J_alternates"] = all(e["kind"] == ("diff", "same")[t % 2] for t, e in enumerate(J))
chk["J_diff_constraint"] = all(e["c"] not in (B[e["x"]], PB[e["x"]]) and e["label"] == e["c"] for e in J if e["kind"] == "diff")
chk["J_same_constraint"] = all(e["c"] == B[e["x"]] and e["label"] == e["c"] for e in J if e["kind"] == "same")
chk["O_label_is_natural_bin"] = all(e["label"] == B[e["x"]] for e in G["train"]["O"])
chk["X_list_equals_O"] = [dict(e, arm="O") for e in G["train"]["X"]] == G["train"]["O"]
chk["list_lengths"] = {a: len(v) for a, v in G["train"].items()}
chk["labels_in_0_3"] = all(0 <= e["label"] <= 3 for v in G["train"].values() for e in v)
t1 = T["T1"]
chk["T1_same_product_diff_key"] = all(prod[e["x"]] == prod[e["y"]] and e["x"] != e["y"] for e in t1)
chk["T1_diff_constraint"] = all(B[e["y"]] not in (B[e["x"]], PB[e["x"]]) for e in t1 if e["kind"] == "diff")
chk["T1_same_constraint"] = all(B[e["y"]] == B[e["x"]] for e in t1 if e["kind"] == "same")
chk["T1_layers"] = sorted({tuple(e["layers"]) for e in t1}) == [tuple(range(16, 28))]
chk["T1_at_most_one_row_per_kind_per_recipient"] = len({(e["x"], e["kind"]) for e in t1}) == len(t1)
chk["T2a_same_product_diff_bin"] = all(prod[e["x"]] == prod[e["y"]] and B[e["y"]] != B[e["x"]] for e in T["T2a"])
chk["T2a_layers"] = sorted({tuple(e["layers"]) for e in T["T2a"]}) == [(22,)]
chk["T2c_c_constraint"] = all(e["c"] not in (B[e["x"]], PB[e["x"]]) for e in T["T2c"])
chk["T2c_scale"] = sorted({e["scale"] for e in T["T2c"]})
chk["T4_kinds_scales"] = sorted({(e["kind"], e["scale"]) for e in T["T4"]})
chk["T4_nat_is_bin"] = all(e["nat"] == B[e["x"]] for e in T["T4"])
te_order = [e["x"] for e in T["T3"]]
chk["T2c_T4_on_first_400_T3"] = [e["x"] for e in T["T2c"]] == te_order[:400] and \
    [e["x"] for e in T["T4"] if e["kind"] == "kl"] == te_order[:400] == [e["x"] for e in T["T4"] if e["kind"] == "norm"]
chk["T5_n_and_label_first"] = len(T["T5"]) == 48 and all(e["label"] == e["opts"][0] for e in T["T5"])
# report option sets: every report row has 4 rotations x 4 letters (the fixed BINS list, rotated)
shape_ok = True
for nm, ev in E.items():
    for tname, lst in ev.items():
        if tname in ("T1", "T2a", "T2c", "T3", "T4"):
            for r in lst:
                if len(r["lp"]) != 4 or any(len(x) != 4 for x in r["lp"]):
                    shape_ok = False
                if abs(sum(math.exp(v) for v in r["lp"][0]) - 1) > 1e-4:
                    shape_ok = False
chk["all_report_rows_4x4_and_normalised"] = shape_ok
chk["eval_row_counts_match_gen"] = all(len(ev[t]) == len(T[t]) for ev in E.values() for t in ev if t in T)
L["design"] = chk
for k, v in L.items():
    if k != "design":
        print(k, v)
for k, v in chk.items():
    print("  ", k, v)
OUT["leakage"] = L

# ------------------------------------------------------------------ 3. row-order alignment check
print("\n=== 3. row alignment (report log-probs of the same recipient, N and O0) ===")
t3pos = {e["x"]: j for j, e in enumerate(T["T3"])}
rng = np.random.default_rng(5)
align = {}
for nm in ("N", "O0", "I0"):
    for tname in ("T4", "T1", "T2c"):
        lst = T[tname]
        a = np.array([E[nm][tname][k]["lp"] for k in range(len(lst))]).reshape(len(lst), 16)
        b = np.array([E[nm]["T3"][t3pos[e["x"]]]["lp"] for e in lst]).reshape(len(lst), 16)
        d_al = float(np.abs(a - b).mean())
        d_sh = float(np.abs(a - b[rng.permutation(len(b))]).mean())
        align[f"{nm}:{tname}"] = {"aligned": d_al, "shuffled": d_sh}
        print(f"  {nm} {tname}: mean |lp - lp_T3(same x)| aligned {d_al:.4f}  shuffled {d_sh:.4f}")
OUT["alignment"] = align

# ------------------------------------------------------------------ 4. diagnostics
print("\n=== 4. diagnostics ===")
picks = {nm: {t: np.array([r["pick"] for r in ev[t]]) for t in ev if t in ("T1", "T2a", "T2c", "T3", "T4")} for nm, ev in E.items()}
truthT3 = np.array([e["label"] for e in T["T3"]])
x3 = [e["x"] for e in T["T3"]]


def subsets():
    d = [k for k, e in enumerate(T["T1"]) if e["kind"] == "diff"]
    s = [k for k, e in enumerate(T["T1"]) if e["kind"] == "same"]
    out = {"T1 diff": ("T1", d, np.array([T["T1"][k]["label"] for k in d]), [T["T1"][k]["x"] for k in d]),
           "T1 same": ("T1", s, np.array([T["T1"][k]["label"] for k in s]), [T["T1"][k]["x"] for k in s]),
           "T3": ("T3", list(range(len(T["T3"]))), truthT3, x3)}
    for name in ("T2a", "T2c"):
        rows = [k for k, e in enumerate(T[name]) if e["cb_int"] != B[e["x"]] and e["cb_int"] != PB[e["x"]]]
        out[f"{name} changed"] = (name, rows, np.array([T[name][k]["cb_int"] for k in rows]), [T[name][k]["x"] for k in rows])
    return out


diag = {}
prng = np.random.default_rng(17)
for sub, (tname, idx, lab, xs) in subsets().items():
    for arm in ("N", "O", "I", "J"):
        names = ["N"] if arm == "N" else [f"{arm}{s}" for s in SEEDS]
        pk = np.array([picks[nm][tname][idx] for nm in names])
        acc = float((pk == lab[None, :]).mean())
        ind = float(np.mean([sum((p == b).mean() * (lab == b).mean() for b in range(4)) for p in pk]))
        perm = np.array([(pk == prng.permutation(lab)[None, :]).mean() for _ in range(5000)])
        pv = float((np.sum(perm >= acc) + 1) / (len(perm) + 1))
        pin = float((pk == np.array([PB[x] for x in xs])[None, :]).mean())
        nat = float((pk == np.array([B[x] for x in xs])[None, :]).mean())
        diag[f"{sub}:{arm}"] = {"acc": acc, "independence": ind, "perm_p": pv, "picks_para_bin_of_x": pin,
                                "picks_natural_bin_of_x": nat, "n": len(idx)}
        print(f"  {sub:12s} {arm}: acc {acc:.3f}  independence {ind:.3f}  perm p {pv:.4f}  "
              f"picks para-bin(x) {pin:.3f}  picks bin(x) {nat:.3f}  n {len(idx)}")
OUT["independence"] = diag

# score spreads (max - min of rotation-averaged scores) and per-rotation letter spread
spread = {}
for nm, ev in E.items():
    sc = np.array([r["scores"] for r in ev["T3"]])
    lp = np.array([r["lp"] for r in ev["T3"]])
    spread[nm] = {"score_range_median": float(np.median(sc.max(1) - sc.min(1))),
                  "letter_lp_range_median": float(np.median(lp.max(2) - lp.min(2))),
                  "max_letter_prob_median": float(np.median(np.exp(lp).max(2)))}
    print(f"  {nm}: median score range {spread[nm]['score_range_median']:.4f}; median per-rotation letter lp range "
          f"{spread[nm]['letter_lp_range_median']:.4f}; median top-letter prob {spread[nm]['max_letter_prob_median']:.3f}")
OUT["spread"] = spread
# distinct logit-difference grid (bf16 quantisation of the letter logits)
d = []
for r in E["O0"]["T3"][:200]:
    for rot in r["lp"]:
        d += [round(rot[j] - rot[0], 6) for j in range(1, 4)]
OUT["O0_letter_lp_differences_distinct_first200"] = sorted(set(d))[:30]
print("  O0 letter log-prob differences (first 200 rows), distinct values:", sorted(set(d))[:20])

# training-label entropy per arm (the label-blind loss floor) vs ln 4
ent = {}
for arm, lst in G["train"].items():
    c = np.bincount([e["label"] for e in lst], minlength=4) / len(lst)
    ent[arm] = {"label_shares": c.round(4).tolist(), "entropy": float(-(c[c > 0] * np.log(c[c > 0])).sum())}
print("  training-label shares and entropy (ln 4 = 1.3863):", ent)
OUT["train_label_entropy"] = ent

# exploratory: recalibrated paraphrase bin (para_lp2 binned at its own train quartiles) -- input-only, train-fit
tr = [i for i in scr.values() if i["split"] == "train" and i["correct"]]
th_para = cuts([i["para_lp2"] for i in tr])
rec = np.array([pbin(scr[k]["para_lp2"], th_para) for k in x3])
para = np.array([PB[k] for k in x3])
OUT["exploratory_recalibrated_para"] = {"th_para": th_para, "acc": float((rec == truthT3).mean()),
                                        "frozen_para_acc": float((para == truthT3).mean()),
                                        "n_div_recal": int((rec != truthT3).sum())}
print("  exploratory recalibrated paraphrase bin: acc", OUT["exploratory_recalibrated_para"])
# para bin shift
OUT["para_minus_truth_bin_shift_mean"] = float((para - truthT3).mean())
print("  mean (para bin - truth bin):", OUT["para_minus_truth_bin_shift_mean"])

# X seeds: picks of bin 0 and ties
for s in SEEDS:
    sc = np.array([r["scores"] for r in E[f"X{s}"]["T3"]])
    ties = int(sum((row == row.max()).sum() > 1 for row in sc))
    print(f"  X seed {s}: picks bin0 share {(picks[f'X{s}']['T3'] == 0).mean():.3f}, tied rows {ties}")

with open(os.path.join(ROOT, "audit", "v3_checks.json"), "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("\nwrote audit/v3_checks.json")

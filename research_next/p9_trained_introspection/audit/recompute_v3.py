"""Paper 9 v3: independent recomputation by the auditor (numpy + standard library only; no study module imported).

Inputs (read-only): res_v3/screen.json, res_v3/gen.json, res_v3/eval_*.json, res_v3/trainlog_*.json.
res_v3/analysis.json is read only at the end, for the comparison table.
Output: audit/recomputed_v3.json and a printed report.

Definitions follow PREDICTIONS_V3.md section 3 (and v1's PREDICTIONS.md sections 7-8 where v3 says "as v1").
Bootstrap: 10,000 percentile resamples with the auditor's own seed (the study used seed 9)."""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "res_v3")
OUT = os.path.join(ROOT, "audit", "recomputed_v3.json")
NB = 10000
SEED = 4242
SEEDS = [0, 1]


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- basic definitions (own code)
def cuts(values):
    """Quartile cut points as frozen in v3 (sorted values at n//4, n//2, 3n//4)."""
    v = sorted(values)
    n = len(v)
    return [v[n // 4], v[n // 2], v[(3 * n) // 4]]


def pbin(lp2, th):  # bin 0 = most sure
    return 0 if lp2 >= th[2] else 1 if lp2 >= th[1] else 2 if lp2 >= th[0] else 3


def ebin(e, th):  # entropy: higher = less sure
    return 0 if e <= th[0] else 1 if e <= th[1] else 2 if e <= th[2] else 3


def scores_from_lp(lp):
    """Rotation r shows option (j + r) % 4 at letter j, so option k sits at letter (k - r) % 4 in rotation r."""
    return [sum(lp[r][(k - r) % 4] for r in range(4)) / 4 for k in range(4)]


def first_argmax(s):
    best = 0
    for k in range(1, len(s)):
        if s[k] > s[best]:
            best = k
    return best


def pct(vals):
    vals = np.asarray(vals, float)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot_rows(mat_fn, n, rng, chunk=1000):
    """Item bootstrap. mat_fn(W) gets W [b, n] resample counts and returns b statistics."""
    out = []
    left = NB
    while left > 0:
        b = min(chunk, left)
        idx = rng.integers(0, n, size=(b, n))
        W = np.zeros((b, n))
        np.add.at(W, (np.repeat(np.arange(b), n), idx.ravel()), 1.0)
        out.append(mat_fn(W))
        left -= b
    return np.concatenate(out)


def boot_cluster(vals, groups, rng):
    """Cluster bootstrap of the pooled row mean: resample clusters (with replacement), keep all their rows."""
    keys = sorted(set(groups))
    pos = {k: i for i, k in enumerate(keys)}
    g = np.array([pos[x] for x in groups])
    sums = np.bincount(g, weights=vals, minlength=len(keys))
    cnts = np.bincount(g, minlength=len(keys)).astype(float)
    stats = []
    for _ in range(NB):
        w = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))
        stats.append((w @ sums) / (w @ cnts))
    return pct(stats)


def power_sim(n, n_div, a_agree=0.97, a_div=0.70, nsim=4000, seed=7):
    """Strong reader: right w.p. a_div where P* errs, w.p. a_agree where P* is right. Per-item d = acc(R) - acc(P*).
    Power = share of simulations with mean(d) - 1.96 * sd(d, ddof=1)/sqrt(n) > 0."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(nsim):
        d = np.empty(n)
        d[:n_div] = (rng.random(n_div) < a_div).astype(float)
        d[n_div:] = -(rng.random(n - n_div) >= a_agree).astype(float)
        hits += (d.mean() - 1.96 * d.std(ddof=1) / math.sqrt(n)) > 0
    return hits / nsim


def cond_logit_newton(X, y, l2=1e-2, tol=1e-12, maxit=100):
    """Conditional logit over 4 candidates, mean NLL + 0.5*l2*|w|^2, fit by damped Newton (own implementation)."""
    n, K, f = X.shape
    w = np.zeros(f)

    def obj(w):
        z = X @ w
        zm = z.max(1, keepdims=True)
        lse = (zm[:, 0] + np.log(np.exp(z - zm).sum(1)))
        return (lse - z[np.arange(n), y]).mean() + 0.5 * l2 * w @ w

    for it in range(maxit):
        z = X @ w
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        mx = (X * p[..., None]).sum(1)  # [n, f]
        g = (mx - X[np.arange(n), y]).mean(0) + l2 * w
        if np.abs(g).max() < tol:
            break
        Sxx = np.einsum("nk,nkf,nkg->fg", p, X, X) / n
        H = Sxx - (mx.T @ mx) / n + l2 * np.eye(f)
        step = np.linalg.solve(H, g)
        t, f0 = 1.0, obj(w)
        while obj(w - t * step) > f0 - 1e-4 * t * (g @ step) and t > 1e-8:
            t *= 0.5
        w = w - t * step
    z = X @ w
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    g = ((X * p[..., None]).sum(1) - X[np.arange(n), y]).mean(0) + l2 * w
    return w, float(np.abs(g).max()), it


# ----------------------------------------------------------------------------- main
def main():
    R = {}
    scr = load("screen.json")["items"]
    G = load("gen.json")
    T = G["tests"]
    E = {"N": load("eval_N.json")}
    for a in "OIJX":
        for s in SEEDS:
            E[f"{a}{s}"] = load(f"eval_{a}_s{s}.json")

    def runs(arm):
        return [E["N"]] if arm == "N" else [E[f"{arm}{s}"] for s in SEEDS]

    # ---------------- items, cut points, bins
    tr = [i for i in scr.values() if i["split"] == "train" and i["correct"]]
    te = [i for i in scr.values() if i["split"] == "test" and i["correct"]]
    te_all = [i for i in scr.values() if i["split"] == "test"]
    th = cuts([i["lp2"] for i in tr])
    th_e = cuts([i["ent2"] for i in tr])
    R["n_items_screen"] = len(scr)
    R["n_train_correct"] = len(tr)
    R["n_test_correct"] = len(te)
    R["th"] = th
    R["th_equals_gen"] = th == G["th"]
    R["th_ent_equals_gen"] = th_e == G["th_ent"]
    R["th_numpy_linear_percentiles"] = [float(np.percentile([i["lp2"] for i in tr], q)) for q in (25, 50, 75)]
    R["th_p2"] = [math.exp(t) for t in th]
    # cut points from other splits, to show th is the train one
    for sp in ("test", "pilot"):
        R[f"th_from_{sp}_items_equals_gen"] = cuts([i["lp2"] for i in scr.values() if i["split"] == sp and i["correct"]]) == G["th"]
    B = {k: pbin(v["lp2"], th) for k, v in scr.items()}
    PB = {k: pbin(v["para_lp2"], th) for k, v in scr.items()}
    EB = {k: ebin(v["ent2"], th_e) for k, v in scr.items()}
    # items table in gen.json
    IT = G["items"]
    R["items_table"] = {"n": len(IT), "splits": sorted({scr[k]["split"] for k in IT}),
                        "all_correct": all(scr[k]["correct"] for k in IT),
                        "equals_train_plus_test_correct": set(IT) == {i["key"] for i in tr + te},
                        "bin_mismatch": sum(IT[k]["bin"] != B[k] for k in IT),
                        "para_bin_mismatch": sum(IT[k]["para_bin"] != PB[k] for k in IT),
                        "ent_bin_mismatch": sum(IT[k]["ent_bin"] != EB[k] for k in IT)}
    R["bin_counts_train"] = np.bincount([B[i["key"]] for i in tr], minlength=4).tolist()
    R["bin_counts_test"] = np.bincount([B[i["key"]] for i in te], minlength=4).tolist()

    # ---------------- gates
    g1_acc = float(np.mean([i["correct"] for i in te_all]))
    R["G1"] = {"test_accuracy": g1_acc, "n_test_items": len(te_all), "n_usable_test": len(te),
               "gen_n_test_items": G["n_test_items"], "pass": g1_acc >= 0.70 and len(te) >= 150}
    kept = [e["answer_kept"] for e in T["T1"]]
    R["G2"] = {"kept": float(np.mean(kept)), "n": len(kept), "pass": float(np.mean(kept)) >= 0.95}
    t3 = T["T3"]
    t3k = [e["x"] for e in t3]
    R["T3_rows_are_all_usable_test_items_in_order"] = t3k == [i["key"] for i in te]
    R["T3_label_mismatch"] = sum(e["label"] != B[e["x"]] for e in t3)
    agree = float(np.mean([B[k] == EB[k] for k in t3k]))
    R["G3"] = {"agree": agree, "n": len(t3k), "pass": agree >= 0.70}
    # recompute the intervened bins from lp2_int
    cbm = {}
    for name in ("T1", "T2a", "T2c", "T4"):
        cbm[name] = sum(e["cb_int"] != pbin(e["lp2_int"], th) for e in T[name])
    R["cb_int_mismatch_vs_own_bin_of_lp2_int"] = cbm
    R["T1_label_mismatch_vs_donor_bin"] = sum(e["label"] != B[e["y"]] for e in T["T1"])
    d1 = [e for e in T["T1"] if e["kind"] == "diff"]
    s1 = [e for e in T["T1"] if e["kind"] == "same"]
    g4 = float(np.mean([pbin(e["lp2_int"], th) == B[e["y"]] for e in d1]))
    R["G4"] = {"diff_carry_donor_bin": g4, "n": len(d1), "pass": g4 >= 0.90,
               "same_rows_carry_donor_bin": float(np.mean([pbin(e["lp2_int"], th) == B[e["y"]] for e in s1])),
               "n_same": len(s1), "answer_kept_diff": float(np.mean([e["answer_kept"] for e in d1])),
               "answer_kept_same": float(np.mean([e["answer_kept"] for e in s1]))}

    # ---------------- T3 picks (re-derived from lp) and the predictors
    n3 = len(t3)
    truth = np.array([B[k] for k in t3k])
    R["T3_truth_counts"] = np.bincount(truth, minlength=4).tolist()
    picks = {}
    pick_mismatch = {}
    score_mismatch = {}
    for name, ev in E.items():
        for tname in ("T1", "T2a", "T2c", "T3", "T4"):
            if tname not in ev:
                continue
            mm = sm = 0
            pk = []
            for r in ev[tname]:
                s = scores_from_lp(r["lp"])
                p = first_argmax(s)
                mm += p != r["pick"]
                sm += s != r["scores"]
                pk.append(p)
            picks[(name, tname)] = np.array(pk)
            pick_mismatch[f"{name}:{tname}"] = mm
            score_mismatch[f"{name}:{tname}"] = sm
    R["pick_mismatch_rederived_vs_stored"] = int(sum(pick_mismatch.values()))
    R["score_mismatch_rederived_vs_stored"] = int(sum(score_mismatch.values()))
    R["rows_rederived"] = int(sum(len(v) for v in picks.values()))
    # opposite convention, to show the check bites
    wrong = 0
    for name, ev in E.items():
        for r in ev["T3"]:
            s = [sum(r["lp"][r_][(k + r_) % 4] for r_ in range(4)) / 4 for k in range(4)]
            wrong += first_argmax(s) != r["pick"]
    R["pick_mismatch_opposite_rotation_convention_T3"] = wrong

    # ties at the top of the rotation-averaged scores (picks then default to the lowest bin index)
    ties = {}
    for name, ev in E.items():
        nt = 0
        for r in ev["T3"]:
            s = r["scores"]
            m = max(s)
            nt += sum(v == m for v in s) > 1
        ties[name] = nt / len(ev["T3"])
    R["T3_share_rows_with_tied_top_score"] = ties

    # predictor 1: paraphrase bin
    P = {"para": np.array([PB[k] for k in t3k])}
    # predictor 2: majority train bin (two readings of "train items")
    o_keys = sorted({e["x"] for e in G["train"]["O"]})
    maj_olist = int(np.bincount([B[k] for k in o_keys], minlength=4).argmax())
    maj_all = int(np.bincount([B[i["key"]] for i in tr], minlength=4).argmax())
    R["majority"] = {"olist_unique_items": len(o_keys), "bin_counts_olist": np.bincount([B[k] for k in o_keys], minlength=4).tolist(),
                     "majority_olist": maj_olist, "majority_all_train": maj_all}
    P["majority"] = np.full(n3, maj_olist)

    # predictor 3: conditional logit (own Newton fit), fit on train items only
    def feat_rows(k, mu, sd, with_bin_intercepts=True):
        pb_ = PB[k]
        z = (scr[k]["para_lp2"] - mu) / sd
        rows = []
        for b in range(4):
            oh = [float(b == j) for j in (1, 2, 3)]
            r_ = [float(b == pb_), float(abs(b - pb_))]
            if with_bin_intercepts:
                r_ += oh
            r_ += [o * z for o in oh]
            rows.append(r_)
        return rows

    learned = {}
    for tag, keys, bi in (("olist_8feat", o_keys, True), ("alltrain_8feat", [i["key"] for i in tr], True),
                          ("olist_5feat_text_only", o_keys, False), ("alltrain_5feat_text_only", [i["key"] for i in tr], False)):
        pl = np.array([scr[k]["para_lp2"] for k in keys])
        mu, sd = pl.mean(), pl.std() + 1e-9
        Xtr = np.array([feat_rows(k, mu, sd, bi) for k in keys])
        ytr = np.array([B[k] for k in keys])
        w, gmax, its = cond_logit_newton(Xtr, ytr)
        Xte = np.array([feat_rows(k, mu, sd, bi) for k in t3k])
        pred = (Xte @ w).argmax(1)
        learned[tag] = {"w": w.tolist(), "max_abs_grad": gmax, "newton_iters": its, "n_fit": len(keys),
                        "acc_T3": float((pred == truth).mean()), "n_differs_from_para": int((pred != P["para"]).sum()),
                        "pred_counts": np.bincount(pred, minlength=4).tolist()}
        if tag == "olist_8feat":
            P["learned"] = pred
    R["learned"] = learned

    # predictor 4: X, seed-averaged scores
    xs = np.mean([np.array([r["scores"] for r in E[f"X{s}"]["T3"]]) for s in SEEDS], 0)
    P["external_X"] = xs.argmax(1)
    R["X_seedavg_ties"] = int(sum((row == row.max()).sum() > 1 for row in xs))

    corr = {k: (v == truth).astype(float) for k, v in P.items()}
    R["T3_predictors"] = {k: float(v.mean()) for k, v in corr.items()}
    best = max(corr, key=lambda k: corr[k].mean())  # first max in order para, majority, learned, X
    R["P_star"] = best
    R["learned_equals_para_on_all_T3"] = bool((P["learned"] == P["para"]).all())
    n_div = int(n3 - corr[best].sum())
    pw = power_sim(n3, n_div)
    R["G5"] = {"best": best, "n_T3": n3, "n_div": n_div, "div_rate": n_div / n3, "power": pw,
               "pass": n_div >= 25 and pw >= 0.80}
    # bin-level confusion of truth vs paraphrase bin
    conf = np.zeros((4, 4), int)
    for a_, b_ in zip(truth, P["para"]):
        conf[a_, b_] += 1
    R["T3_confusion_truth_rows_para_cols"] = conf.tolist()
    R["T3_para_within_one_bin"] = float((np.abs(truth - P["para"]) <= 1).mean())
    lp2 = np.array([scr[k]["lp2"] for k in t3k])
    plp2 = np.array([scr[k]["para_lp2"] for k in t3k])

    def rank(a):
        o = np.argsort(a, kind="mergesort")
        r_ = np.empty(len(a))
        r_[o] = np.arange(len(a))
        # average ties
        vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r_)
        return (sums / cnt)[inv]
    R["T3_spearman_lp2_vs_para_lp2"] = float(np.corrcoef(rank(lp2), rank(plp2))[0, 1])

    # ---------------- T3 arms, Delta and bootstrap CIs (P* re-chosen in every resample)
    base = np.stack([corr[k] for k in ("para", "majority", "learned", "external_X")])  # [4, n]
    rng = np.random.default_rng(SEED)
    R["T3"] = {}
    div = corr[best] == 0
    for arm in ("N", "O", "I", "J"):
        cs = [(picks[(nm, "T3")] == truth).astype(float) for nm in (["N"] if arm == "N" else [f"{arm}{s}" for s in SEEDS])]
        cmean = np.mean(cs, 0)
        delta = float(cmean.mean() - corr[best].mean())

        def stat(W, cmean=cmean):
            return (W @ cmean) / n3 - ((W @ base.T) / n3).max(1)
        ci = pct(boot_rows(stat, n3, rng))
        nb = [(picks[(nm, "T3")] == P[best]).astype(float) for nm in (["N"] if arm == "N" else [f"{arm}{s}" for s in SEEDS])]
        R["T3"][arm] = {"acc_per_seed": [float(c.mean()) for c in cs], "delta": delta, "ci": ci,
                        "div_names_truth": float(np.mean([c[div].mean() for c in cs])),
                        "div_names_Pstar": float(np.mean([b[div].mean() for b in nb])),
                        "pick_counts": [np.bincount(picks[(nm, "T3")], minlength=4).tolist()
                                        for nm in (["N"] if arm == "N" else [f"{arm}{s}" for s in SEEDS])]}
        # expected accuracy with random (not lowest-index) tie-breaking
        exp_acc = []
        for nm in (["N"] if arm == "N" else [f"{arm}{s}" for s in SEEDS]):
            tot = 0.0
            for r, t in zip(E[nm]["T3"], truth):
                m = max(r["scores"])
                top = [k for k in range(4) if r["scores"][k] == m]
                tot += (t in top) / len(top)
            exp_acc.append(tot / n3)
        R["T3"][arm]["acc_random_tiebreak"] = exp_acc
    R["T3"]["X"] = {"acc": float(corr["external_X"].mean()),
                    "acc_per_seed_own_pick": [float((np.array([r["pick"] for r in E[f"X{s}"]["T3"]]) == truth).mean()) for s in SEEDS]}
    R["T3_best_constant"] = {"bin": int(np.bincount(truth, minlength=4).argmax()),
                             "acc": float(np.bincount(truth, minlength=4).max() / n3)}

    # ---------------- T1
    R["T1"] = {}
    rows_d = [(k, e) for k, e in enumerate(T["T1"]) if e["kind"] == "diff"]
    rows_s = [(k, e) for k, e in enumerate(T["T1"]) if e["kind"] == "same"]
    lab_d = np.array([B[e["y"]] for _, e in rows_d])
    for arm in ("N", "O", "I", "J"):
        names = ["N"] if arm == "N" else [f"{arm}{s}" for s in SEEDS]
        d = np.mean([[picks[(nm, "T1")][k] == B[e["y"]] for k, e in rows_d] for nm in names], 0)
        sm = np.mean([[picks[(nm, "T1")][k] == B[e["y"]] for k, e in rows_s] for nm in names], 0)
        own = np.mean([[picks[(nm, "T1")][k] == B[e["x"]] for k, e in rows_d] for nm in names], 0)
        ci = pct(boot_rows(lambda W, d=d: (W @ d) / len(d), len(d), rng))
        # independence level: picks and labels independent, same marginals (per seed, then averaged)
        ind = np.mean([sum(np.mean(picks[(nm, "T1")][[k for k, _ in rows_d]] == b) * np.mean(lab_d == b) for b in range(4))
                       for nm in names])
        # permutation test of acc_diff against independence (labels shuffled across diff rows)
        pk = np.array([[picks[(nm, "T1")][k] for k, _ in rows_d] for nm in names])
        prng = np.random.default_rng(11)
        perm = []
        for _ in range(5000):
            lp_ = prng.permutation(lab_d)
            perm.append(float(np.mean(pk == lp_[None, :])))
        R["T1"][arm] = {"acc_diff": float(d.mean()), "ci_diff": ci, "acc_same": float(sm.mean()),
                        "names_own_bin_diff": float(own.mean()), "n_diff": len(d), "n_same": len(sm),
                        "independence_level_diff": float(ind),
                        "perm_p_one_sided": float((np.sum(np.array(perm) >= d.mean()) + 1) / (len(perm) + 1))}
    R["T1_donor_bin_counts_diff"] = np.bincount(lab_d, minlength=4).tolist()

    # ---------------- T2
    R["T2"] = {}
    for arm in ("N", "O", "I", "J"):
        names = ["N"] if arm == "N" else [f"{arm}{s}" for s in SEEDS]
        fam = {}
        for name in ("T2a", "T2c"):
            rows = [(k, e) for k, e in enumerate(T[name])
                    if pbin(e["lp2_int"], th) != B[e["x"]] and pbin(e["lp2_int"], th) != PB[e["x"]]]
            fam[name] = [(e["x"], float(np.mean([picks[(nm, name)][k] == pbin(e["lp2_int"], th) for nm in names]))) for k, e in rows]
        pooled = fam["T2a"] + fam["T2c"]
        vals = np.array([v for _, v in pooled])
        grp = [x for x, _ in pooled]
        ci = boot_cluster(vals, grp, rng)
        evaluable = [nm for nm in fam if len(fam[nm]) >= 20]
        accs = {nm: float(np.mean([v for _, v in fam[nm]])) for nm in fam}
        R["T2"][arm] = {"n_changed": {nm: len(fam[nm]) for nm in fam}, "acc": accs, "pooled": float(vals.mean()),
                        "pooled_ci": ci, "n_recipients": len(set(grp)), "evaluable": evaluable,
                        "rule_pass": ci[0] > 1 / 3 and all(accs[f] > 1 / 3 for f in evaluable)}
    R["T2_rows_total"] = {"T2a": len(T["T2a"]), "T2c": len(T["T2c"])}

    # ---------------- T4
    R["T4"] = {}
    t3pos = {k: j for j, k in enumerate(t3k)}
    for arm in ("N", "O", "I", "J"):
        names = ["N"] if arm == "N" else [f"{arm}{s}" for s in SEEDS]
        R["T4"][arm] = {}
        for kind in ("kl", "norm"):
            dd = []
            for k, e in enumerate(T["T4"]):
                if e["kind"] != kind or e["x"] not in t3pos:
                    continue
                j = t3pos[e["x"]]
                cbi = pbin(e["lp2_int"], th)
                dd.append(np.mean([picks[(nm, "T4")][k] != cbi for nm in names]) -
                          np.mean([picks[(nm, "T3")][j] != truth[j] for nm in names]))
            dd = np.array(dd)
            ci = pct(boot_rows(lambda W, dd=dd: (W @ dd) / len(dd), len(dd), rng))
            R["T4"][arm][kind] = {"n": len(dd), "switch_increase": float(dd.mean()), "ci": ci,
                                  "bin_changed_rate": float(np.mean([pbin(e["lp2_int"], th) != B[e["x"]] for e in T["T4"] if e["kind"] == kind])),
                                  "pass_upper_le_0.15": ci[1] <= 0.15}

    # ---------------- T5
    R["T5"] = {}
    for arm in ("N", "O", "I", "J"):
        names = ["N"] if arm == "N" else [f"{arm}{s}" for s in SEEDS]
        mc = float(np.mean([np.mean([E[nm]["T5"][k]["pick"] == e["label"] for k, e in enumerate(T["T5"])]) for nm in names]))
        mu_ = float(np.mean([np.mean([u["correct"] for u in E[nm]["T5u"]]) for nm in names]))
        R["T5"][arm] = {"mc_acc_gated": mc, "mult_acc_ungated": mu_,
                        "T5u_keys_are_first_400_T3": all([u["key"] for u in E[nm]["T5u"]] == t3k[:400] for nm in names)}
    for arm in ("O", "I", "J"):
        R["T5"][arm]["drift_flag"] = (R["T5"]["N"]["mc_acc_gated"] - R["T5"][arm]["mc_acc_gated"] > 0.10 or
                                      R["T5"]["N"]["mult_acc_ungated"] - R["T5"][arm]["mult_acc_ungated"] > 0.10)

    # ---------------- verdict (v1 rule as adapted in PREDICTIONS_V3 section 3)
    gates = {g: R[g]["pass"] for g in ("G1", "G2", "G3", "G4", "G5")}
    R["gates"] = gates
    reasons = []
    if not all(gates.values()):
        verdict = "UNINFORMATIVE"
        reasons = [g for g, v in gates.items() if not v]
    elif not R["T2"]["I"]["evaluable"]:
        verdict = "UNINFORMATIVE"
        reasons = ["T2 not evaluable"]
    else:
        i_ok = R["T3"]["I"]["ci"][0] > 0
        o_ok = R["T3"]["O"]["ci"][0] > 0
        t2_ok = R["T2"]["I"]["rule_pass"]
        t4_ok = R["T4"]["I"]["kl"]["pass_upper_le_0.15"]
        if not i_ok:
            reasons.append("I does not beat the best input-only predictor on T3")
        if o_ok:
            reasons.append("O also beats the best input-only predictor on T3")
        if not t2_ok:
            reasons.append("I is not above 1/3 on unseen intervention families (T2)")
        if not t4_ok:
            reasons.append("I is tamper-sensitive (T4-kl)")
        verdict = "SUPPORTED" if (i_ok and not o_ok and t2_ok and t4_ok) else "NOT SUPPORTED"
    R["verdict"], R["reasons"] = verdict, reasons

    # ---------------- pick distributions on all report rows
    dist = {}
    for name, ev in E.items():
        allp = np.concatenate([picks[(name, t)] for t in ("T1", "T2a", "T2c", "T3", "T4") if (name, t) in picks])
        dist[name] = (np.bincount(allp, minlength=4) / len(allp)).round(4).tolist()
    R["pick_share_all_report_rows"] = dist
    R["N_always_bin1"] = {t: bool((picks[("N", t)] == 1).all()) for t in ("T1", "T2a", "T2c", "T3", "T4")}

    # ---------------- training logs
    tl = {}
    allv = []
    for a in "OIJX":
        for s in SEEDS:
            h = load(f"trainlog_{a}_s{s}.json")["hist"]
            st = {x["step"]: x["loss"] for x in h}
            later = [v for k, v in st.items() if k >= 25]
            later50 = [v for k, v in st.items() if k >= 50]
            allv += later50
            tl[f"{a}{s}"] = {"step0": st[0], "step25": st[25], "mean_ge25": float(np.mean(later)),
                             "mean_ge50": float(np.mean(later50)), "min_ge25": min(later), "max_ge25": max(later),
                             "n_logged": len(h)}
    R["trainlogs"] = tl
    R["trainlog_all_runs_mean_ge50"] = float(np.mean(allv))
    R["ln4"] = math.log(4)
    R["X_train_equals_O"] = [dict(e, arm="O") for e in G["train"]["X"]] == G["train"]["O"]

    # ---------------- save and compare
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return R


def compare(R):
    A = json.load(open(os.path.join(RES, "analysis.json"), encoding="utf-8"))
    rows = []

    def add(name, mine, rep, tol=1e-9):
        if isinstance(mine, (list, tuple)):
            ok = all(abs(a - b) <= tol for a, b in zip(mine, rep))
            ms = "[" + ", ".join(f"{v:.4f}" for v in mine) + "]"
            rs = "[" + ", ".join(f"{v:.4f}" for v in rep) + "]"
        elif isinstance(mine, (bool, str)) or isinstance(rep, (bool, str)):
            ok = mine == rep
            ms, rs = str(mine), str(rep)
        else:
            ok = abs(mine - rep) <= tol
            ms, rs = (f"{mine:.4f}", f"{rep:.4f}") if isinstance(mine, float) or isinstance(rep, float) else (str(mine), str(rep))
        rows.append((name, ms, rs, "match" if ok else "MISMATCH"))

    gv = A["gate_values"]
    add("G1 test accuracy", R["G1"]["test_accuracy"], gv["G1"]["accuracy_test"])
    add("G1 usable test items", R["G1"]["n_usable_test"], gv["G1"]["n_test_usable"])
    add("G2 digits kept (T1)", R["G2"]["kept"], gv["G2"]["answer_kept_T1"])
    add("G3 prob/entropy bin agreement", R["G3"]["agree"], gv["G3"]["bin_prob_vs_entropy_agreement"])
    add("G4 diff rows carry donor bin", R["G4"]["diff_carry_donor_bin"], gv["G4"]["T1_diff_bin_equals_donor"])
    add("G4 n diff rows", R["G4"]["n"], gv["G4"]["n"])
    add("G5 P*", R["G5"]["best"], gv["G5"]["best_baseline"])
    add("G5 n divergent", R["G5"]["n_div"], gv["G5"]["n_divergent"])
    add("G5 power", R["G5"]["power"], gv["G5"]["power_strong_reader"], 0.01)
    for k in ("G1", "G2", "G3", "G4", "G5"):
        add(f"gate {k}", bool(R["gates"][k]), bool(A["gates"][k]))
    for k in ("para", "majority", "learned", "external_X"):
        add(f"T3 predictor {k}", R["T3_predictors"][k], A["T3_baselines"][k])
    add("majority bin", R["majority"]["majority_olist"], A["T3_learned"]["majority_bin"])
    add("learned fit items", R["learned"]["olist_8feat"]["n_fit"], A["T3_learned"]["n_train_items"])
    for arm in ("N", "O", "I", "J"):
        add(f"T3 {arm} acc per seed", R["T3"][arm]["acc_per_seed"], A["T3"][arm]["acc_per_seed"])
        add(f"T3 {arm} Delta", R["T3"][arm]["delta"], A["T3"][arm]["delta_vs_best_baseline"])
        add(f"T3 {arm} Delta CI", R["T3"][arm]["ci"], A["T3"][arm]["ci"], 0.01)
        add(f"T3 {arm} divergent: names truth", R["T3"][arm]["div_names_truth"], A["T3"][arm]["divergent_names_cb"])
        add(f"T3 {arm} divergent: names P*", R["T3"][arm]["div_names_Pstar"], A["T3"][arm]["divergent_names_baseline"])
    for arm in ("N", "O", "I", "J"):
        add(f"T1 {arm} acc diff", R["T1"][arm]["acc_diff"], A["T1"][arm]["acc_diff"])
        add(f"T1 {arm} acc diff CI", R["T1"][arm]["ci_diff"], A["T1"][arm]["ci_diff"], 0.01)
        add(f"T1 {arm} acc same", R["T1"][arm]["acc_same"], A["T1"][arm]["acc_same"])
        add(f"T1 {arm} names own bin", R["T1"][arm]["names_own_bin_diff"], A["T1"][arm]["names_input_bin_diff"])
    for arm in ("N", "O", "I", "J"):
        for fam in ("T2a", "T2c"):
            add(f"T2 {arm} n changed {fam}", R["T2"][arm]["n_changed"][fam], A["T2"][arm]["n_changed"][fam])
            add(f"T2 {arm} acc {fam}", R["T2"][arm]["acc"][fam], A["T2"][arm]["acc"][fam])
        add(f"T2 {arm} pooled", R["T2"][arm]["pooled"], A["T2"][arm]["pooled_acc"])
        add(f"T2 {arm} pooled CI", R["T2"][arm]["pooled_ci"], A["T2"][arm]["pooled_ci"], 0.01)
    for arm in ("N", "O", "I", "J"):
        for kind in ("kl", "norm"):
            add(f"T4 {arm} {kind} switch increase", R["T4"][arm][kind]["switch_increase"], A["T4"][arm][kind]["switch_increase"])
            add(f"T4 {arm} {kind} CI", R["T4"][arm][kind]["ci"], A["T4"][arm][kind]["ci"], 0.01)
            add(f"T4 {arm} {kind} bin changed", R["T4"][arm][kind]["bin_changed_rate"], A["T4"][arm][kind]["bin_changed_rate"])
    for arm in ("N", "O", "I", "J"):
        add(f"T5 {arm} MC gated", R["T5"][arm]["mc_acc_gated"], A["T5"][arm]["mc_acc_gated"])
        add(f"T5 {arm} mult ungated", R["T5"][arm]["mult_acc_ungated"], A["T5"][arm]["mult_acc_ungated"])
        if arm != "N":
            add(f"T5 {arm} drift flag", R["T5"][arm]["drift_flag"], A["T5"][arm]["drift_flag"])
    add("verdict", R["verdict"], A["verdict"])
    add("reasons", " | ".join(R["reasons"]), " | ".join(A["reasons"]))
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{w}}  mine {r[1]:<34} reported {r[2]:<34} {r[3]}")
    print("mismatches:", sum(r[3] != "match" for r in rows), "of", len(rows))
    return rows


if __name__ == "__main__":
    R = main()
    skip = ("T3_confusion_truth_rows_para_cols",)
    for k, v in R.items():
        if k in ("T3", "T1", "T2", "T4", "T5", "learned", "trainlogs"):
            continue
        print(k, json.dumps(v, default=str)[:400])
    print("\nT3:", json.dumps(R["T3"], default=str))
    print("\nT1:", json.dumps(R["T1"], default=str))
    print("\nT2:", json.dumps(R["T2"], default=str))
    print("\nT4:", json.dumps(R["T4"], default=str))
    print("\nT5:", json.dumps(R["T5"], default=str))
    print("\nlearned:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "w"} for k, v in R["learned"].items()}))
    print("\ntrainlogs:", json.dumps(R["trainlogs"]))
    print()
    compare(R)
    sys.stdout.flush()

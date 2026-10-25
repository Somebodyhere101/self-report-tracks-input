"""Paper 9 analysis: gates, T1-T5 statistics and the preregistered verdict (PREDICTIONS.md section 6).
CPU only. python analyze.py [--selftest]   -> <out>/analysis.json and a printed summary."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p9lib import load_json, outdir, save_json  # noqa: E402

ROOT = os.path.dirname(HERE)
NBOOT = 10000
GATE = {"min_test_items": 150, "min_acc2": 0.70, "min_kept": 0.95, "min_calib": 0.80, "min_reliab": 0.70,
        "min_manip": 0.90, "min_div": 25, "min_power": 0.80, "t2_min_rows": 20, "t4_max_upper": 0.15,
        "power_a_agree": 0.97, "power_a_div": 0.70}


def boot_ci(stat, n, groups=None, seed=9):
    """Percentile 95% CI of stat(idx) over NBOOT resamples of items (or of groups of items)."""
    rng = np.random.default_rng(seed)
    vals = []
    if groups is None:
        for _ in range(NBOOT):
            vals.append(stat(rng.integers(0, n, n)))
    else:
        keys = sorted(set(groups))
        members = {k: [i for i, g in enumerate(groups) if g == k] for k in keys}
        for _ in range(NBOOT):
            pick = rng.integers(0, len(keys), len(keys))
            vals.append(stat(np.array([i for p in pick for i in members[keys[p]]])))
    vals = np.array(vals, dtype=float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def power_sim(n, p_div, a_agree, a_div, nsim=4000, seed=3):
    """Power of 'lower 95% bound of acc(R) - acc(P) > 0' when P errs exactly on the divergent items and a
    reader is right with prob a_agree on the others and a_div on the divergent ones (normal-approx CI)."""
    rng = np.random.default_rng(seed)
    nd = int(round(p_div * n))
    hits = 0
    for _ in range(nsim):
        d = np.zeros(n)
        d[:nd] = (rng.random(nd) < a_div).astype(float)               # P wrong, R right: +1
        d[nd:] = -(rng.random(n - nd) > a_agree).astype(float)        # P right, R wrong: -1
        se = d.std(ddof=1) / np.sqrt(n)
        hits += (d.mean() - 1.96 * se) > 0
    return hits / nsim


def cond_logit_fit(X, y, l2=1e-2):
    """Conditional logit over 4 candidates (X [n, 4, f] float64, y [n] true index), L2-penalised, fit by
    L-BFGS to convergence (asserted)."""
    from scipy.optimize import minimize
    n = len(y)

    def f(w):
        z = X @ w
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        p = np.exp(z - lse[:, None])
        nll = (lse - z[np.arange(n), y]).mean() + 0.5 * l2 * w @ w
        g = ((X * p[..., None]).sum(1) - X[np.arange(n), y]).mean(0) + l2 * w
        return nll, g
    r = minimize(f, np.zeros(X.shape[2]), jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-9})
    gn = float(np.abs(f(r.x)[1]).max())
    assert gn < 1e-5, f"conditional logit did not converge (max |grad| {gn})"
    return r.x, gn


def feats_raw(it, cands):
    """Input-only features per candidate: is gazetteer, is the 1-hop choice, is the free 1-hop answer, and
    the rotation-averaged 1-hop log-prob clipped at -10 (standardised later with train statistics)."""
    s = np.clip(np.array(it["p1_scores"], float), -10, None)
    return np.array([[float(c == it["country"]), float(c == it["p1"]), float(c == it.get("free1")), s[k]]
                     for k, c in enumerate(cands)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    out = outdir("res_selftest" if a.selftest else "res")
    G = load_json(os.path.join(out, "gen.json"))
    cfg = G["cfg"]
    seeds = [0] if a.selftest else cfg["seeds"]
    scr = {i["city"]: i for i in load_json(os.path.join(outdir("res"), "screen.json"))["items"] if i.get("ok")}
    T = G["tests"]
    cb = G["cb"]
    E = {"N": load_json(os.path.join(out, "eval_N.json"))}
    for arm in ("O", "I", "J", "X"):
        for s in seeds:
            E[f"{arm}{s}"] = load_json(os.path.join(out, f"eval_{arm}_s{s}.json"))
    A = {"seeds": seeds, "gates": {}, "gate_values": {}}

    # ------------------------------------------------------------------ gates
    test_ok = [i for i in scr.values() if i["split"] == ("pilot" if a.selftest else "test")]
    acc2 = float(np.mean([i["acc2"] for i in test_ok]))
    n_test = G["n_test_items"]
    A["gate_values"]["G1"] = {"acc2_test": acc2, "n_test_usable": n_test}
    A["gates"]["G1"] = acc2 >= GATE["min_acc2"] and (a.selftest or n_test >= GATE["min_test_items"])
    kept = [e["answer_kept"] for e in T["T1"]]
    A["gate_values"]["G2"] = {"answer_kept_T1": float(np.mean(kept))}
    A["gates"]["G2"] = np.mean(kept) >= GATE["min_kept"]
    nat_test = [e["x"] for e in T["T3"]]
    all_test = sorted({e["x"] for e in T["T4"]})
    reliab = float(np.mean([G["natural"][c]["reliable"] for c in all_test]))
    known_like = [c for c in all_test if scr[c]["p1"] == scr[c]["country"] and scr[c].get("free1") == scr[c]["country"]]
    calib = float(np.mean([cb.get(c) == scr[c]["country"] for c in known_like]))
    A["gate_values"]["G3"] = {"reliability_natural": reliab, "calibration_known": calib, "n_known_like": len(known_like)}
    A["gates"]["G3"] = reliab >= GATE["min_reliab"] and calib >= GATE["min_calib"]
    man = [e["cb_int"] == e["label"] for e in T["T1"] if e["cb_int"] and e["kind"] == "diff"]
    man_s = [e["cb_int"] == e["label"] for e in T["T1"] if e["cb_int"] and e["kind"] == "same"]
    A["gate_values"]["G4"] = {"T1_diff_cb_equals_donor": float(np.mean(man)), "n": len(man),
                              "T1_same_cb_equals_donor": float(np.mean(man_s)) if man_s else None}
    A["gates"]["G4"] = np.mean(man) >= GATE["min_manip"]

    # ------------------------------------------------------------------ T3 predictors and statistic
    t3 = T["T3"]
    n3 = len(t3)
    truth = [e["label"] for e in t3]
    P = {"gaz": [scr[e["x"]]["country"] for e in t3], "1hop": [scr[e["x"]]["p1"] for e in t3]}
    tr = [e for e in G["train"]["O"]]
    seen, Xtr, ytr = set(), [], []
    for e in tr:
        if e["x"] in seen:
            continue
        seen.add(e["x"])
        Xtr.append(feats_raw(scr[e["x"]], e["cands"]))
        ytr.append(e["cands"].index(e["label"]))
    Xtr = np.array(Xtr)
    mu, sd = Xtr[..., 3].mean(), Xtr[..., 3].std() + 1e-9  # train statistics only

    def feats(it, cands):
        x = feats_raw(it, cands)
        x[:, 3] = (x[:, 3] - mu) / sd
        return x
    Xtr[..., 3] = (Xtr[..., 3] - mu) / sd
    w, gnorm = cond_logit_fit(Xtr, np.array(ytr))
    P["learned"] = [e["cands"][int(np.argmax(feats(scr[e["x"]], e["cands"]) @ w))] for e in t3]
    xs = np.mean([np.array([r["scores"] for r in E[f"X{s}"]["T3"]]) for s in seeds], 0)
    P["external_X"] = [e["cands"][int(np.argmax(xs[k]))] for k, e in enumerate(t3)]
    corr = {k: np.array([p == t for p, t in zip(v, truth)], float) for k, v in P.items()}
    A["T3_baselines"] = {k: float(v.mean()) for k, v in corr.items()}
    A["T3_learned_weights"] = {"w": w.tolist(), "max_abs_grad": gnorm, "n_train_items": len(ytr)}
    best = max(corr, key=lambda k: corr[k].mean())
    n_div = int(n3 - corr[best].sum())
    pw = power_sim(n3, n_div / max(1, n3), GATE["power_a_agree"], GATE["power_a_div"]) if n3 else 0.0
    A["gate_values"]["G5"] = {"best_baseline": best, "n_T3": n3, "n_divergent": n_div,
                              "div_rate": n_div / max(1, n3), "power_strong_reader": pw,
                              "n_div_vs_1hop": int(n3 - corr["1hop"].sum())}
    A["gates"]["G5"] = n_div >= GATE["min_div"] and pw >= GATE["min_power"]

    def refc(x):
        return G["cb"].get(x) or scr[x]["country"]

    def runs_of(arm):
        return [E["N"]] if arm == "N" else [E[f"{arm}{s}"] for s in seeds]
    A["T3"] = {}
    base_stack = np.stack([corr[k] for k in corr])  # [baselines, items]
    div = corr[best] == 0
    for arm in ("N", "O", "I", "J"):
        runs = runs_of(arm)
        cs = [np.array([r["T3"][k]["pick"] == e["label"] for k, e in enumerate(t3)], float) for r in runs]
        accs = [float(c.mean()) for c in cs]

        def stat(idx, cs=cs):
            return np.mean([c[idx].mean() for c in cs]) - base_stack[:, idx].mean(1).max()
        ci = boot_ci(stat, n3) if n3 else [float("nan")] * 2
        nb = [np.array([r["T3"][k]["pick"] == P[best][k] for k in range(n3)], float) for r in runs]
        A["T3"][arm] = {"acc_per_seed": accs, "delta_vs_best_baseline": float(np.mean(accs) - corr[best].mean()),
                        "ci": ci,
                        "divergent_names_cb": float(np.mean([c[div].mean() for c in cs])) if div.any() else None,
                        "divergent_names_baseline": float(np.mean([b[div].mean() for b in nb])) if div.any() else None}
    # ------------------------------------------------------------------ T1, T2, T4 per arm
    A["T1"], A["T2"], A["T4"] = {}, {}, {}
    for arm in ("N", "O", "I", "J"):
        runs = runs_of(arm)
        t1 = T["T1"]
        d = [np.mean([r["T1"][k]["pick"] == e["label"] for r in runs]) for k, e in enumerate(t1) if e["kind"] == "diff"]
        sm = [np.mean([r["T1"][k]["pick"] == e["label"] for r in runs]) for k, e in enumerate(t1) if e["kind"] == "same"]
        inp = [np.mean([r["T1"][k]["pick"] == refc(e["x"]) for r in runs]) for k, e in enumerate(t1)
               if e["kind"] == "diff"]
        A["T1"][arm] = {"acc_diff": float(np.mean(d)) if d else None, "acc_same": float(np.mean(sm)) if sm else None,
                        "names_input_country_diff": float(np.mean(inp)) if inp else None,
                        "ci_diff": boot_ci(lambda idx, d=np.array(d): d[idx].mean(), len(d)) if d else None}
        fam = {}
        for name in ("T2a", "T2c"):
            rows = [(k, e) for k, e in enumerate(T[name]) if e["cb_int"] and e["cb_int"] != refc(e["x"])
                    and e["cb_int"] != scr[e["x"]]["country"]]
            fam[name] = [(e["x"], np.mean([r[name][k]["pick"] == e["cb_int"] for r in runs])) for k, e in rows]
        pooled = fam["T2a"] + fam["T2c"]
        vals = np.array([v for _, v in pooled], float)
        grp = [x for x, _ in pooled]
        evaluable = [n for n in fam if len(fam[n]) >= GATE["t2_min_rows"]]
        A["T2"][arm] = {"n_changed": {n: len(fam[n]) for n in fam},
                        "acc": {n: (float(np.mean([v for _, v in fam[n]])) if fam[n] else None) for n in fam},
                        "pooled_acc": float(vals.mean()) if len(vals) else None,
                        "pooled_ci": boot_ci(lambda idx: vals[idx].mean(), len(vals), groups=grp) if len(vals) else None,
                        "evaluable_families": evaluable}
        t4 = {}
        for kind in ("kl", "norm"):
            rows = [(k, e) for k, e in enumerate(T["T4"]) if e["kind"] == kind and e["cb_int"] and e["nat_cb"]]
            t3idx = {e["x"]: k for k, e in enumerate(t3)}
            diffs = []
            for k, e in rows:
                if e["x"] not in t3idx:
                    continue
                j = t3idx[e["x"]]
                sw_p = np.mean([r["T4"][k]["pick"] != e["cb_int"] for r in runs])
                sw_c = np.mean([r["T3"][j]["pick"] != t3[j]["label"] for r in runs])
                diffs.append(sw_p - sw_c)
            dd = np.array(diffs, float)
            t4[kind] = {"n": len(dd), "switch_increase": float(dd.mean()) if len(dd) else None,
                        "ci": boot_ci(lambda idx: dd[idx].mean(), len(dd)) if len(dd) else None,
                        "cb_changed_rate": float(np.mean([e["cb_int"] != e["nat_cb"] for _, e in rows])) if rows else None}
        A["T4"][arm] = t4
    # ------------------------------------------------------------------ T5
    A["T5"] = {}
    for arm in ("N", "O", "I", "J"):
        runs = runs_of(arm)
        mc = float(np.mean([np.mean([r["T5"][k]["pick"] == e["label"] for k, e in enumerate(T["T5"])]) for r in runs]))
        u2 = float(np.mean([np.mean([u["cont_pred"] == scr[u["city"]]["continent"] for u in r["T5u"]]) for r in runs]))
        u1 = float(np.mean([np.mean([u["p1"] == scr[u["city"]]["country"] for u in r["T5u"]]) for r in runs]))
        A["T5"][arm] = {"mc_acc_gated": mc, "acc2_ungated": u2, "acc1_ungated": u1}
    for arm in ("O", "I", "J"):
        A["T5"][arm]["drift_flag"] = any(A["T5"]["N"][k] - A["T5"][arm][k] > 0.10
                                         for k in ("mc_acc_gated", "acc2_ungated", "acc1_ungated"))
    A["T3"]["X"] = {"acc": A["T3_baselines"]["external_X"]}

    # ------------------------------------------------------------------ verdict
    failing = [g for g, v in A["gates"].items() if not v]
    t2I = A["T2"]["I"]
    t2_eval = len(t2I["evaluable_families"]) > 0
    if failing:
        verdict, reasons = "UNINFORMATIVE", [f"gate {g} failed" for g in failing]
    elif not t2_eval:
        verdict, reasons = "UNINFORMATIVE", ["T2 not evaluable (fewer than 20 changed rows in both families)"]
    else:
        reasons = []
        i_ok = A["T3"]["I"]["ci"][0] > 0
        o_ok = A["T3"]["O"]["ci"][0] > 0
        fam_ok = all(t2I["acc"][f] is not None and t2I["acc"][f] > 1 / 3 for f in t2I["evaluable_families"])
        t2_ok = t2I["pooled_ci"] is not None and t2I["pooled_ci"][0] > 1 / 3 and fam_ok
        t4_ok = A["T4"]["I"]["kl"]["ci"] is not None and A["T4"]["I"]["kl"]["ci"][1] <= GATE["t4_max_upper"]
        if not i_ok:
            reasons.append("I does not beat the best input-only predictor on T3")
        if o_ok:
            reasons.append("O also beats the best input-only predictor on T3 (observational labels suffice)")
        if not t2_ok:
            reasons.append("I is not above 1/3 on unseen intervention families (T2)")
        if not t4_ok:
            reasons.append("I is tamper-sensitive (T4, KL-matched): T2 not credited")
        verdict = "SUPPORTED" if (i_ok and not o_ok and t2_ok and t4_ok) else "NOT SUPPORTED"
    A["verdict"], A["reasons"] = verdict, reasons
    save_json(A, os.path.join(out, "analysis.json"))
    print(json.dumps({"verdict": verdict, "reasons": reasons, "gates": A["gates"], "gate_values": A["gate_values"],
                      "T3": A["T3"], "T3_baselines": A["T3_baselines"]}, indent=1, default=str))


if __name__ == "__main__":
    main()

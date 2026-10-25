"""Paper 9 v3 analysis: v1's decision rule (PREDICTIONS.md section 8) adapted to the v3 task as listed in
PREDICTIONS_V3.md section 6. Reuses v1's boot_ci, power_sim and cond_logit_fit (../code/analyze.py, read-only).
CPU only. Writes ../res_v3/analysis.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
from analyze import GATE, boot_ci, cond_logit_fit, power_sim  # noqa: E402  (v1, frozen)

RES = os.path.join(ROOT, "res_v3_smoke" if os.environ.get("V3_SMOKE") else "res_v3")


def load(p):
    return json.load(open(p, encoding="utf-8"))


def main():
    G = load(os.path.join(RES, "gen.json"))
    scr = load(os.path.join(ROOT, "res_v3", "screen.json"))["items"]
    cfg = G["cfg"]
    seeds = cfg["seeds"]
    T = G["tests"]
    IT = G["items"]
    E = {"N": load(os.path.join(RES, "eval_N.json"))}
    for arm in ("O", "I", "J", "X"):
        for s in seeds:
            E[f"{arm}{s}"] = load(os.path.join(RES, f"eval_{arm}_s{s}.json"))
    A = {"seeds": seeds, "gates": {}, "gate_values": {}}

    def runs_of(arm):
        return [E["N"]] if arm == "N" else [E[f"{arm}{s}"] for s in seeds]

    # ------------------------------------------------------------ gates (v3 adaptations noted)
    test_all = [i for i in scr.values() if i["split"] == ("pilot" if os.environ.get("V3_SMOKE") else "test")]
    acc = float(np.mean([i["correct"] for i in test_all]))
    A["gate_values"]["G1"] = {"accuracy_test": acc, "n_test_usable": G["n_test_items"]}
    A["gates"]["G1"] = acc >= GATE["min_acc2"] and G["n_test_items"] >= GATE["min_test_items"]
    kept = [e["answer_kept"] for e in T["T1"]]
    A["gate_values"]["G2"] = {"answer_kept_T1": float(np.mean(kept))}
    A["gates"]["G2"] = float(np.mean(kept)) >= GATE["min_kept"]
    tk = [e["x"] for e in T["T3"]]
    agree = float(np.mean([IT[k]["bin"] == IT[k]["ent_bin"] for k in tk]))
    A["gate_values"]["G3"] = {"bin_prob_vs_entropy_agreement": agree}  # adaptation: no readout, so no calibration
    A["gates"]["G3"] = agree >= GATE["min_reliab"]
    man = [e["cb_int"] == e["label"] for e in T["T1"] if e["kind"] == "diff"]
    A["gate_values"]["G4"] = {"T1_diff_bin_equals_donor": float(np.mean(man)), "n": len(man)}
    A["gates"]["G4"] = float(np.mean(man)) >= GATE["min_manip"]

    # ------------------------------------------------------------ T3: input-only predictors
    t3 = T["T3"]
    n3 = len(t3)
    truth = np.array([e["label"] for e in t3])
    tr_keys = sorted({e["x"] for e in G["train"]["O"]})
    tr_bins = [IT[k]["bin"] for k in tr_keys]
    maj = int(np.bincount(tr_bins, minlength=4).argmax())
    P = {"para": np.array([IT[e["x"]]["para_bin"] for e in t3]), "majority": np.full(n3, maj)}
    pl = np.array([scr[k]["para_lp2"] for k in tr_keys])
    mu, sd = pl.mean(), pl.std() + 1e-9

    def feats(k):
        pb = IT[k]["para_bin"]
        z = (scr[k]["para_lp2"] - mu) / sd
        rows = []
        for b in range(4):
            oh = [float(b == j) for j in (1, 2, 3)]
            rows.append([float(b == pb), abs(b - pb)] + oh + [o * z for o in oh])
        return np.array(rows)
    Xtr = np.array([feats(k) for k in tr_keys])
    w, gn = cond_logit_fit(Xtr, np.array(tr_bins))
    P["learned"] = np.array([int(np.argmax(feats(e["x"]) @ w)) for e in t3])
    xs = np.mean([np.array([r["scores"] for r in E[f"X{s}"]["T3"]]) for s in seeds], 0)
    P["external_X"] = xs.argmax(1)
    corr = {k: (v == truth).astype(float) for k, v in P.items()}
    A["T3_baselines"] = {k: float(v.mean()) for k, v in corr.items()}
    A["T3_learned"] = {"w": w.tolist(), "max_abs_grad": gn, "n_train_items": len(tr_keys), "majority_bin": maj}
    best = max(corr, key=lambda k: corr[k].mean())
    n_div = int(n3 - corr[best].sum())
    pw = power_sim(n3, n_div / max(1, n3), GATE["power_a_agree"], GATE["power_a_div"]) if n3 else 0.0
    A["gate_values"]["G5"] = {"best_baseline": best, "n_T3": n3, "n_divergent": n_div, "div_rate": n_div / max(1, n3),
                              "power_strong_reader": pw}
    A["gates"]["G5"] = n_div >= GATE["min_div"] and pw >= GATE["min_power"]
    base_stack = np.stack([corr[k] for k in corr])
    div = corr[best] == 0
    A["T3"] = {}
    for arm in ("N", "O", "I", "J"):
        cs = [np.array([r["T3"][k]["pick"] == e["label"] for k, e in enumerate(t3)], float) for r in runs_of(arm)]

        def stat(idx, cs=cs):
            return np.mean([c[idx].mean() for c in cs]) - base_stack[:, idx].mean(1).max()
        nb = [np.array([r["T3"][k]["pick"] == P[best][k] for k in range(n3)], float) for r in runs_of(arm)]
        A["T3"][arm] = {"acc_per_seed": [float(c.mean()) for c in cs],
                        "delta_vs_best_baseline": float(np.mean([c.mean() for c in cs]) - corr[best].mean()),
                        "ci": boot_ci(stat, n3),
                        "divergent_names_cb": float(np.mean([c[div].mean() for c in cs])) if div.any() else None,
                        "divergent_names_baseline": float(np.mean([b[div].mean() for b in nb])) if div.any() else None}
    A["T3"]["X"] = {"acc": A["T3_baselines"]["external_X"]}
    # ------------------------------------------------------------ T1, T2, T4
    A["T1"], A["T2"], A["T4"] = {}, {}, {}
    t3idx = {e["x"]: k for k, e in enumerate(t3)}
    for arm in ("N", "O", "I", "J"):
        runs = runs_of(arm)
        d = [np.mean([r["T1"][k]["pick"] == e["label"] for r in runs]) for k, e in enumerate(T["T1"]) if e["kind"] == "diff"]
        sm = [np.mean([r["T1"][k]["pick"] == e["label"] for r in runs]) for k, e in enumerate(T["T1"]) if e["kind"] == "same"]
        inp = [np.mean([r["T1"][k]["pick"] == IT[e["x"]]["bin"] for r in runs]) for k, e in enumerate(T["T1"]) if e["kind"] == "diff"]
        A["T1"][arm] = {"acc_diff": float(np.mean(d)) if d else None, "acc_same": float(np.mean(sm)) if sm else None,
                        "names_input_bin_diff": float(np.mean(inp)) if inp else None,
                        "ci_diff": boot_ci(lambda idx, d=np.array(d): d[idx].mean(), len(d)) if d else None}
        fam = {}
        for name in ("T2a", "T2c"):
            rows = [(k, e) for k, e in enumerate(T[name]) if e["cb_int"] != IT[e["x"]]["bin"] and e["cb_int"] != IT[e["x"]]["para_bin"]]
            fam[name] = [(e["x"], np.mean([r[name][k]["pick"] == e["cb_int"] for r in runs])) for k, e in rows]
        pooled = fam["T2a"] + fam["T2c"]
        vals = np.array([v for _, v in pooled], float)
        A["T2"][arm] = {"n_changed": {n: len(fam[n]) for n in fam},
                        "acc": {n: (float(np.mean([v for _, v in fam[n]])) if fam[n] else None) for n in fam},
                        "pooled_acc": float(vals.mean()) if len(vals) else None,
                        "pooled_ci": boot_ci(lambda idx: vals[idx].mean(), len(vals), groups=[x for x, _ in pooled]) if len(vals) else None,
                        "evaluable_families": [n for n in fam if len(fam[n]) >= GATE["t2_min_rows"]]}
        t4 = {}
        for kind in ("kl", "norm"):
            diffs = []
            for k, e in enumerate(T["T4"]):
                if e["kind"] != kind or e["x"] not in t3idx:
                    continue
                j = t3idx[e["x"]]
                diffs.append(np.mean([r["T4"][k]["pick"] != e["cb_int"] for r in runs]) -
                             np.mean([r["T3"][j]["pick"] != t3[j]["label"] for r in runs]))
            dd = np.array(diffs, float)
            t4[kind] = {"n": len(dd), "switch_increase": float(dd.mean()) if len(dd) else None,
                        "ci": boot_ci(lambda idx: dd[idx].mean(), len(dd)) if len(dd) else None,
                        "bin_changed_rate": float(np.mean([e["cb_int"] != e["nat"] for e in T["T4"] if e["kind"] == kind]))}
        A["T4"][arm] = t4
    # ------------------------------------------------------------ T5
    A["T5"] = {}
    for arm in ("N", "O", "I", "J"):
        runs = runs_of(arm)
        A["T5"][arm] = {"mc_acc_gated": float(np.mean([np.mean([r["T5"][k]["pick"] == e["label"] for k, e in enumerate(T["T5"])]) for r in runs])),
                        "mult_acc_ungated": float(np.mean([np.mean([u["correct"] for u in r["T5u"]]) for r in runs]))}
    for arm in ("O", "I", "J"):
        A["T5"][arm]["drift_flag"] = any(A["T5"]["N"][k] - A["T5"][arm][k] > 0.10 for k in ("mc_acc_gated", "mult_acc_ungated"))
    # ------------------------------------------------------------ verdict (v1 rule)
    failing = [g for g, v in A["gates"].items() if not v]
    t2I = A["T2"]["I"]
    if failing:
        verdict, reasons = "UNINFORMATIVE", [f"gate {g} failed" for g in failing]
    elif not t2I["evaluable_families"]:
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
    json.dump(A, open(os.path.join(RES, "analysis.json"), "w", encoding="utf-8"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"verdict": verdict, "reasons": reasons, "gates": A["gates"], "gate_values": A["gate_values"],
                      "T3": A["T3"], "T3_baselines": A["T3_baselines"]}, indent=1, default=str))


if __name__ == "__main__":
    main()

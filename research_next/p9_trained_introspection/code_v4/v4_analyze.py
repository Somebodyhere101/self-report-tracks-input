"""Paper 9 v4 analysis (PREDICTIONS_V4.md section 6): the learnability gate, v3's decision rule with the two-bin
thresholds (on the IMMEDIATE-report arms N, O, I, J and X), the TRANSFER secondary, and the immediate-against-delayed
secondary (arm Id = arm I with the delayed report; Nd = untrained, delayed). Reuses v1's GATE, boot_ci and power_sim
(../code/analyze.py, read-only) and v4stats (the input-only predictors). CPU only. Writes ../res_v4/analysis.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from v4stats import bin2, ent_bin2, predictors  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "code"))
from analyze import GATE, boot_ci, power_sim  # noqa: E402  (v1, frozen)
import freeze_v4  # noqa: E402

SMOKE = bool(os.environ.get("V4_SMOKE"))
SUF = "_smoke" if SMOKE else ""
RES = os.path.join(ROOT, "res_v4" + SUF)
PIL = os.path.join(ROOT, "res_pilot_v4" + SUF)
CH2 = 0.5          # chance with two options: the T2 bar (v3: 1/3, the level of a pure tamper detector with four)
TRANSFER_POINT = 2 / 3


def load(p):
    return json.load(open(p, encoding="utf-8"))


def dump(A):
    os.makedirs(RES, exist_ok=True)
    json.dump(A, open(os.path.join(RES, "analysis.json"), "w", encoding="utf-8"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))


def ladder_summary(Ld):
    out = {}
    for name in Ld["order"]:
        r = Ld["rungs"].get(name)
        if not r:
            continue
        row = {"gate_I": r.get("gate_I"), "gate_O": r.get("gate_O"), "pass": r.get("pass")}
        for arm in ("I", "O"):
            if arm in r:
                row[arm] = {k: r[arm][k] for k in ("acc_diff", "acc_same", "acc_nat", "diff_names_recipient_bin", "frac_pick0_nat")}
                row[arm]["final_loss"] = r[arm]["loss_hist"][-1]["loss"] if r[arm]["loss_hist"] else None
        out[name] = row
    return out


def main():
    Ld = load(os.path.join(PIL, "ladder.json"))
    A = {"ladder": ladder_summary(Ld), "ladder_bed": Ld["bed"], "Pstar_pilot": Ld["Pstar_pilot"],
         "Pstar_pilot_acc": Ld["Pstar_pilot_acc"], "ladder_N": {k: v for k, v in Ld["N"].items() if "picks" not in k},
         "chosen": Ld.get("chosen"), "fallback": Ld.get("fallback"), "gates": {}, "gate_values": {},
         "freeze_check": freeze_v4.check(verbose=False), "s_kl_calibration": Ld.get("s_kl")}
    if os.path.exists(os.path.join(RES, "NO_RUNG")):
        A["verdict"] = "UNINFORMATIVE"
        A["reasons"] = [Ld.get("stopped") or "learnability gate: no budget rung passed, and none met the exploratory "
                        "fallback (the main phase was not run)"]
        dump(A)
        print(json.dumps({k: A[k] for k in ("verdict", "reasons", "ladder", "Pstar_pilot", "ladder_bed")}, indent=1, default=str))
        return
    if not os.path.exists(os.path.join(RES, "gen.json")):
        raise SystemExit("main-phase data missing: run the main phase first")
    G = load(os.path.join(RES, "gen.json"))
    scr = load(os.path.join(ROOT, "res_v3", "screen.json"))["items"]
    cfg, M, T = G["cfg"], G["medians"], G["tests"]
    seeds = cfg["seeds"]
    b = lambda k: bin2(scr[k]["lp2"], M["med"])  # noqa: E731
    pb = lambda k: bin2(scr[k]["para_lp2"], M["med"])  # noqa: E731
    E = {a: load(os.path.join(RES, f"eval_{a}.json")) for a in ("N", "Nd")}
    for arm in ("O", "I", "J", "X", "Id"):
        for s in seeds:
            E[f"{arm}{s}"] = load(os.path.join(RES, f"eval_{arm}_s{s}.json"))
    ARMS = ("N", "O", "I", "J", "Nd", "Id")

    def runs_of(arm):
        return [E[arm]] if arm in ("N", "Nd") else [E[f"{arm}{s}"] for s in seeds]

    # ------------------------------------------------------------ gates (v3's, two bins)
    test_all = [i for i in scr.values() if i["split"] == ("pilot" if SMOKE else "test")]
    acc = float(np.mean([i["correct"] for i in test_all]))
    A["gate_values"]["G1"] = {"accuracy_test": acc, "n_test_usable": G["n_test_items"]}
    A["gates"]["G1"] = acc >= GATE["min_acc2"] and G["n_test_items"] >= GATE["min_test_items"]
    kept = [e["answer_kept"] for e in T["T1"]]
    A["gate_values"]["G2"] = {"answer_kept_T1": float(np.mean(kept))}
    A["gates"]["G2"] = float(np.mean(kept)) >= GATE["min_kept"]
    t3 = T["T3"]
    agree = float(np.mean([b(e["x"]) == ent_bin2(scr[e["x"]]["ent2"], M["med_ent"]) for e in t3]))
    A["gate_values"]["G3"] = {"bin_prob_vs_entropy_agreement": agree}
    A["gates"]["G3"] = agree >= GATE["min_reliab"]
    man = [e["cb_int"] == e["label"] for e in T["T1"] if e["kind"] == "diff"]
    A["gate_values"]["G4"] = {"T1_diff_bin_equals_donor": float(np.mean(man)), "n": len(man)}
    A["gates"]["G4"] = float(np.mean(man)) >= GATE["min_manip"]
    A["kl_medians"] = {"T1_all": float(np.median([e["kl"] for e in T["T1"]])),
                       "T1_diff": float(np.median([e["kl"] for e in T["T1"] if e["kind"] == "diff"])),
                       "T4_kl": float(np.median([e["kl"] for e in T["T4"] if e["kind"] == "kl"])),
                       "T4_norm": float(np.median([e["kl"] for e in T["T4"] if e["kind"] == "norm"]))}

    # ------------------------------------------------------------ T3: input-only predictors (P*)
    n3 = len(t3)
    truth = np.array([e["label"] for e in t3])
    tr = sorted([i for i in scr.values() if i["split"] == ("pilot" if SMOKE else "train") and i["correct"]], key=lambda i: i["key"])
    P, info = predictors(tr, [scr[e["x"]] for e in t3], M)
    xs = np.mean([np.array([r["scores"] for r in E[f"X{s}"]["T3"]]) for s in seeds], 0)
    P["external_X"] = xs.argmax(1)
    corr = {k: (np.asarray(v) == truth).astype(float) for k, v in P.items()}
    A["T3_baselines"] = {k: float(v.mean()) for k, v in corr.items()}
    A["T3_learned"] = info
    best = max(corr, key=lambda k: corr[k].mean())
    n_div = int(n3 - corr[best].sum())
    pw = power_sim(n3, n_div / max(1, n3), GATE["power_a_agree"], GATE["power_a_div"]) if n3 else 0.0
    A["gate_values"]["G5"] = {"best_baseline": best, "n_T3": n3, "n_divergent": n_div, "div_rate": n_div / max(1, n3),
                              "power_strong_reader": pw}
    A["gates"]["G5"] = n_div >= GATE["min_div"] and pw >= GATE["min_power"]
    base_stack = np.stack([corr[k] for k in corr])
    div = corr[best] == 0
    A["T3"] = {}
    for arm in ARMS:
        cs = [np.array([r["T3"][k]["pick"] == e["label"] for k, e in enumerate(t3)], float) for r in runs_of(arm)]

        def stat(idx, cs=cs):  # P* re-chosen in every resample
            return np.mean([c[idx].mean() for c in cs]) - base_stack[:, idx].mean(1).max()
        nb = [np.array([r["T3"][k]["pick"] == P[best][k] for k in range(n3)], float) for r in runs_of(arm)]
        A["T3"][arm] = {"acc_per_seed": [float(c.mean()) for c in cs],
                        "delta_vs_best_baseline": float(np.mean([c.mean() for c in cs]) - corr[best].mean()),
                        "ci": boot_ci(stat, n3),
                        "frac_pick0": float(np.mean([np.mean([r["T3"][k]["pick"] == 0 for k in range(n3)]) for r in runs_of(arm)])),
                        "divergent_names_truth": float(np.mean([c[div].mean() for c in cs])) if div.any() else None,
                        "divergent_names_baseline": float(np.mean([x[div].mean() for x in nb])) if div.any() else None}
    A["T3"]["X"] = {"acc": A["T3_baselines"]["external_X"]}

    # ------------------------------------------------------------ T1, T2, T4
    A["T1"], A["T2"], A["T4"], A["TRANSFER"] = {}, {}, {}, {}
    t3idx = {e["x"]: k for k, e in enumerate(t3)}
    for arm in ARMS:
        runs = runs_of(arm)
        d = [np.mean([r["T1"][k]["pick"] == e["label"] for r in runs]) for k, e in enumerate(T["T1"]) if e["kind"] == "diff"]
        sm = [np.mean([r["T1"][k]["pick"] == e["label"] for r in runs]) for k, e in enumerate(T["T1"]) if e["kind"] == "same"]
        inp = [np.mean([r["T1"][k]["pick"] == b(e["x"]) for r in runs]) for k, e in enumerate(T["T1"]) if e["kind"] == "diff"]
        A["T1"][arm] = {"acc_diff": float(np.mean(d)) if d else None, "acc_same": float(np.mean(sm)) if sm else None,
                        "names_input_bin_diff": float(np.mean(inp)) if inp else None, "n_diff": len(d), "n_same": len(sm),
                        "ci_diff": boot_ci(lambda idx, d=np.array(d): d[idx].mean(), len(d)) if d else None}
        fam = {}
        for name in ("T2a", "T2c"):
            rows = [(k, e) for k, e in enumerate(T[name]) if e["cb_int"] != b(e["x"]) and e["cb_int"] != pb(e["x"])]
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
        t4_pass = t4["kl"]["ci"] is not None and t4["kl"]["ci"][1] <= GATE["t4_max_upper"]
        tr_ = {}
        for name in ("T2a", "T2c"):
            v = np.array([x for _, x in fam[name]], float)
            if len(v) < GATE["t2_min_rows"]:
                tr_[name] = {"result": "NOT EVALUABLE", "n": len(v)}
                continue
            ci = boot_ci(lambda idx, v=v: v[idx].mean(), len(v), groups=[x for x, _ in fam[name]])
            if not t4_pass:
                res = "NO (tamper-sensitive: T4 failed)"
            elif ci[0] > CH2 and v.mean() >= TRANSFER_POINT:
                res = "YES"
            elif ci[0] > CH2:
                res = "PARTIAL"
            else:
                res = "NO"
            tr_[name] = {"result": res, "acc": float(v.mean()), "ci": ci, "n": len(v)}
        A["TRANSFER"][arm] = tr_
    # ------------------------------------------------------------ T5
    A["T5"] = {}
    for arm in ARMS:
        runs = runs_of(arm)
        A["T5"][arm] = {"mc_acc_gated": float(np.mean([np.mean([r["T5"][k]["pick"] == e["label"] for k, e in enumerate(T["T5"])]) for r in runs])),
                        "mult_acc_ungated": float(np.mean([np.mean([u["correct"] for u in r["T5u"]]) for r in runs]))}
    for arm in ("O", "I", "J", "Id"):
        A["T5"][arm]["drift_flag"] = any(A["T5"]["N"][k] - A["T5"][arm][k] > 0.10 for k in ("mc_acc_gated", "mult_acc_ungated"))
    # ------------------------------------------------------------ immediate against delayed (secondary, descriptive)
    def paired(name, rows_sel, label_of):
        ks = [k for k, e in enumerate(T[name]) if rows_sel(e)]
        if not ks:
            return None
        a = np.array([np.mean([r[name][k]["pick"] == label_of(T[name][k]) for r in runs_of("I")]) for k in ks])
        d = np.array([np.mean([r[name][k]["pick"] == label_of(T[name][k]) for r in runs_of("Id")]) for k in ks])
        return {"n": len(ks), "imm": float(a.mean()), "del": float(d.mean()), "imm_minus_del": float((a - d).mean()),
                "ci": boot_ci(lambda idx: (a[idx] - d[idx]).mean(), len(ks), groups=[T[name][k]["x"] for k in ks])}
    A["IMM_vs_DEL"] = {
        "T3_natural": paired("T3", lambda e: True, lambda e: e["label"]),
        "T1_diff": paired("T1", lambda e: e["kind"] == "diff", lambda e: e["label"]),
        "T1_same": paired("T1", lambda e: e["kind"] == "same", lambda e: e["label"]),
        "T2a_changed": paired("T2a", lambda e: e["cb_int"] != b(e["x"]) and e["cb_int"] != pb(e["x"]), lambda e: e["cb_int"]),
        "T2c_changed": paired("T2c", lambda e: e["cb_int"] != b(e["x"]) and e["cb_int"] != pb(e["x"]), lambda e: e["cb_int"]),
        "TRANSFER": {"I": A["TRANSFER"]["I"], "Id": A["TRANSFER"]["Id"]},
        "T3_delta_vs_Pstar": {"I": A["T3"]["I"]["delta_vs_best_baseline"], "Id": A["T3"]["Id"]["delta_vs_best_baseline"]}}
    # ------------------------------------------------------------ verdict (v3's rule; T2 bar = 1/2 with two options)
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
        fam_ok = all(t2I["acc"][f] is not None and t2I["acc"][f] > CH2 for f in t2I["evaluable_families"])
        t2_ok = t2I["pooled_ci"] is not None and t2I["pooled_ci"][0] > CH2 and fam_ok
        t4_ok = A["T4"]["I"]["kl"]["ci"] is not None and A["T4"]["I"]["kl"]["ci"][1] <= GATE["t4_max_upper"]
        if not i_ok:
            reasons.append("I does not beat the best input-only predictor on T3")
        if o_ok:
            reasons.append("O also beats the best input-only predictor on T3 (observational labels suffice)")
        if not t2_ok:
            reasons.append("I is not above 1/2 on unseen intervention families (T2)")
        if not t4_ok:
            reasons.append("I is tamper-sensitive (T4, KL-matched): T2 not credited")
        verdict = "SUPPORTED" if (i_ok and not o_ok and t2_ok and t4_ok) else "NOT SUPPORTED"
    A["mode"] = cfg.get("mode", "primary")
    if A["mode"] != "primary":  # exploratory fallback (or smoke): the rule's outcome is reported, never the verdict
        A["exploratory_rule_outcome"] = {"verdict": verdict, "reasons": reasons}
        A["TRANSFER_status"] = "exploratory"
        verdict = "UNINFORMATIVE"
        reasons = [f"learnability gate: no budget rung passed; the main phase ran as the exploratory fallback at {cfg['rung']}"
                   + (" (smoke)" if A["mode"] == "smoke" else "")]
    A["verdict"], A["reasons"] = verdict, reasons
    dump(A)
    print(json.dumps({"verdict": verdict, "reasons": reasons, "gates": A["gates"], "gate_values": A["gate_values"],
                      "T3": A["T3"], "T3_baselines": A["T3_baselines"], "TRANSFER": A["TRANSFER"],
                      "IMM_vs_DEL": A["IMM_vs_DEL"]}, indent=1, default=str))


if __name__ == "__main__":
    main()

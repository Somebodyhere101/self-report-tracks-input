"""Paper 9 independent audit: recompute the v1 bed checks and the v2 analysis from the raw files.

Auditor's own code. It imports no study module (only json, os, numpy) and reads:
  res/screen.json, res_pilot/pilot.json, res_pilot/pilot_gates.json, res_pilot/config_pilot.json,
  res_selftest/gen.json (split check only), res_v2/premise.json, res_v2/analysis_v2.json (comparison only).
Writes audit/recomputed_v1v2.json and prints a report. CPU, numpy only, < 1 GB RAM.

Definitions follow PREDICTIONS.md section 10 (v1 bed checks) and PREDICTIONS_V2.md sections 2-4 (v2).
Rotation scoring follows the preregistered description: rotation r shows cands[(j + r) % 4] at letter j, so
candidate k sits at letter (k - r) % 4; its score is the mean over r of that letter's log-probability.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "audit", "recomputed_v1v2.json")
NB = 10000
SEED = 90210  # auditor's own seed (the study used 11)


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def pct_ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_mean(x, rng, nb=NB, chunk=1000):
    """Percentile bootstrap of the mean of a 1-D array (resampling its elements)."""
    x = np.asarray(x, float)
    n = len(x)
    out = []
    for s in range(0, nb, chunk):
        idx = rng.integers(0, n, (min(chunk, nb - s), n))
        out.append(x[idx].mean(1))
    return pct_ci(np.concatenate(out))


def boot_cluster_mean(x, cluster, rng, nb=NB):
    """Percentile bootstrap of the row mean, resampling whole clusters (pairs)."""
    x = np.asarray(x, float)
    keys = sorted(set(cluster))
    pos = {k: i for i, k in enumerate(keys)}
    s = np.zeros(len(keys))
    c = np.zeros(len(keys))
    for v, k in zip(x, cluster):
        s[pos[k]] += v
        c[pos[k]] += 1
    vals = []
    for _ in range(nb // 1000):
        idx = rng.integers(0, len(keys), (1000, len(keys)))
        vals.append(s[idx].sum(1) / c[idx].sum(1))
    return pct_ci(np.concatenate(vals))


def rot_pick(lp, cands):
    lp = np.asarray(lp, float)
    score = [np.mean([lp[r][(k - r) % 4] for r in range(4)]) for k in range(4)]
    return cands[int(np.argmax(score))], score


def rot_pick_alt(lp, cands):
    """The other rotation convention (candidate k at letter (k + r) % 4): used only to show the check bites."""
    lp = np.asarray(lp, float)
    score = [np.mean([lp[r][(k + r) % 4] for r in range(4)]) for k in range(4)]
    return cands[int(np.argmax(score))]


R = {"v1": {}, "v2": {}, "checks": {}}
rng = np.random.default_rng(SEED)

# =============================================================================================== v1
items = load("res/screen.json")["items"]
names = [i["city"] for i in items]
assert len(names) == len(set(names)), "duplicate city names in screen.json"
city = {i["city"]: i for i in items}
okd = [i for i in items if i.get("ok") is True]
acc2_all = float(np.mean([bool(i["acc2"]) for i in okd]))
n_test_usable = sum(1 for i in okd if i["split"] == "test" and i["acc2"])
split_counts = {s: sum(1 for i in items if i["split"] == s) for s in ("train", "test", "pilot")}
R["v1"]["PG1"] = {"n_screened": len(items), "n_ok": len(okd), "n_acc2_correct": int(sum(bool(i["acc2"]) for i in okd)),
                  "acc2_all": acc2_all, "n_test_usable": n_test_usable, "split_counts_all": split_counts,
                  "pass": acc2_all >= 0.70 and n_test_usable >= 150}

pil = load("res_pilot/pilot.json")
gates = load("res_pilot/pilot_gates.json")
cfgp = load("res_pilot/config_pilot.json")
per = pil["P1"]["per_item"]
pilot_usable = [i for i in okd if i["split"] == "pilot" and i["acc2"]]
assert set(i["city"] for i in pilot_usable) == set(per), "pilot per_item cities != usable pilot cities"
kl = [i for i in pilot_usable if i["p1"] == i["country"] and i["free1"] == i["country"]]
rel_flag_consistent = all(per[c]["reliable"] == (per[c]["cb_country"] == per[c]["cb_capital"]) for c in per)
reliable = {c: bool(per[c]["reliable"]) for c in per}
cbv = {c: (per[c]["cb_country"] if reliable[c] else None) for c in per}  # CB as frozen: None if unreliable
rel_all = float(np.mean([reliable[i["city"]] for i in pilot_usable]))
rel_kl = float(np.mean([reliable[i["city"]] for i in kl]))
cal_frozen = float(np.mean([cbv[i["city"]] == i["country"] for i in kl]))           # unreliable counts as a miss
cal_argmax_only = float(np.mean([per[i["city"]]["cb_country"] == i["country"] for i in kl]))  # ignore reliability
kl_rel = [i for i in kl if reliable[i["city"]]]
cal_among_reliable_kl = float(np.mean([per[i["city"]]["cb_country"] == i["country"] for i in kl_rel]))
relp = [i for i in pilot_usable if reliable[i["city"]]]
margins = sorted(round(per[i["city"]]["margin"], 3) for i in relp)
R["v1"]["PG2"] = {
    "n_pilot_usable": len(pilot_usable), "n_known_like": len(kl), "n_reliable_all": len(relp),
    "reliable_flag_equals_country_capital_agreement": rel_flag_consistent,
    "reliability_all_usable_pilot": rel_all, "reliability_known_like_only": rel_kl,
    "calibration_known_frozen_def": cal_frozen, "n_cal_hits": int(round(cal_frozen * len(kl))),
    "calibration_known_cb_argmax_ignoring_reliability": cal_argmax_only,
    "calibration_known_among_reliable_only": cal_among_reliable_kl, "n_known_like_reliable": len(kl_rel),
    "cb_eq_gaz_among_reliable": float(np.mean([cbv[i["city"]] == i["country"] for i in relp])),
    "cb_eq_p1_among_reliable": float(np.mean([cbv[i["city"]] == i["p1"] for i in relp])),
    "n_divergent_from_p1": sum(1 for i in relp if cbv[i["city"]] != i["p1"]),
    "n_divergent_from_gaz": sum(1 for i in relp if cbv[i["city"]] != i["country"]),
    "margins_match_stored": margins == pil["P1"]["margins_reliable"],
    "pass": cal_frozen >= 0.80 and rel_all >= 0.70,
    "pass_under_any_definition": max(cal_frozen, cal_argmax_only, cal_among_reliable_kl) >= 0.80 and max(rel_all, rel_kl) >= 0.70,
}
P2 = pil["P2"]
l1 = max([B for B in (0, 4, 8, 12, 16) if P2[f"from{B}"]["transfer_to_donor"] >= 0.90] or [0])
t = P2[f"from{l1}"]["transfer_to_donor"]
nrel = P2[f"from{l1}"]["n_reliable"]
swap_keys = [k for k in P2 if k.startswith("from") or k.startswith("single")]
tr = {k: P2[k]["transfer_to_donor"] for k in swap_keys}
R["v1"]["PG3"] = {"l1_by_rule": l1, "l1_stored": pil["chosen"]["l1"], "transfer_at_l1": t, "n_reliable": nrel,
                  "hits": t * nrel, "transfer_min": min(tr.values()), "transfer_max": max(tr.values()),
                  "pass": t >= 0.90,
                  "n_report_from0": P2["N_report_from0"], "n_report_from12": P2["N_report_from12"]}
# other pilot-set parameters, re-derived from the stored aggregates by the frozen rules (information only)
ls = max((6, 10, 14, 18, 22), key=lambda S: (P2[f"single{S}"]["transfer_to_donor"], -S))
best = None
for la in (10, 14, 18):
    for s in (1, 2, 4):
        cell = pil["P3"][f"L{la}_s{s}"]
        if cell["retention"] >= 0.9 and (best is None or cell["change_to_injected"] > best[2] + 1e-9):
            best = (la, s, cell["change_to_injected"])
P5 = pil["P5"]
lr = min((1e-4, 3e-4), key=lambda v: (-(P5[str(v) if str(v) in P5 else f"{v:.4f}"]["acc_diff"] +
                                        P5[str(v) if str(v) in P5 else f"{v:.4f}"]["acc_same"]), v))
rnd = pil["P4"]["random"]
R["v1"]["pilot_params"] = {"ls": ls, "la_s_add": best[:2], "lr": lr, "stored": {k: pil["chosen"][k] for k in ("ls", "la", "s_add", "lr", "s_kl")},
                           "swap_kl_median": pil["P4"]["swap_kl_median"],
                           "random_kl_median_at_max_scale_2": rnd["2"]["kl_median"],
                           "config_pilot_matches_chosen": all(cfgp[k] == pil["chosen"][k] for k in ("l1", "ls", "la", "s_add", "s_kl", "lr"))}
R["v1"]["PG5"] = {"selftest_analysis_exists": os.path.exists(os.path.join(ROOT, "res_selftest", "analysis.json"))}
R["v1"]["gates_file_matches"] = {
    "PG1_acc2": abs(gates["checks"]["PG1_task"]["acc2_all"] - acc2_all) < 1e-12,
    "PG1_n": gates["checks"]["PG1_task"]["n_test_usable"] == n_test_usable,
    "PG2_cal": abs(gates["checks"]["PG2_readout"]["calibration_known"] - cal_frozen) < 1e-12,
    "PG2_rel": abs(gates["checks"]["PG2_readout"]["reliability"] - rel_all) < 1e-12,
    "PG3": gates["checks"]["PG3_manipulation"]["l1"] == l1 and gates["checks"]["PG3_manipulation"]["transfer"] == t,
    "all_pass_false": gates["all_pass"] is False,
}
# chain stopped before main data
res_files = sorted(os.listdir(os.path.join(ROOT, "res")))
R["v1"]["res_contents"] = res_files
R["v1"]["root_config_json_exists"] = os.path.exists(os.path.join(ROOT, "config.json"))
# self-test cities: all pilot split?
G = load("res_selftest/gen.json")
st_cities = set(G["natural"]) | set(G["cb"])
for part in ("tests", "train"):
    for rows in G[part].values():
        for r in rows:
            for k in ("x", "y"):
                if k in r and isinstance(r[k], str) and r[k] in city:
                    st_cities.add(r[k])
R["v1"]["selftest_city_splits"] = {s: sum(1 for c in st_cities if city[c]["split"] == s) for s in ("train", "test", "pilot")}
R["v1"]["selftest_n_cities"] = len(st_cities)

# =============================================================================================== v2
P = load("res_v2/premise.json")
A2 = load("res_v2/analysis_v2.json")
scr = {i["city"]: i for i in okd}
test = sorted(c for c, i in scr.items() if i["split"] == "test" and i["acc2"])
known = [c for c in test if scr[c]["p1"] == scr[c]["country"] and scr[c]["free1"] == scr[c]["country"]]
R["v2"]["items"] = {"n_test": len(test), "n_known": len(known), "known_list_identical": known == P["known"],
                    "n_test_stored": P["n_test"], "bounds": P["bounds"]}
gaz = lambda c: scr[c]["country"]  # noqa: E731

# ---- Part A
variants = ("prior", "R", "city+R")
A = {}
mism = {v: 0 for v in variants}
mism_alt = {v: 0 for v in variants}
gaz_in_cands = all(gaz(c) in scr[c]["cands"] for c in known)
for v in variants:
    hit = []
    for c in known:
        rec = P["A"][v][c]
        p, _ = rot_pick(rec["lp"], scr[c]["cands"])
        mism[v] += p != rec["pick"]
        mism_alt[v] += rot_pick_alt(rec["lp"], scr[c]["cands"]) != rec["pick"]
        hit.append(rec["pick"] == gaz(c))
    A[v] = np.array(hit, float)
n = len(known)
idx_all = rng.integers(0, n, (NB, n))
bA = {v: A[v][idx_all].mean(1) for v in variants}
d1 = A["city+R"] - A["R"]
d2 = A["R"] - A["prior"]
partA = {v: {"calibration": float(A[v].mean()), "hits": int(A[v].sum()), "ci": pct_ci(bA[v])} for v in variants}
partA["cityR_minus_R"] = {"diff": float(d1.mean()), "ci": pct_ci(d1[idx_all].mean(1))}
partA["R_minus_prior"] = {"diff": float(d2.mean()), "ci": pct_ci(d2[idx_all].mean(1))}
partA["R_equals_prior_pick"] = float(np.mean([P["A"]["R"][c]["pick"] == P["A"]["prior"][c]["pick"] for c in known]))
# McNemar-style breakdown of R vs prior
rp, Rp = A["prior"].astype(bool), A["R"].astype(bool)
partA["R_vs_prior_2x2"] = {"both_right": int((rp & Rp).sum()), "R_only": int((~rp & Rp).sum()),
                           "prior_only": int((rp & ~Rp).sum()), "both_wrong": int((~rp & ~Rp).sum())}
partA["pick_rederivation_mismatches"] = mism
partA["pick_mismatches_under_wrong_rotation_convention"] = mism_alt
partA["gazetteer_in_cands_all"] = gaz_in_cands
# CI stability of the H1 bound closest to its threshold (R - prior upper bound vs 0.10), over 5 more seeds
ub = []
for s in range(5):
    rr = np.random.default_rng(1000 + s)
    ub.append(pct_ci(d2[rr.integers(0, n, (NB, n))].mean(1)))
partA["R_minus_prior_ci_other_seeds"] = ub
R["v2"]["A"] = partA

# ---- pairs and Part B rows
pairs = [tuple(p) for p in P["pairs"]]
pair_ok = {"n_pairs": len(pairs), "distinct_recipients": len(set(x for x, _ in pairs)),
           "distinct_donors": len(set(y for _, y in pairs)),
           "max_donor_reuse": max(sum(1 for _, y in pairs if y == d) for d in set(y for _, y in pairs)),
           "x_in_test": all(x in test for x, _ in pairs), "y_known_like": all(y in known for _, y in pairs),
           "same_token_count": all(scr[x]["n"] == scr[y]["n"] for x, y in pairs),
           "same_continent": all(scr[x]["continent"] == scr[y]["continent"] for x, y in pairs),
           "different_country": all(gaz(x) != gaz(y) for x, y in pairs), "x_ne_y": all(x != y for x, y in pairs),
           "n_recipients_known_like": sum(1 for x, _ in pairs if x in known)}
rows = P["B"]
kinds = {}
for r in rows:
    kinds.setdefault(r["kind"], 0)
    kinds[r["kind"]] += 1
pair_ok["row_kinds"] = kinds
pair_ok["rows_pairs_equal_pairs"] = set((r["x"], r["y"]) for r in rows) == set(pairs)
cands_by_pair = {}
cands_consistent = True
both_in = True
mismB = 0
mismB_alt = 0
for r in rows:
    key = (r["x"], r["y"])
    if key in cands_by_pair and cands_by_pair[key] != r["cands"]:
        cands_consistent = False
    cands_by_pair.setdefault(key, r["cands"])
    both_in &= gaz(r["x"]) in r["cands"] and gaz(r["y"]) in r["cands"] and len(set(r["cands"])) == 4
    p, _ = rot_pick(r["lp"], r["cands"])
    mismB += p != r["pick"]
    mismB_alt += rot_pick_alt(r["lp"], r["cands"]) != r["pick"]
pair_ok["cands_identical_across_rows_of_a_pair"] = cands_consistent
pair_ok["cands_contain_both_countries"] = bool(both_in)
pair_ok["pick_rederivation_mismatches"] = mismB
pair_ok["pick_mismatches_under_wrong_rotation_convention"] = mismB_alt
cell = {}
for r in rows:
    cell.setdefault((r["kind"], r["B"]), []).append(r)
for (k, B), rr in cell.items():
    assert k == "report_Rswap" or len(rr) == 200, (k, B, len(rr))
    assert len(set((r["x"], r["y"]) for r in rr)) == len(rr)
R["v2"]["pairs"] = pair_ok

bounds = P["bounds"]
curves = {}
for k in ("report", "readout"):
    curves[k] = {}
    for B in bounds:
        rr = cell[(k, B)]
        don = np.array([r["pick"] == gaz(r["y"]) for r in rr], float)
        rec = np.array([r["pick"] == gaz(r["x"]) for r in rr], float)
        rec_kl = np.array([r["pick"] == gaz(r["x"]) for r in rr if r["x"] in known], float)
        curves[k][B] = {"donor": float(don.mean()), "donor_ci": boot_mean(don, rng), "recipient": float(rec.mean()),
                        "other": float(1 - don.mean() - rec.mean()), "n": len(rr),
                        "recipient_known_like_x_only": float(rec_kl.mean())}
R["v2"]["B"] = curves


def transition(k):
    for B in bounds:
        if curves[k][B]["donor"] <= 0.2:
            return B
    return None


trans = {k: transition(k) for k in ("report", "readout")}
R["v2"]["transition"] = trans

# agreement over all pair x boundary rows
ag, ag_cluster, per_b = [], [], {}
for B in bounds:
    rep = {(r["x"], r["y"]): r for r in cell[("report", B)]}
    ro = {(r["x"], r["y"]): r for r in cell[("readout", B)]}
    assert set(rep) == set(ro)
    obs, cat_rep, cat_ro = [], [], []
    for key in sorted(rep):
        a = rep[key]["pick"] == ro[key]["pick"]
        obs.append(a)
        ag.append(a)
        ag_cluster.append(key)
        x, y = key
        f = lambda p: "donor" if p == gaz(y) else ("recipient" if p == gaz(x) else "other")  # noqa: E731
        cat_rep.append(f(rep[key]["pick"]))
        cat_ro.append(f(ro[key]["pick"]))
    # agreement expected if report and readout picks were independent given this boundary's marginal rates
    # (3 categories; lumping the two fillers makes this an upper bound on chance agreement)
    exp = sum(np.mean([c == g for c in cat_rep]) * np.mean([c == g for c in cat_ro]) for g in ("donor", "recipient", "other"))
    o = float(np.mean(obs))
    per_b[B] = {"observed": o, "expected_if_independent": float(exp),
                "kappa": float((o - exp) / (1 - exp)) if exp < 1 else None,
                "rep_donor_ro_not": int(sum(1 for a, b in zip(cat_rep, cat_ro) if a == "donor" and b != "donor")),
                "ro_donor_rep_not": int(sum(1 for a, b in zip(cat_rep, cat_ro) if b == "donor" and a != "donor"))}
ag = np.array(ag, float)
exp_pooled = float(np.mean([per_b[B]["expected_if_independent"] for B in bounds]))
R["v2"]["agreement"] = {"mean": float(ag.mean()), "n_rows": len(ag), "ci_rows": boot_mean(ag, rng),
                        "ci_pair_cluster": boot_cluster_mean(ag, ag_cluster, rng),
                        "expected_if_independent_pooled": exp_pooled,
                        "kappa_pooled": float((ag.mean() - exp_pooled) / (1 - exp_pooled)),
                        "per_B": per_b}

# ---- Part C
rc = cell[("report_Rswap", -1)]
R["v2"]["C"] = {"report_names_donor": float(np.mean([r["pick"] == gaz(r["y"]) for r in rc])),
                "report_names_recipient": float(np.mean([r["pick"] == gaz(r["x"]) for r in rc])), "n": len(rc),
                "donor_ci": boot_mean(np.array([r["pick"] == gaz(r["y"]) for r in rc], float), rng)}

# ---- hypotheses and verdict (PREDICTIONS_V2.md section 3-4)
H1 = partA["cityR_minus_R"]["ci"][0] > 0.3 and -0.10 <= partA["R_minus_prior"]["ci"][0] and partA["R_minus_prior"]["ci"][1] <= 0.10
adj = None not in trans.values() and abs(bounds.index(trans["report"]) - bounds.index(trans["readout"])) <= 1
ends = all(curves[k][4]["donor"] >= 0.8 and curves[k][20]["donor"] <= 0.2 for k in ("report", "readout"))
H2 = R["v2"]["agreement"]["mean"] >= 0.85 and ends and adj
H3 = R["v2"]["C"]["report_names_donor"] <= 0.10
verdict = "SAME WINDOW" if (H1 and H2) else ("BRIDGE AT CITY TOKENS, but report and readout diverge" if H1 else "NOT CONFIRMED")
R["v2"]["H"] = {"H1": H1, "H2": H2, "H2_parts": {"agreement>=0.85": R["v2"]["agreement"]["mean"] >= 0.85,
                                                 "ends": ends, "transitions_equal_or_adjacent": adj}, "H3": H3}
R["v2"]["verdict"] = verdict

# ---- comparison with res_v2/analysis_v2.json
cmp = []


def c(name, mine, theirs, tol):
    d = abs(mine - theirs)
    cmp.append({"what": name, "recomputed": mine, "reported": theirs, "absdiff": d, "mismatch": d > tol})


for v in variants:
    c(f"A {v} calibration", partA[v]["calibration"], A2["A"][v]["calibration"], 1e-9)
    c(f"A {v} CI low", partA[v]["ci"][0], A2["A"][v]["ci"][0], 0.01)
    c(f"A {v} CI high", partA[v]["ci"][1], A2["A"][v]["ci"][1], 0.01)
for k, kk in (("cityR_minus_R", "cityR_minus_R"), ("R_minus_prior", "R_minus_prior")):
    c(f"A {k} diff", partA[k]["diff"], A2["A"][kk]["diff"], 1e-9)
    c(f"A {k} CI low", partA[k]["ci"][0], A2["A"][kk]["ci"][0], 0.01)
    c(f"A {k} CI high", partA[k]["ci"][1], A2["A"][kk]["ci"][1], 0.01)
c("A R_equals_prior_pick", partA["R_equals_prior_pick"], A2["A"]["R_equals_prior_pick"], 1e-9)
for k in ("report", "readout"):
    for B in bounds:
        th = A2["B"][k][str(B)]
        c(f"B {k} B={B} donor", curves[k][B]["donor"], th["donor"], 1e-9)
        c(f"B {k} B={B} recipient", curves[k][B]["recipient"], th["recipient"], 1e-9)
        c(f"B {k} B={B} donor CI low", curves[k][B]["donor_ci"][0], th["donor_ci"][0], 0.0101)
        c(f"B {k} B={B} donor CI high", curves[k][B]["donor_ci"][1], th["donor_ci"][1], 0.0101)
c("B agreement", R["v2"]["agreement"]["mean"], A2["B_agreement"]["mean"], 1e-9)
c("B agreement CI low", R["v2"]["agreement"]["ci_rows"][0], A2["B_agreement"]["ci"][0], 0.01)
c("B agreement CI high", R["v2"]["agreement"]["ci_rows"][1], A2["B_agreement"]["ci"][1], 0.01)
c("transition report", trans["report"], A2["transition"]["report"], 0)
c("transition readout", trans["readout"], A2["transition"]["readout"], 0)
c("C donor", R["v2"]["C"]["report_names_donor"], A2["C"]["report_names_donor"], 1e-9)
c("C recipient", R["v2"]["C"]["report_names_recipient"], A2["C"]["report_names_recipient"], 1e-9)
c("n_test", len(test), A2["n_test"], 0)
c("n_known", len(known), A2["n_known"], 0)
c("n_pairs", len(pairs), A2["n_pairs"], 0)
hmatch = (A2["H"]["H1_bridge_held_at_city_tokens"] == H1 and A2["H"]["H2_report_and_readout_read_same_window"] == H2
          and A2["H"]["H3_R_swap_invisible_to_report"] == H3)
R["v2"]["comparison"] = {"rows": cmp, "n_mismatch": sum(r["mismatch"] for r in cmp), "H_match": hmatch,
                         "verdict_reported": A2["verdict"], "verdict_match": A2["verdict"].startswith(verdict)}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---------------------------------------------------------------------------------------------- print
print(json.dumps({"v1": R["v1"]}, indent=1, default=str))
pv = {k: v for k, v in R["v2"].items() if k != "comparison"}
print(json.dumps({"v2": pv}, indent=1, default=str))
print("\nComparison with analysis_v2.json:")
for r in cmp:
    flag = "MISMATCH" if r["mismatch"] else "ok"
    print(f"  {flag:8s} {r['what']:32s} recomputed {r['recomputed']!s:>22s}  reported {r['reported']!s:>22s}")
print("mismatches:", R["v2"]["comparison"]["n_mismatch"], "| H match:", hmatch, "| verdict:", verdict,
      "| reported verdict:", A2["verdict"])

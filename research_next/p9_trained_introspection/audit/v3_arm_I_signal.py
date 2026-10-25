"""Exploratory (auditor, not preregistered): is arm I's above-independence accuracy on intervened rows present in
both seeds, and what does it track? Per seed: accuracy vs independence level and a label-permutation p-value
on T1 different-bin rows, T1 same-bin rows, T2a/T2c changed rows and T3; on T2c, whether picks follow the
injected bin c or the measured bin; on T1 diff, whether picks shift with the donor bin."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "res_v3")


def load(n):
    with open(os.path.join(RES, n), encoding="utf-8") as f:
        return json.load(f)


G = load("gen.json")
T = G["tests"]
IT = G["items"]
rng = np.random.default_rng(23)
out = {}


def test(pk, lab, nperm=5000):
    acc = float((pk == lab).mean())
    ind = float(sum((pk == b).mean() * (lab == b).mean() for b in range(4)))
    perm = np.array([(pk == rng.permutation(lab)).mean() for _ in range(nperm)])
    return acc, ind, float((np.sum(perm >= acc) + 1) / (nperm + 1))


sets = {}
d = [k for k, e in enumerate(T["T1"]) if e["kind"] == "diff"]
s = [k for k, e in enumerate(T["T1"]) if e["kind"] == "same"]
sets["T1 diff"] = ("T1", d, np.array([T["T1"][k]["label"] for k in d]))
sets["T1 same"] = ("T1", s, np.array([T["T1"][k]["label"] for k in s]))
for name in ("T2a", "T2c"):
    rows = [k for k, e in enumerate(T[name]) if e["cb_int"] != IT[e["x"]]["bin"] and e["cb_int"] != IT[e["x"]]["para_bin"]]
    sets[f"{name} changed"] = (name, rows, np.array([T[name][k]["cb_int"] for k in rows]))
sets["T3"] = ("T3", list(range(len(T["T3"]))), np.array([e["label"] for e in T["T3"]]))
for arm in ("O", "I", "J"):
    for sd in (0, 1):
        ev = load(f"eval_{arm}_s{sd}.json")
        for nm, (t, idx, lab) in sets.items():
            pk = np.array([ev[t][k]["pick"] for k in idx])
            a, i, p = test(pk, lab)
            out[f"{arm}{sd} {nm}"] = (round(a, 3), round(i, 3), round(p, 4))
            print(f"{arm} seed {sd} {nm:12s} acc {a:.3f} independence {i:.3f} perm p {p:.4f}")
        # T2c: picks vs injected c vs measured bin
        rows = sets["T2c changed"][1]
        c = np.array([T["T2c"][k]["c"] for k in rows])
        cb = np.array([T["T2c"][k]["cb_int"] for k in rows])
        pk = np.array([ev["T2c"][k]["pick"] for k in rows])
        print(f"   T2c changed: cb_int == c on {np.mean(cb == c):.3f}; pick == c {np.mean(pk == c):.3f}; "
              f"pick == cb_int {np.mean(pk == cb):.3f}")
        # all T2c rows: pick == c
        allc = np.array([e["c"] for e in T["T2c"]])
        allp = np.array([r["pick"] for r in ev["T2c"]])
        a, i, p = test(allp, allc)
        print(f"   T2c all 400 rows, pick == injected c: acc {a:.3f} independence {i:.3f} perm p {p:.4f}")
        out[f"{arm}{sd} T2c all pick==c"] = (round(a, 3), round(i, 3), round(p, 4))
with open(os.path.join(ROOT, "audit", "v3_arm_I_signal.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)

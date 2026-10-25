"""Paper 9 v2 (PREDICTIONS_V2.md): where does Qwen3-1.7B hold the two-hop bridge, and do its untrained
self-report and its own causal readout read it from the same place? Test-split cities; base model only.
Reuses the frozen v1 engine (../code: p9lib, build) read-only.

Part A (known-like test cities): readout of the country in a placeholder context when
  prior   nothing is copied;
  R       the post-city region R is copied at every layer (v1's readout);
  city+R  the city tokens and R are copied at every layer.
Part B (pairs: recipient x, known different-country donor y, same token count and continent): the city
  tokens and R are swapped from y at layers >= B, for B in BOUNDS.
  report   the untrained self-report ("Which country did you use...?"), city visible;
  readout  placeholder context fed the swapped run's city tokens and R at every layer (donor's at >= B,
           recipient's below B): the country the swapped computation makes available downstream.
Part C (same pairs): R alone swapped from y at every layer (v1's training family); untrained report.
Writes ../res_v2/premise.json."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
from p9lib import P9, Q_READ_COUNTRY, Q_REPORT, load_json, log, rot_scores, rotate, save_json, seeded  # noqa: E402
from build import fill  # noqa: E402

TINY = bool(os.environ.get("P9_TINY"))
OUT = os.path.join(ROOT, "res_v2_tiny" if TINY else "res_v2")
SCREEN = os.path.join(ROOT, "res_tiny" if TINY else "res", "screen.json")
BOUNDS = [4, 6, 8, 9, 10, 11, 12, 14, 16, 20]
MAX_PAIRS = 200


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "premise.json")
    if os.path.exists(path):
        log("exists, skipping", path)
        return
    t0 = time.time()
    m = P9()
    L = m.L
    scr = {i["city"]: i for i in load_json(SCREEN)["items"] if i.get("ok")}
    test = sorted(c for c, i in scr.items() if i["split"] == "test" and (i["acc2"] or TINY))
    known = [c for c in test if TINY or (scr[c]["p1"] == scr[c]["country"] and scr[c].get("free1") == scr[c]["country"])]
    log(len(test), "usable test cities;", len(known), "known-like")

    def row(city, from_city, ph=False):
        ids, rc, r0 = m.turn1(city, scr[city]["continent"], placeholder=ph)
        return {"ids": ids, "r0": rc if from_city else r0}

    def reqs(ri, q, cands):
        return [(ri, m.tail(q, rotate(cands, r)), r) for r in range(4)]

    # ---------------------------------------------------------------- part A
    unitsA, metaA = [], []
    for c in known:
        cands = scr[c]["cands"]
        unitsA.append({"rows": [row(c, True, ph=True)], "reqs": reqs(0, Q_READ_COUNTRY, cands)})
        metaA.append((c, "prior"))
        p = row(c, False, ph=True)
        p["copy"] = {k: 0 for k in range(L)}
        unitsA.append({"rows": [row(c, False), p], "reqs": reqs(1, Q_READ_COUNTRY, cands)})
        metaA.append((c, "R"))
        p = row(c, True, ph=True)
        p["copy"] = {k: 0 for k in range(L)}
        unitsA.append({"rows": [row(c, True), p], "reqs": reqs(1, Q_READ_COUNTRY, cands)})
        metaA.append((c, "city+R"))
    resA = m.run_units(unitsA)
    A = {}
    for (c, v), r in zip(metaA, resA):
        lp = [r[k][0] for k in range(4)]
        s = rot_scores(lp)
        A.setdefault(v, {})[c] = {"pick": scr[c]["cands"][max(range(4), key=lambda k: s[k])], "lp": lp}
    log("part A done", {v: sum(d[c]["pick"] == scr[c]["country"] for c in d) / len(d) for v, d in A.items()})

    # ---------------------------------------------------------------- pairs
    rng = seeded("v2-pairs")
    idx = {}
    for c in known:
        idx.setdefault((scr[c]["n"], scr[c]["continent"]), []).append(c)
    pairs = []
    for c in sorted(test, key=lambda z: seeded("v2-order", z).random()):
        pool = [d for d in idx.get((scr[c]["n"], scr[c]["continent"]), [])
                if d != c and scr[d]["country"] != scr[c]["country"]]
        if pool:
            pairs.append((c, rng.choice(sorted(pool))))
        if len(pairs) >= MAX_PAIRS:
            break
    log(len(pairs), "pairs")

    # ---------------------------------------------------------------- part B
    unitsB, metaB = [], []
    for x, y in pairs:
        cands = fill([scr[y]["country"], scr[x]["country"]], "v2" + x)
        for B in BOUNDS:
            sw = row(x, True)
            sw["copy"] = {k: 0 for k in range(B, L)}
            unitsB.append({"rows": [row(y, True), sw], "reqs": reqs(1, Q_REPORT, cands)})
            metaB.append((x, y, B, "report", cands))
            sw = row(x, True)
            sw["copy"] = {k: 0 for k in range(B, L)}
            p = row(x, True, ph=True)
            p["copy"] = {k: (0 if k >= B else 1) for k in range(L)}  # donor at >= B (no chained copy), else x
            unitsB.append({"rows": [row(y, True), sw, p], "reqs": reqs(2, Q_READ_COUNTRY, cands)})
            metaB.append((x, y, B, "readout", cands))
    # ---------------------------------------------------------------- part C
    for x, y in pairs:
        cands = fill([scr[y]["country"], scr[x]["country"]], "v2" + x)
        sw = row(x, False)
        sw["copy"] = {k: 0 for k in range(L)}
        unitsB.append({"rows": [row(y, False), sw], "reqs": reqs(1, Q_REPORT, cands)})
        metaB.append((x, y, -1, "report_Rswap", cands))
    resB = m.run_units(unitsB)
    Bres = []
    for (x, y, B, kind, cands), r in zip(metaB, resB):
        lp = [r[k][0] for k in range(4)]
        s = rot_scores(lp)
        Bres.append({"x": x, "y": y, "B": B, "kind": kind, "cands": cands, "lp": lp,
                     "pick": cands[max(range(4), key=lambda k: s[k])]})
    save_json({"A": A, "B": Bres, "pairs": pairs, "known": known, "n_test": len(test), "bounds": BOUNDS,
               "minutes": (time.time() - t0) / 60}, path)
    log("wrote", path, f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

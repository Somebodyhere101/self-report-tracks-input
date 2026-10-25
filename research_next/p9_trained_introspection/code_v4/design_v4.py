"""Paper 9 v4 pre-freeze design numbers (PREDICTIONS_V4.md, sections 3 and 7). CPU only. Reads only the base-model
item screen res_v3/screen.json (each item's own lp2, ent2 and paraphrase lp2; no reporter output, no v4 data).
Writes ../res_v4_design/design.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from v4stats import bin2, ent_bin2, medians, predictors  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "code"))
from analyze import GATE, power_sim  # noqa: E402
from p9lib import seeded  # noqa: E402


def main():
    scr = json.load(open(os.path.join(ROOT, "res_v3", "screen.json"), encoding="utf-8"))["items"]
    M = medians(scr)
    b = lambda i: bin2(i["lp2"], M["med"])  # noqa: E731
    pb = lambda i: bin2(i["para_lp2"], M["med"])  # noqa: E731
    tr = [i for i in scr.values() if i["split"] == "train" and i["correct"]]
    te = sorted([i for i in scr.values() if i["split"] == "test" and i["correct"]], key=lambda i: i["key"])
    D = {"medians": M, "n_train": len(tr), "n_test": len(te),
         "bin_counts_train": np.bincount([b(i) for i in tr], minlength=2).tolist(),
         "bin_counts_test": np.bincount([b(i) for i in te], minlength=2).tolist(),
         "G3_prob_vs_entropy_test": float(np.mean([b(i) == ent_bin2(i["ent2"], M["med_ent"]) for i in te]))}
    P, info = predictors(tr, te, M)
    truth = np.array([b(i) for i in te])
    acc = {k: float((v == truth).mean()) for k, v in P.items()}
    best = max(acc, key=acc.get)
    n_div = int(round((1 - acc[best]) * len(te)))
    D["T3_input_only"] = {"acc": acc, "best": best, "n_div": n_div, "learned": info,
                          "power_strong_reader_n_test": power_sim(len(te), n_div / len(te), GATE["power_a_agree"], GATE["power_a_div"]),
                          "power_strong_reader_n600": power_sim(600, n_div / len(te), GATE["power_a_agree"], GATE["power_a_div"]),
                          "power_weak_reader_0.90_0.60": power_sim(len(te), n_div / len(te), 0.90, 0.60)}
    # T1 / T2c eligibility at two bins (the different-bin donor must also differ from the paraphrase bin)
    by_p = {}
    for i in te:
        by_p.setdefault(i["product"], []).append(i)
    diff = sum(1 for x in te if any(y["key"] != x["key"] and b(y) not in (b(x), pb(x)) for y in by_p[x["product"]]))
    same = sum(1 for x in te if any(y["key"] != x["key"] and b(y) == b(x) for y in by_p[x["product"]]))
    D["T1_rows_test"] = {"diff": diff, "same": same}
    D["T2c_eligible_first400"] = sum(1 for x in te[:400] if b(x) == pb(x))
    # pilot halves (by product) for the learnability ladder
    pil = [i for i in scr.values() if i["split"] == "pilot" and i["correct"]]
    half = {p: ("A" if seeded("v4-half", p).random() < 0.5 else "B") for p in sorted({i["product"] for i in pil})}
    A = sorted([i for i in pil if half[i["product"]] == "A"], key=lambda i: i["key"])
    B = sorted([i for i in pil if half[i["product"]] == "B"], key=lambda i: i["key"])
    byB = {}
    for i in B:
        byB.setdefault(i["product"], []).append(i)
    byA = {}
    for i in A:
        byA.setdefault(i["product"], []).append(i)
    PB, infoB = predictors(A, B, M)
    tB = np.array([b(i) for i in B])
    accB = {k: float((v == tB).mean()) for k, v in PB.items()}
    D["pilot"] = {"n_A": len(A), "n_B": len(B), "bin_counts_B": np.bincount(tB, minlength=2).tolist(),
                  "A_recipients_diff": sum(1 for x in A if any(y["key"] != x["key"] and b(y) not in (b(x), pb(x)) for y in byA[x["product"]])),
                  "A_recipients_same": sum(1 for x in A if any(y["key"] != x["key"] and b(y) == b(x) for y in byA[x["product"]])),
                  "B_rows_diff": sum(1 for x in B if any(y["key"] != x["key"] and b(y) not in (b(x), pb(x)) for y in byB[x["product"]])),
                  "B_rows_same": sum(1 for x in B if any(y["key"] != x["key"] and b(y) == b(x) for y in byB[x["product"]])),
                  "Pstar_pilot_acc": accB, "Pstar_pilot": max(accB.values()), "learned": infoB}
    os.makedirs(os.path.join(ROOT, "res_v4_design"), exist_ok=True)
    json.dump(D, open(os.path.join(ROOT, "res_v4_design", "design.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(D, indent=1))


if __name__ == "__main__":
    main()

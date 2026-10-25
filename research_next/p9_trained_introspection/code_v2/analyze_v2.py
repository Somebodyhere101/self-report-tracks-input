"""Paper 9 v2 analysis (PREDICTIONS_V2.md section 4). CPU. Writes ../res_v2/analysis_v2.json and prints it."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB = 10000


def ci(stat, n, seed=11):
    rng = np.random.default_rng(seed)
    v = np.array([stat(rng.integers(0, n, n)) for _ in range(NB)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


TINY = bool(os.environ.get("P9_TINY"))
RES = os.path.join(ROOT, "res_v2_tiny" if TINY else "res_v2")


def main():
    P = json.load(open(os.path.join(RES, "premise.json"), encoding="utf-8"))
    scr = {i["city"]: i for i in json.load(open(os.path.join(ROOT, "res_tiny" if TINY else "res", "screen.json"),
                                                encoding="utf-8"))["items"] if i.get("ok")}
    R = {"n_test": P["n_test"], "n_known": len(P["known"]), "n_pairs": len(P["pairs"])}
    # ------------------------------------------------------------ part A
    known = P["known"]
    hit = {v: np.array([P["A"][v][c]["pick"] == scr[c]["country"] for c in known], float) for v in ("prior", "R", "city+R")}
    R["A"] = {v: {"calibration": float(h.mean()), "ci": ci(lambda i, h=h: h[i].mean(), len(h))} for v, h in hit.items()}
    d1 = hit["city+R"] - hit["R"]
    d2 = hit["R"] - hit["prior"]
    R["A"]["cityR_minus_R"] = {"diff": float(d1.mean()), "ci": ci(lambda i: d1[i].mean(), len(d1))}
    R["A"]["R_minus_prior"] = {"diff": float(d2.mean()), "ci": ci(lambda i: d2[i].mean(), len(d2))}
    R["A"]["R_equals_prior_pick"] = float(np.mean([P["A"]["R"][c]["pick"] == P["A"]["prior"][c]["pick"] for c in known]))
    # ------------------------------------------------------------ part B
    rows = P["B"]
    by = {}
    for r in rows:
        by.setdefault((r["kind"], r["B"]), []).append(r)
    curves = {}
    for kind in ("report", "readout"):
        curves[kind] = {}
        for B in P["bounds"]:
            rr = by[(kind, B)]
            don = np.array([r["pick"] == scr[r["y"]]["country"] for r in rr], float)
            rec = np.array([r["pick"] == scr[r["x"]]["country"] for r in rr], float)
            curves[kind][B] = {"donor": float(don.mean()), "donor_ci": ci(lambda i, d=don: d[i].mean(), len(don)),
                               "recipient": float(rec.mean()), "n": len(rr)}
    R["B"] = curves
    agree = []
    for B in P["bounds"]:
        rep = {(r["x"], r["y"]): r["pick"] for r in by[("report", B)]}
        for r in by[("readout", B)]:
            agree.append(rep[(r["x"], r["y"])] == r["pick"])
    agree = np.array(agree, float)
    R["B_agreement"] = {"mean": float(agree.mean()), "ci": ci(lambda i: agree[i].mean(), len(agree))}

    def transition(kind):
        for B in P["bounds"]:
            if curves[kind][B]["donor"] <= 0.2:
                return B
        return None
    t_rep, t_ro = transition("report"), transition("readout")
    R["transition"] = {"report": t_rep, "readout": t_ro}
    # ------------------------------------------------------------ part C
    rc = by[("report_Rswap", -1)]
    R["C"] = {"report_names_donor": float(np.mean([r["pick"] == scr[r["y"]]["country"] for r in rc])),
              "report_names_recipient": float(np.mean([r["pick"] == scr[r["x"]]["country"] for r in rc])), "n": len(rc)}
    # ------------------------------------------------------------ hypotheses and verdict
    bl = P["bounds"]
    H1 = R["A"]["cityR_minus_R"]["ci"][0] > 0.3 and -0.10 <= R["A"]["R_minus_prior"]["ci"][0] and \
        R["A"]["R_minus_prior"]["ci"][1] <= 0.10
    adj = t_rep is not None and t_ro is not None and abs(bl.index(t_rep) - bl.index(t_ro)) <= 1
    H2 = R["B_agreement"]["mean"] >= 0.85 and adj and all(
        curves[k][bl[0]]["donor"] >= 0.8 and curves[k][bl[-1]]["donor"] <= 0.2 for k in ("report", "readout"))
    H3 = R["C"]["report_names_donor"] <= 0.10
    R["H"] = {"H1_bridge_held_at_city_tokens": H1, "H2_report_and_readout_read_same_window": H2,
              "H3_R_swap_invisible_to_report": H3}
    if H1 and H2:
        R["verdict"] = "SAME WINDOW: this bed cannot separate reading the computation from reconstructing from the input"
    elif H1:
        R["verdict"] = "BRIDGE AT CITY TOKENS, but report and readout diverge (see H2 details)"
    else:
        R["verdict"] = "NOT CONFIRMED (see H1 details)"
    json.dump(R, open(os.path.join(RES, "analysis_v2.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()

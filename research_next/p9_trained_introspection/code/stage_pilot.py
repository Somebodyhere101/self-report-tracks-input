"""Paper 9 pilot (before the freeze; pilot cities only, never analysed later).

P1 causal natural-bridge readout: reliability (country vs capital readout agree), calibration on known
   cities, divergence from the 1-hop forced choice.
P2 swap families: CB of the swapped run = donor's bridge, for swaps of R at layers >= B and at one layer S;
   the untrained report on swaps.
P3 additive country vectors: change rate and answer retention over a layer x strength grid.
P4 KL of the training swap and of random directions (scale for the KL-matched tamper control).
P5 learning-rate check: arm I trained on half of the pilot cities, tested on the other half.
Writes res_pilot/pilot.json and res_pilot/config_pilot.json (parameters chosen by the fixed rules below).
"""
import argparse
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p9lib import P9, TINY, log, outdir, save_json, seeded  # noqa: E402
from build import cont_of, fill, readout_result, unit_natural, with_clean  # noqa: E402
from items import (base_norm, donor_index, example_unit, is_known, jvec_vectors, load_screen,  # noqa: E402
                   natural_readouts, pick_donor, random_unit, ref, usable)
import trainlib  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = outdir("res_pilot")
BOUNDARIES = [0, 12] if TINY else [0, 4, 8, 12, 16, 20]
SINGLES = [10] if TINY else [6, 10, 14, 18, 22]
ADD_LAYERS = [14] if TINY else [10, 14, 18]
ADD_SCALES = [1] if TINY else [1, 2, 4]
CAP_LAYERS = sorted(set(ADD_LAYERS))


def rate(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-recip", type=int, default=120)
    ap.add_argument("--lr-steps", type=int, default=150)
    a = ap.parse_args()
    if TINY:
        a.max_recip, a.lr_steps = 4, 3
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "pilot.json")):
        log("exists, skipping", os.path.join(OUT, "pilot.json"))
        return
    t0 = time.time()
    m = P9()
    L = m.L
    scr = load_screen()
    pilot = [it for it in scr.values() if it["split"] == "pilot" and usable(it)]
    pilot.sort(key=lambda it: it["city"])
    log(len(pilot), "usable pilot items")
    R = {"n_pilot_usable": len(pilot)}

    # P1 natural readouts
    nr = natural_readouts(m, pilot)
    cb = {c: (r["cb"] if r["reliable"] else None) for c, r in nr.items()}
    rel = [r["reliable"] for r in nr.values()]
    known_like = [it for it in pilot if it["p1"] == it["country"] and it["free1"] == it["country"]]
    R["P1"] = {
        "reliability": rate(rel),
        "cb_eq_gaz_among_reliable": rate([cb[it["city"]] == it["country"] for it in pilot if cb[it["city"]]]),
        "cb_eq_p1_among_reliable": rate([cb[it["city"]] == it["p1"] for it in pilot if cb[it["city"]]]),
        "calibration_known_cb_eq_gaz": rate([cb[it["city"]] == it["country"] for it in known_like]),
        "n_known_like": len(known_like),
        "n_divergent_from_p1": sum(1 for it in pilot if cb[it["city"]] and cb[it["city"]] != it["p1"]),
        "n_divergent_from_gaz": sum(1 for it in pilot if cb[it["city"]] and cb[it["city"]] != it["country"]),
        "margins_reliable": sorted(round(nr[it["city"]]["margin"], 3) for it in pilot if cb[it["city"]]),
        "per_item": {c: {k: v for k, v in r.items() if k in ("cb_country", "cb_capital", "reliable", "margin")}
                     for c, r in nr.items()},
    }
    log("P1", {k: v for k, v in R["P1"].items() if k not in ("per_item", "margins_reliable")})

    known = [it for it in pilot if is_known(it, cb)]
    idx = donor_index(known)
    rng = seeded("pilot-pairs")
    recips = pilot[:]
    rng.shuffle(recips)
    pairs = []
    for x in recips:
        y = pick_donor(x, idx, cb, "diff", rng)
        if y:
            pairs.append((x, y))
        if len(pairs) >= a.max_recip:
            break
    log(len(known), "known donors;", len(pairs), "diff pairs")
    if not pairs:
        raise SystemExit("no recipient-donor pairs among pilot items")

    # P2 swap families
    R["P2"] = {}
    for kind, Ls in [("from", BOUNDARIES), ("single", SINGLES)]:
        for B in Ls:
            layers = list(range(B, L)) if kind == "from" else [B]
            exs = [{"arm": "swap", "x": x["city"], "y": y["city"], "cont": x["continent"], "layers": layers,
                    "cands": fill([y["country"], ref(x, cb), x["country"]], f"p2{x['city']}")} for x, y in pairs]
            res = m.run_units([example_unit(m, ex, {}, None, readout=True) for ex in exs])
            rr = [readout_result(r, ex["cands"]) for ex, r in zip(exs, res)]
            ok = [r for r in rr if r["reliable"]]
            R["P2"][f"{kind}{B}"] = {
                "reliability": rate([r["reliable"] for r in rr]),
                "transfer_to_donor": rate([r["cb"] == y["country"] for r, (x, y) in zip(rr, pairs) if r["reliable"]]),
                "stay_recipient": rate([r["cb"] == ref(x, cb) for r, (x, y) in zip(rr, pairs) if r["reliable"]]),
                "n_reliable": len(ok)}
            log("P2", kind, B, R["P2"][f"{kind}{B}"])
    # untrained report on swaps
    for B in (0, 12):
        exs = [{"arm": "swap", "x": x["city"], "y": y["city"], "cont": x["continent"], "layers": list(range(B, L)),
                "cands": fill([y["country"], ref(x, cb), x["country"]], f"p2{x['city']}")} for x, y in pairs]
        rep = trainlib.eval_reports(m, exs, {}, None, lora=False)
        R["P2"][f"N_report_from{B}"] = {
            "names_donor": rate([r["pick"] == y["country"] for r, (x, y) in zip(rep, pairs)]),
            "names_recipient": rate([r["pick"] == ref(x, cb) for r, (x, y) in zip(rep, pairs)])}
        log("P2 untrained report", B, R["P2"][f"N_report_from{B}"])

    # P3 additive vectors: capture natural R states
    units = [unit_natural(m, it["city"], it["continent"]) for it in pilot]
    _, caps = m.run_units(units, capture=tuple(CAP_LAYERS))
    capc = {l: {it["city"]: c[l][0] for it, c in zip(pilot, caps)} for l in CAP_LAYERS}
    bn = {l: base_norm(capc[l], [(x["city"], y["city"]) for x, y in pairs]) for l in CAP_LAYERS}
    jv = {l: jvec_vectors(capc[l], pilot, min_items=2) for l in CAP_LAYERS}
    R["P3"] = {"base_norm": bn, "n_jvec": {l: len(v) for l, v in jv.items()}}
    best = None
    for la in ADD_LAYERS:
        for s in ADD_SCALES:
            exs = []
            for x in [p[0] for p in pairs]:
                rx = ref(x, cb)
                opts = sorted(c for c in jv[la] if c not in (rx, x["country"]) and cont_of(c) == x["continent"])
                if not opts:
                    continue
                c = seeded("p3c", x["city"]).choice(opts)
                exs.append({"arm": "add", "x": x["city"], "cont": x["continent"], "c": c, "layer": la,
                            "vec": jv[la][c] * (s * bn[la]), "cands": fill([c, rx, x["country"]], f"p3{x['city']}")})
            res = m.run_units([example_unit(m, ex, {}, None, readout=True) for ex in exs])
            rr = [readout_result(r, ex["cands"]) for ex, r in zip(exs, res)]
            klu = [with_clean(m, example_unit(m, ex, {}, None, rotations=()), ex["x"], ex["cont"]) for ex in exs]
            kr = m.kl_units(klu)
            cell = {"n": len(exs), "reliability": rate([r["reliable"] for r in rr]),
                    "change_to_injected": rate([r["cb"] == ex["c"] for r, ex in zip(rr, exs) if r["reliable"]]),
                    "retention": rate([k[1] == k[2] for k in kr]),
                    "kl_median": statistics.median(k[0] for k in kr)}
            R["P3"][f"L{la}_s{s}"] = cell
            log("P3", la, s, cell)
            if cell["retention"] >= 0.9 and (best is None or cell["change_to_injected"] > best[2] + 1e-9):
                best = (la, s, cell["change_to_injected"])

    # P4 KL: training swap vs random directions at the chosen add layer
    l1 = max([B for B in BOUNDARIES if B <= 16 and R["P2"][f"from{B}"]["transfer_to_donor"] >= 0.9] or [0])
    la, s_add = (best[0], best[1]) if best else (14, 1)
    exs = [{"arm": "swap", "x": x["city"], "y": y["city"], "cont": x["continent"], "layers": list(range(l1, L)),
            "cands": fill([y["country"], ref(x, cb), x["country"]], "k")} for x, y in pairs]
    kr = m.kl_units([with_clean(m, example_unit(m, ex, {}, None, rotations=()), ex["x"], ex["cont"]) for ex in exs])
    swap_kl = statistics.median(k[0] for k in kr)
    R["P4"] = {"swap_kl_median": swap_kl, "swap_retention": rate([k[1] == k[2] for k in kr]), "random": {}}
    for sc in ((0.5, 1) if TINY else (0.0625, 0.125, 0.25, 0.5, 1, 2)):
        exs = [{"arm": "add", "x": x["city"], "cont": x["continent"], "layer": la,
                "vec": random_unit(m.d, x["city"]) * (sc * bn[la]), "cands": x["cands"]} for x, _ in pairs]
        kr = m.kl_units([with_clean(m, example_unit(m, ex, {}, None, rotations=()), ex["x"], ex["cont"]) for ex in exs])
        R["P4"]["random"][str(sc)] = {"kl_median": statistics.median(k[0] for k in kr),
                                      "retention": rate([k[1] == k[2] for k in kr])}
        log("P4 random", sc, R["P4"]["random"][str(sc)])
    pts = sorted((float(k), v["kl_median"]) for k, v in R["P4"]["random"].items())
    s_kl = pts[0][0]
    for (s0, k0), (s1, k1) in zip(pts, pts[1:]):
        if k0 <= swap_kl <= k1 and k1 > k0 > 0:
            f = (math.log(swap_kl) - math.log(k0)) / (math.log(k1) - math.log(k0))
            s_kl = math.exp(math.log(s0) + f * (math.log(s1) - math.log(s0)))
            break
    else:
        if swap_kl > pts[-1][1]:
            s_kl = pts[-1][0]
    R["P4"]["s_kl"] = s_kl

    # single-layer family for T2: the layer with the largest transfer (ties: earlier)
    ls = max(SINGLES, key=lambda S: (R["P2"][f"single{S}"]["transfer_to_donor"], -S))

    # P5 learning-rate check (arm I, half A -> half B)
    half = {it["city"]: ("A" if seeded("half", it["city"]).random() < 0.5 else "B") for it in pilot}
    cfg = {"l1": l1, "la": la, "s_add": s_add, "base_norm": bn[la], "steps": a.lr_steps, "bs": 8}

    def i_examples(h, n, key):
        r = seeded("p5", key)
        xs = [it for it in pilot if half[it["city"]] == h]
        kidx = donor_index([it for it in known if half[it["city"]] == h])
        out = []
        for t in range(10 * n):
            if len(out) >= n:
                break
            kind = "diff" if t % 2 == 0 else "same"
            x = r.choice(xs)
            y = pick_donor(x, kidx, cb, kind, r)
            if y:
                out.append({"arm": "I", "x": x["city"], "y": y["city"], "cont": x["continent"], "label": y["country"],
                            "kind": kind, "cands": fill([y["country"], ref(x, cb), x["country"]], f"p5{key}{t}")})
        return out
    trA = i_examples("A", 16 if TINY else 400, "trainA")
    teB = i_examples("B", 8 if TINY else 160, "testB")
    R["P5"] = {"n_train": len(trA), "n_test": len(teB)}
    for lr in ((1e-4,) if TINY else (1e-4, 3e-4)):
        cfg["lr"] = lr
        state, hist = trainlib.train(m, "I", trA, cfg, seed=0)
        rep = trainlib.eval_reports(m, teB, cfg, None, lora=True)
        acc_d = rate([r["pick"] == ex["label"] for r, ex in zip(rep, teB) if ex["kind"] == "diff"])
        acc_s = rate([r["pick"] == ex["label"] for r, ex in zip(rep, teB) if ex["kind"] == "same"])
        natB = [it for it in pilot if half[it["city"]] == "B" and cb[it["city"]]]
        nat_ex = [{"arm": "nat", "x": it["city"], "cont": it["continent"], "cands": it["cands"],
                   "label": cb[it["city"]]} for it in natB]
        repn = trainlib.eval_reports(m, nat_ex, cfg, None, lora=True)
        acc_n = rate([r["pick"] == ex["label"] for r, ex in zip(repn, nat_ex)])
        R["P5"][str(lr)] = {"acc_diff": acc_d, "acc_same": acc_s, "acc_natural_cb": acc_n,
                            "loss_first": hist[0]["loss"], "loss_last": hist[-1]["loss"]}
        log("P5 lr", lr, R["P5"][str(lr)])
        m.remove_lora()
        torch.cuda.empty_cache()
    nz = lambda v: 0.0 if v != v else v  # a missing (NaN) accuracy counts as 0
    lr_best = max(((1e-4,) if TINY else (1e-4, 3e-4)),
                  key=lambda v: (nz(R["P5"][str(v)]["acc_diff"]) + nz(R["P5"][str(v)]["acc_same"]), -v))

    chosen = {"l1": l1, "ls": ls, "la": la, "s_add": s_add, "s_kl": s_kl, "lr": lr_best,
              "base_norm_pilot": bn[la], "rules": {
                  "l1": "largest B in 0..16 (step 4) with swap-from-B transfer >= 0.90",
                  "ls": "single layer with the largest transfer (ties: earlier)",
                  "la,s_add": "largest change-to-injected rate with answer retention >= 0.90 (ties: first in grid)",
                  "s_kl": "random-direction scale (x base_norm) whose median R-KL equals the training swap's (log-log interpolation)",
                  "lr": "higher acc_diff + acc_same on pilot half B (ties: lower lr)"}}
    R["chosen"] = chosen
    R["minutes"] = (time.time() - t0) / 60
    save_json(R, os.path.join(OUT, "pilot.json"))
    save_json(chosen, os.path.join(OUT, "config_pilot.json"))
    log("CHOSEN", chosen)


if __name__ == "__main__":
    main()

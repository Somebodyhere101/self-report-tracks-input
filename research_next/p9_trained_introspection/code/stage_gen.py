"""Paper 9 stage 1: ground truth and example lists (base model only; no reporter exists yet).

1. Causal natural bridge (CB) of every train and test item: placeholder readout of its natural run.
2. Known donors (2-hop right, free and forced-choice 1-hop = gazetteer, CB reliable = gazetteer).
3. Country concept vectors at layer la from train items (J arm and T2c), and the swap norm at la.
4. Test sets (test cities only): T1 training-family swaps (diff and same donors), T2a single-layer swaps,
   T2c additive country vectors, T3 natural items, T4 random directions (KL-matched and norm-matched),
   T5 general questions. Every intervened test row gets its own CB readout.
5. Training lists (train cities only), the same size for every arm: O and X (natural, label CB),
   I (training-family swaps, label = donor's CB), J (additive country vectors, label = injected country).
Writes <out>/gen.json and <out>/jvec.pt.
"""
import argparse
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p9lib import P9, TINY, load_json, log, outdir, save_json, seeded  # noqa: E402
from build import cont_of, fill, readout_result, unit_natural, with_clean  # noqa: E402
from items import (base_norm, donor_index, example_unit, is_known, jvec_vectors, load_screen,  # noqa: E402
                   natural_readouts, pick_donor, ref, usable)
from mcq import MCQ  # noqa: E402

ROOT = os.path.dirname(HERE)


def readouts(m, exs, cfg, jvec):
    res = m.run_units([example_unit(m, ex, cfg, jvec, readout=True) for ex in exs])
    return [readout_result(r, ex["cands"]) for ex, r in zip(exs, res)]


def retention(m, exs, cfg, jvec):
    kr = m.kl_units([with_clean(m, example_unit(m, ex, cfg, jvec, rotations=()), ex["x"], ex["cont"]) for ex in exs])
    return [{"kl": k[0], "kept": k[1] == k[2]} for k in kr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.selftest:
        out = outdir("res_selftest")
        cfg = load_json(os.path.join(outdir("res_pilot"), "config_pilot.json"))
        cfg.update({"n_train": 16 if TINY else 64, "jvec_min_items": 2, "max_test": 6 if TINY else 40, "seeds": [0]})
    else:
        out = outdir("res")
        cfg = load_json(os.path.join(ROOT, "config.json"))
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "gen.json")):
        log("exists, skipping", os.path.join(out, "gen.json"))
        return
    m = P9()
    L = m.L
    scr = load_screen()
    if a.selftest:
        pil = sorted((it for it in scr.values() if it["split"] == "pilot" and usable(it)), key=lambda i: i["city"])
        half = {it["city"]: ("train" if seeded("half", it["city"]).random() < 0.5 else "test") for it in pil}
        train = [it for it in pil if half[it["city"]] == "train"]
        test = [it for it in pil if half[it["city"]] == "test"][:cfg["max_test"]]
    else:
        train = sorted((it for it in scr.values() if it["split"] == "train" and usable(it)), key=lambda i: i["city"])
        test = sorted((it for it in scr.values() if it["split"] == "test" and usable(it)), key=lambda i: i["city"])
    log(len(train), "train items,", len(test), "test items")
    G = {"cfg": cfg, "n_train_items": len(train), "n_test_items": len(test)}

    # 1. natural readouts
    nat = natural_readouts(m, train + test)
    cb = {c: (r["cb"] if r["reliable"] else None) for c, r in nat.items()}
    G["natural"] = nat
    log("natural readouts done; reliable", sum(1 for v in cb.values() if v) / len(cb))

    # 2. donors
    kn_tr = [it for it in train if is_known(it, cb)]
    kn_te = [it for it in test if is_known(it, cb)]
    idx_tr, idx_te = donor_index(kn_tr), donor_index(kn_te)
    G["n_known_train"], G["n_known_test"] = len(kn_tr), len(kn_te)

    # 3. concept vectors and the swap norm at la (train items only)
    la = cfg["la"]
    _, caps = m.run_units([unit_natural(m, it["city"], it["continent"]) for it in train], capture=(la,))
    capc = {it["city"]: c[la][0] for it, c in zip(train, caps)}
    jvec = jvec_vectors(capc, train, min_items=cfg.get("jvec_min_items", 3))
    rng = seeded("gen-norm")
    npairs = []
    for x in train:
        y = pick_donor(x, idx_tr, cb, "diff", rng)
        if y:
            npairs.append((x["city"], y["city"]))
    cfg["base_norm"] = base_norm(capc, npairs[:300])
    G["base_norm"], G["n_jvec"] = cfg["base_norm"], len(jvec)
    torch.save({"jvec": jvec, "base_norm": cfg["base_norm"], "la": la}, os.path.join(out, "jvec.pt"))
    log("jvec", len(jvec), "countries; base norm", cfg["base_norm"])
    del caps, capc

    # 4. test sets
    T = {}
    r1 = seeded("T1")
    T["T1"] = []
    for x in test:
        for kind in ("diff", "same"):
            y = pick_donor(x, idx_te, cb, kind, r1)
            if y:
                T["T1"].append({"arm": "swap", "x": x["city"], "y": y["city"], "cont": x["continent"], "kind": kind,
                                "layers": list(range(cfg["l1"], L)), "label": y["country"],
                                "cands": fill([y["country"], ref(x, cb), x["country"]], "T1" + x["city"] + kind)})
    r2 = seeded("T2a")
    t1d = {e["x"]: e["y"] for e in T["T1"] if e["kind"] == "diff"}
    T["T2a"] = []
    for x in test:
        y = pick_donor(x, idx_te, cb, "diff", r2, exclude=(t1d.get(x["city"]),)) or \
            pick_donor(x, idx_te, cb, "diff", r2)
        if y:
            T["T2a"].append({"arm": "swap", "x": x["city"], "y": y["city"], "cont": x["continent"],
                             "layers": [cfg["ls"]], "donor_country": y["country"],
                             "cands": fill([y["country"], ref(x, cb), x["country"]], "T2a" + x["city"])})
    r3 = seeded("T2c")
    T["T2c"] = []
    for x in test:
        rx = ref(x, cb)
        opts = sorted(c for c in jvec if c not in (rx, x["country"]) and cont_of(c) == x["continent"])
        if opts:
            c = r3.choice(opts)
            T["T2c"].append({"arm": "add", "x": x["city"], "cont": x["continent"], "c": c, "scale": cfg["s_add"],
                             "cands": fill([c, rx, x["country"]], "T2c" + x["city"])})
    T["T3"] = [{"arm": "nat", "x": x["city"], "cont": x["continent"], "cands": x["cands"], "label": cb[x["city"]]}
               for x in test if cb[x["city"]]]
    T["T4"] = []
    for x in test:
        for kind, sc in (("kl", cfg["s_kl"]), ("norm", cfg["s_add"])):
            T["T4"].append({"arm": "add", "x": x["city"], "cont": x["continent"], "rand": x["city"], "scale": sc,
                            "kind": kind, "cands": x["cands"], "nat_cb": cb[x["city"]]})
    T["T5"] = [{"arm": "mc", "x": test[k % len(test)]["city"], "cont": test[k % len(test)]["continent"],
                "q": q, "opts": opts, "label": opts[0]} for k, (q, opts) in enumerate(MCQ)]
    for name in ("T1", "T2a", "T2c", "T4"):
        rr = readouts(m, T[name], cfg, jvec)
        for ex, r in zip(T[name], rr):
            ex["cb_int"] = r["cb"] if r["reliable"] else None
            ex["readout"] = {k: r[k] for k in ("cb_country", "cb_capital", "reliable", "margin", "lp_country",
                                               "lp_capital")}
        log(name, len(T[name]), "rows; reliable", sum(1 for e in T[name] if e["cb_int"]) / max(1, len(T[name])))
    for name in ("T1", "T2c", "T4"):
        for ex, k in zip(T[name], retention(m, T[name], cfg, jvec)):
            ex["kl"], ex["answer_kept"] = k["kl"], k["kept"]
    G["tests"] = T

    # 5. training lists (same size for every arm)
    N = cfg["n_train"]
    tr = [it for it in train if cb[it["city"]]]
    rO = seeded("trainO")
    O = []
    for k in range(N):
        it = tr[rO.randrange(len(tr))]
        O.append({"arm": "O", "x": it["city"], "cont": it["continent"], "label": cb[it["city"]], "cands": it["cands"]})
    # I: alternate different-country and same-country rows; recipients are drawn from those that have a
    # known train donor of the needed kind (same token count and continent), so the list always reaches N.
    rI = seeded("trainI")
    pools = {}
    for x in train:
        d = [y for y in idx_tr.get((x["n"], x["continent"]), []) if y["city"] != x["city"]]
        rx = ref(x, cb)
        pools[x["city"]] = ([y for y in d if y["country"] not in (rx, x["country"])], [y for y in d if y["country"] == rx])
    xs_kind = {0: [x for x in train if pools[x["city"]][0]], 1: [x for x in train if pools[x["city"]][1]]}
    strict = not a.selftest  # the self-test pools (pilot cities) may lack same-country donors
    assert xs_kind[0] and (xs_kind[1] or not strict), "no train recipients with a donor of each kind"
    I_ = []
    for t in range(N):
        k = t % 2 if xs_kind[1] else 0
        x = xs_kind[k][rI.randrange(len(xs_kind[k]))]
        y = rI.choice(pools[x["city"]][k])
        I_.append({"arm": "I", "x": x["city"], "y": y["city"], "cont": x["continent"], "label": y["country"],
                   "kind": ("diff", "same")[k], "cands": fill([y["country"], ref(x, cb), x["country"]], f"I{t}")})
    # J: alternate an injected country that differs from both the reference and the gazetteer country, and
    # the reference country itself.
    rJ = seeded("trainJ")

    def jdiff(x):
        return sorted(c for c in jvec if c not in (ref(x, cb), x["country"]) and cont_of(c) == x["continent"])
    jx = {0: [x for x in train if jdiff(x)], 1: [x for x in train if ref(x, cb) in jvec]}
    assert jx[0] and (jx[1] or not strict), "no train recipients for the J arm"
    J = []
    for t in range(N):
        k = t % 2 if jx[1] else 0
        x = jx[k][rJ.randrange(len(jx[k]))]
        rx = ref(x, cb)
        c = rJ.choice(jdiff(x)) if k == 0 else rx
        J.append({"arm": "J", "x": x["city"], "c": c, "cont": x["continent"], "label": c, "kind": ("diff", "same")[k],
                  "cands": fill([c, rx, x["country"]], f"J{t}")})
    assert len(O) == len(I_) == len(J) == N
    G["train"] = {"O": O, "X": [dict(e, arm="X") for e in O], "I": I_, "J": J}
    G["cb"] = cb
    G["minutes"] = (time.time() - t0) / 60
    save_json(G, os.path.join(out, "gen.json"))
    log("wrote gen.json;", {k: len(v) for k, v in G["train"].items()}, f"{G['minutes']:.1f} min")


if __name__ == "__main__":
    main()

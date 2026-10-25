"""Paper 9 v3 stages (PREDICTIONS_V3.md). python v3_run.py screen | pilot | gen | train ARM SEED | eval ARM [SEED]
Every stage skips work whose output exists. Outputs: ../res_v3/, ../res_pilot_v3/."""
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v3lib import (BINS, P9, ROOT, add_mc, add_report, all_items, bin_of, external_prompt, key, load_json, log,  # noqa: E402
                   measure, nat_row, paraphrase_lp2, parse, rot_scores, rotate, save_json, seeded, split_of_product,
                   unit_add, unit_natural, unit_swap, with_clean)
sys.path.insert(0, os.path.join(ROOT, "code"))
from mcq import MCQ  # noqa: E402

SMOKE = bool(os.environ.get("V3_SMOKE"))
RES = os.path.join(ROOT, "res_v3_smoke" if SMOKE else "res_v3")
SCREEN = os.path.join(ROOT, "res_v3", "screen.json")
PIL = os.path.join(ROOT, "res_pilot_v3")
BOUNDARIES = [0, 4, 8, 12, 16, 20]
SINGLES = [6, 10, 14, 18, 22]
ADD_LAYERS = [10, 14, 18]
ADD_SCALES = [1, 2, 4]


def rate(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def quartiles(v):
    v = sorted(v)
    return [v[len(v) // 4], v[len(v) // 2], v[(3 * len(v)) // 4]]


def ent_bin(e, th):  # higher entropy = less sure; th ascending quartiles of entropy
    return 0 if e <= th[0] else 1 if e <= th[1] else 2 if e <= th[2] else 3


# ------------------------------------------------------------------------------------------- screen
def screen(m):
    path = SCREEN
    if os.path.exists(path):
        return log("exists", path)
    sp = split_of_product()
    keys = [key(a, b) for a, b in all_items()]
    res = measure(m, [unit_natural(m, k) for k in keys])
    para = paraphrase_lp2(m, keys)
    items = {}
    for k, r in zip(keys, res):
        a, b = parse(k)
        items[k] = {"key": k, "product": a * b, "split": sp[a * b], "correct": r["correct"], "lp": r["lp"],
                    "lp2": r["lp"][1], "ent2": r["ent2"], "para_lp2": para[k]}
    save_json({"items": items, "accuracy": rate([i["correct"] for i in items.values()])}, path)
    log("screen", len(items), "items; accuracy", rate([i["correct"] for i in items.values()]))


def pools(scr, split):
    its = [i for i in scr.values() if i["split"] == split and i["correct"]]
    by_p = {}
    for i in its:
        by_p.setdefault(i["product"], []).append(i)
    return its, {p: sorted(v, key=lambda z: z["key"]) for p, v in by_p.items() if len(v) >= 2}


def diff_pairs(its, by_p, binf, rng, n, para_bin=None):
    """Recipient x with a same-product donor y whose bin differs from x's bin (and from x's paraphrase bin)."""
    out = []
    cand = [i for i in its if i["product"] in by_p]
    rng.shuffle(cand)
    for x in cand:
        ds = [y for y in by_p[x["product"]] if y["key"] != x["key"] and binf(y) != binf(x)
              and (para_bin is None or binf(y) != para_bin(x))]
        if ds:
            out.append((x, rng.choice(ds)))
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------------------------------------- pilot
def pilot(m):
    path = os.path.join(PIL, "pilot.json")
    if os.path.exists(path):
        return log("exists", path)
    os.makedirs(PIL, exist_ok=True)
    L = m.L
    scr = load_json(SCREEN)["items"]
    its, by_p = pools(scr, "pilot")
    th = quartiles([i["lp2"] for i in its])
    the = quartiles([i["ent2"] for i in its])
    binf = lambda i: bin_of(i["lp2"], th)  # noqa: E731
    R = {"n": len(its), "th": th, "accuracy_pilot": rate([i["correct"] for i in scr.values() if i["split"] == "pilot"])}
    R["P1"] = {"bin_agreement_prob_vs_entropy": rate([binf(i) == ent_bin(i["ent2"], the) for i in its]),
               "bin_counts": [sum(binf(i) == k for i in its) for k in range(4)]}
    rng = seeded("v3-pilot")
    pairs = diff_pairs(its, by_p, binf, rng, 120)
    R["n_pairs"] = len(pairs)
    R["P2"] = {}
    for kind, Ls in (("from", BOUNDARIES), ("single", SINGLES)):
        for B in Ls:
            layers = list(range(B, L)) if kind == "from" else [B]
            res = measure(m, [unit_swap(m, x["key"], y["key"], layers) for x, y in pairs])
            R["P2"][f"{kind}{B}"] = {"transfer_to_donor": rate([bin_of(r["lp"][1], th) == binf(y) for r, (x, y) in zip(res, pairs)]),
                                    "digits_kept": rate([r["correct"] for r in res])}
            log("P2", kind, B, R["P2"][f"{kind}{B}"])
    # bin vectors and the swap norm
    _, caps = m.run_units([unit_natural(m, i["key"]) for i in its], capture=tuple(ADD_LAYERS))
    capc = {l: {i["key"]: c[l][0].float() for i, c in zip(its, caps)} for l in ADD_LAYERS}
    bn = {l: rate([(capc[l][y["key"]] - capc[l][x["key"]]).norm(dim=-1).mean().item() for x, y in pairs]) for l in ADD_LAYERS}
    jv = {}
    for l in ADD_LAYERS:
        allm = torch.stack([capc[l][i["key"]].mean(0) for i in its]).mean(0)
        jv[l] = {}
        for k in range(4):
            vs = [capc[l][i["key"]].mean(0) for i in its if binf(i) == k]
            d = torch.stack(vs).mean(0) - allm
            jv[l][k] = d / d.norm()
    R["P3"] = {"base_norm": bn}
    best = None
    for la in ADD_LAYERS:
        for s in ADD_SCALES:
            exs = []
            for i in its[:120]:
                kk = (binf(i) + 2) % 4  # a different bin, fixed rule
                exs.append((i, kk))
            units = [with_clean(m, unit_add(m, i["key"], la, jv[la][kk] * (s * bn[la])), i["key"]) for i, kk in exs]
            res = measure(m, units)
            cell = {"change_to_injected": rate([bin_of(r["lp"][1], th) == kk for r, (i, kk) in zip(res, exs)]),
                    "retention": rate([r["correct"] for r in res]), "kl_median": statistics.median(r["kl"] for r in res)}
            R["P3"][f"L{la}_s{s}"] = cell
            log("P3", la, s, cell)
            if cell["retention"] >= 0.9 and (best is None or cell["change_to_injected"] > best[2] + 1e-9):
                best = (la, s, cell["change_to_injected"])
    l1 = max([B for B in BOUNDARIES if B <= 16 and R["P2"][f"from{B}"]["transfer_to_donor"] >= 0.9] or [0])
    ls = max(SINGLES, key=lambda S: (R["P2"][f"single{S}"]["transfer_to_donor"], -S))
    la, s_add = (best[0], best[1]) if best else (14, 1)
    # KL calibration of random directions against the training swap
    res = measure(m, [with_clean(m, unit_swap(m, x["key"], y["key"], list(range(l1, L))), x["key"]) for x, y in pairs])
    swap_kl = statistics.median(r["kl"] for r in res)
    R["P4"] = {"swap_kl_median": swap_kl, "random": {}}
    from items_v3 import random_unit  # noqa: E402
    for sc in (0.0625, 0.125, 0.25, 0.5, 1, 2):
        units = [with_clean(m, unit_add(m, x["key"], la, random_unit(m.d, x["key"]) * (sc * bn[la])), x["key"]) for x, _ in pairs]
        rr = measure(m, units)
        R["P4"]["random"][str(sc)] = {"kl_median": statistics.median(r["kl"] for r in rr), "retention": rate([r["correct"] for r in rr])}
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
    # learning-rate check: arm I on pilot half A, tested on half B (different-bin swaps)
    half = {p: ("A" if seeded("v3-half", p).random() < 0.5 else "B") for p in by_p}
    cfg = {"l1": l1, "la": la, "s_add": s_add, "base_norm": bn[la], "steps": 150, "bs": 8, "th": th}
    import trainlib_v3 as T  # noqa: E402
    trA = T.i_examples(its, {p: v for p, v in by_p.items() if half[p] == "A"}, binf, 400, seeded("v3-p5A"), para_bin=None)
    pB = diff_pairs([i for i in its if half.get(i["product"]) == "B"], {p: v for p, v in by_p.items() if half[p] == "B"},
                    binf, seeded("v3-p5B"), 80)
    teB = [{"arm": "I", "x": x["key"], "y": y["key"], "label": binf(y)} for x, y in pB]
    R["P5"] = {"n_train": len(trA), "n_test": len(teB)}
    for lr in (1e-4, 3e-4):
        cfg["lr"] = lr
        T.train(m, "I", trA, cfg, seed=0, jvec=None)
        rep = T.eval_reports(m, teB, cfg, None, lora=True)
        R["P5"][str(lr)] = {"acc_diff": rate([r["pick"] == e["label"] for r, e in zip(rep, teB)])}
        log("P5", lr, R["P5"][str(lr)])
        m.remove_lora()
        torch.cuda.empty_cache()
    nz = lambda v: 0.0 if v != v else v  # noqa: E731
    lr_best = max((1e-4, 3e-4), key=lambda v: (nz(R["P5"][str(v)]["acc_diff"]), -v))
    chosen = {"l1": l1, "ls": ls, "la": la, "s_add": s_add, "s_kl": s_kl, "lr": lr_best, "base_norm_pilot": bn[la]}
    R["chosen"] = chosen
    save_json(R, path)
    save_json(chosen, os.path.join(PIL, "config_pilot.json"))
    log("CHOSEN", chosen)


# ------------------------------------------------------------------------------------------- gen
def gen(m):
    path = os.path.join(RES, "gen.json")
    if os.path.exists(path):
        return log("exists", path)
    import trainlib_v3 as T  # noqa: E402
    from items_v3 import random_unit  # noqa: E402,F401
    if SMOKE:
        cfg = load_json(os.path.join(PIL, "config_pilot.json"))
        cfg.update({"n_train": 32, "steps": 4, "bs": 8, "seeds": [0]})
    else:
        cfg = load_json(os.path.join(ROOT, "config_v3.json"))
    os.makedirs(RES, exist_ok=True)
    L = m.L
    scr = load_json(SCREEN)["items"]
    tr_its, tr_by = pools(scr, "pilot" if SMOKE else "train")
    te_its, te_by = pools(scr, "pilot" if SMOKE else "test")
    if SMOKE:
        te_its = te_its[:40]
    th = quartiles([i["lp2"] for i in tr_its])  # bins from train items only
    the = quartiles([i["ent2"] for i in tr_its])
    cfg["th"] = th
    binf = lambda i: bin_of(i["lp2"], th)  # noqa: E731
    pbin = lambda i: bin_of(i["para_lp2"], th)  # noqa: E731
    G = {"cfg": cfg, "th": th, "th_ent": the, "n_train_items": len(tr_its), "n_test_items": len(te_its)}
    # bin vectors (train) and the swap norm at la
    la = cfg["la"]
    _, caps = m.run_units([unit_natural(m, i["key"]) for i in tr_its], capture=(la,))
    capc = {i["key"]: c[la][0].float() for i, c in zip(tr_its, caps)}
    allm = torch.stack([v.mean(0) for v in capc.values()]).mean(0)
    jvec = {}
    for k in range(4):
        vs = [capc[i["key"]].mean(0) for i in tr_its if binf(i) == k]
        d = torch.stack(vs).mean(0) - allm
        jvec[k] = d / d.norm()
    npairs = diff_pairs(tr_its, tr_by, binf, seeded("v3-norm"), 300)
    cfg["base_norm"] = rate([(capc[y["key"]] - capc[x["key"]]).norm(dim=-1).mean().item() for x, y in npairs])
    torch.save({"jvec": jvec, "base_norm": cfg["base_norm"], "la": la}, os.path.join(RES, "jvec.pt"))
    del caps, capc
    Tt = {}
    # T1: different-bin and same-bin donors (same product), test items
    r1 = seeded("v3-T1")
    Tt["T1"] = []
    for x in te_its:
        if x["product"] not in te_by:
            continue
        others = [y for y in te_by[x["product"]] if y["key"] != x["key"]]
        dd = [y for y in others if binf(y) not in (binf(x), pbin(x))]
        ss = [y for y in others if binf(y) == binf(x)]
        for kind, pool in (("diff", dd), ("same", ss)):
            if pool:
                y = r1.choice(pool)
                Tt["T1"].append({"arm": "swap", "x": x["key"], "y": y["key"], "kind": kind, "layers": list(range(cfg["l1"], L)),
                                 "label": binf(y)})
    r2 = seeded("v3-T2a")
    Tt["T2a"] = []
    for x in te_its:
        if x["product"] in te_by:
            dd = [y for y in te_by[x["product"]] if y["key"] != x["key"] and binf(y) != binf(x)]
            if dd:
                Tt["T2a"].append({"arm": "swap", "x": x["key"], "y": r2.choice(dd)["key"], "layers": [cfg["ls"]]})
    r3 = seeded("v3-T2c")
    Tt["T2c"] = []
    for x in te_its[:400]:
        opts = [k for k in range(4) if k not in (binf(x), pbin(x))]
        Tt["T2c"].append({"arm": "add", "x": x["key"], "c": r3.choice(opts), "scale": cfg["s_add"]})
    Tt["T3"] = [{"arm": "nat", "x": x["key"], "label": binf(x)} for x in te_its]
    Tt["T4"] = [{"arm": "add", "x": x["key"], "rand": x["key"], "scale": sc, "kind": kd, "nat": binf(x)}
                for x in te_its[:400] for kd, sc in (("kl", cfg["s_kl"]), ("norm", cfg["s_add"]))]
    Tt["T5"] = [{"arm": "mc", "x": te_its[k % len(te_its)]["key"], "q": q, "opts": o, "label": o[0]} for k, (q, o) in enumerate(MCQ)]
    for name in ("T1", "T2a", "T2c", "T4"):
        units = [with_clean(m, T.example_unit(m, e, cfg, jvec, rotations=()), e["x"]) for e in Tt[name]]
        res = measure(m, units)
        for e, r in zip(Tt[name], res):
            e["cb_int"] = bin_of(r["lp"][1], th)
            e["lp2_int"] = r["lp"][1]
            e["answer_kept"] = r["correct"]
            e["kl"] = r["kl"]
        log(name, len(Tt[name]), "rows")
    G["tests"] = Tt
    # training lists (train items), the same size for every arm
    N = cfg["n_train"]
    rO = seeded("v3-trainO")
    O = [{"arm": "O", "x": i["key"], "label": binf(i)} for i in (tr_its[rO.randrange(len(tr_its))] for _ in range(N))]
    I_ = T.i_examples(tr_its, tr_by, binf, N, seeded("v3-trainI"), para_bin=pbin)
    rJ = seeded("v3-trainJ")
    J = []
    for t in range(N):
        x = tr_its[rJ.randrange(len(tr_its))]
        if t % 2 == 0:
            c = rJ.choice([k for k in range(4) if k not in (binf(x), pbin(x))])
        else:
            c = binf(x)
        J.append({"arm": "J", "x": x["key"], "c": c, "label": c, "kind": ("diff", "same")[t % 2]})
    assert len(O) == len(I_) == len(J) == N
    G["train"] = {"O": O, "X": [dict(e, arm="X") for e in O], "I": I_, "J": J}
    G["items"] = {k: {"bin": binf(v), "para_bin": pbin(v), "ent_bin": ent_bin(v["ent2"], the)} for k, v in scr.items()
                  if v["correct"] and (v["split"] in ("train", "test") or SMOKE)}
    save_json(G, path)
    log("gen done", {k: len(v) for k, v in G["train"].items()}, {k: len(v) for k, v in Tt.items()})


# ------------------------------------------------------------------------------------------- train / eval
def setup():
    G = load_json(os.path.join(RES, "gen.json"))
    cfg = dict(G["cfg"])
    jv = torch.load(os.path.join(RES, "jvec.pt"))
    cfg["base_norm"] = jv["base_norm"]
    return G, cfg, jv["jvec"]


def train(m, arm, seed):
    import trainlib_v3 as T  # noqa: E402
    path = os.path.join(RES, f"lora_{arm}_s{seed}.pt")
    if os.path.exists(path):
        return log("exists", path)
    G, cfg, jvec = setup()
    t0 = time.time()
    state, hist = T.train(m, arm, G["train"][arm], cfg, seed, jvec=jvec)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    save_json({"arm": arm, "seed": seed, "hist": hist, "minutes": (time.time() - t0) / 60},
              os.path.join(RES, f"trainlog_{arm}_s{seed}.json"))
    m.remove_lora()


def evaluate(m, arm, seed):
    import trainlib_v3 as T  # noqa: E402
    tag = "N" if arm == "N" else f"{arm}_s{seed}"
    path = os.path.join(RES, f"eval_{tag}.json")
    if os.path.exists(path):
        return log("exists", path)
    G, cfg, jvec = setup()
    lora = arm != "N"
    if lora:
        m.load_lora(torch.load(os.path.join(RES, f"lora_{arm}_s{seed}.pt")))
    Tt = G["tests"]
    E = {"arm": arm, "seed": seed}
    if arm == "X":
        E["T3"] = T.eval_external(m, Tt["T3"], lora=True)
    else:
        for name in ("T1", "T2a", "T2c", "T3", "T4"):
            E[name] = T.eval_reports(m, Tt[name], cfg, jvec, lora=lora)
        units = [add_mc(m, unit_natural(m, e["x"]), e["q"], e["opts"]) for e in Tt["T5"]]
        res = m.run_units(units, lora=lora)
        E["T5"] = []
        for e, r in zip(Tt["T5"], res):
            s = rot_scores([r[k][0] for k in range(4)])
            E["T5"].append({"pick": e["opts"][max(range(4), key=lambda i: s[i])]})
        # T5u: the adapter on every position; multiplication accuracy on test items
        m.lora_on(False)
        keys = [e["x"] for e in Tt["T3"]][:400]
        res = measure_lora(m, keys, lora)
        E["T5u"] = [{"key": k, "correct": r["correct"]} for k, r in zip(keys, res)]
    save_json(E, path)
    m.remove_lora()


def measure_lora(m, keys, lora):
    """measure() with the adapter applied to turn 1 as well (ungated)."""
    old = m._turn1_pass

    def patched(rows, capture_layers=None, lora_turn1=False):
        return old(rows, capture_layers=capture_layers, lora_turn1=lora)
    m._turn1_pass = patched
    try:
        return measure(m, [unit_natural(m, k) for k in keys])
    finally:
        m._turn1_pass = old


def main():
    st = sys.argv[1]
    os.makedirs(RES, exist_ok=True)
    m = P9()
    if st == "screen":
        screen(m)
    elif st == "pilot":
        pilot(m)
    elif st == "gen":
        gen(m)
    elif st == "train":
        train(m, sys.argv[2], int(sys.argv[3]))
    elif st == "eval":
        evaluate(m, sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        raise SystemExit("unknown stage " + st)


if __name__ == "__main__":
    main()

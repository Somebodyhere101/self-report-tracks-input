"""Paper 9 v4 stages (PREDICTIONS_V4.md). python v4_run.py ladder | gen | train ARM SEED | eval ARM [SEED]
Every stage skips work whose output exists, and training resumes from its checkpoint. Exit code 3 = the GPU-lock
deadline is near (state saved; re-queue the job). Outputs: ../res_pilot_v4/ (ladder), ../res_v4/ (main).
Turn 1 ends with the four answer digits (R = the digits). Arms N, O, I, J report in the IMMEDIATE format (the primary);
arm Id is arm I trained and tested with the DELAYED (turn-2) report, and Nd the untrained delayed report (secondary).
V4_SMOKE=1: code test with tiny counts on pilot items only (outputs *_smoke)."""
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v4lib import (ROOT, BINS, P9v4, add_report, key, load_json, log, measure, pick2, save_json, seeded,  # noqa: E402,F401
                   unit_add4, unit_natural, unit_natural4, unit_swap4, with_clean4)
from items_v3 import random_unit  # noqa: E402  (v3, frozen)
from v4stats import bin2, ent_bin2, medians, predictors  # noqa: E402
import trainlib_v4 as T  # noqa: E402
from v3lib import add_mc  # noqa: E402  (v3, frozen: four-option questions in four orders)
from p9lib import rot_scores  # noqa: E402  (v1, frozen)
from mcq import MCQ  # noqa: E402  (v1, frozen)

SMOKE = bool(os.environ.get("V4_SMOKE"))
SUF = "_smoke" if SMOKE else ""
RES = os.path.join(ROOT, "res_v4" + SUF)
PIL = os.path.join(ROOT, "res_pilot_v4" + SUF)
SCREEN = os.path.join(ROOT, "res_v3", "screen.json")
CKPT = os.environ.get("V4_CKPT") or os.path.join(os.environ.get("TEMP", ROOT), "p9", "v4ckpt" + SUF)
DEADLINE = float(os.environ["GPU_LOCK_DEADLINE"]) if os.environ.get("GPU_LOCK_DEADLINE") else None
EXIT_YIELD = 3
BS = 8
L1 = 16  # v3's frozen training swap: answer positions from layer 16 on
FMT = "imm"  # the primary report format; arm Id (and Nd) use "del"
GATE_I = 2 / 3
# budget ladder: each (rank, lr) setting is trained once on pilot half A and tested at every stop it reaches
CONFIGS = [("r8_lr1e-4", 8, 1e-4), ("r8_lr1e-3", 8, 1e-3), ("r32_lr3e-4", 32, 3e-4), ("r32_lr1e-3", 32, 1e-3)]
STOPS = [300, 1200, 3000]
LONG = ("r32_lr3e-4", "r32_lr1e-3")  # the settings that continue past the first stop
if SMOKE:
    CONFIGS, STOPS = [CONFIGS[0], CONFIGS[3]], [2, 4]


def rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def rungs():
    out = []
    for S in STOPS:
        for name, r, lr in CONFIGS:
            if S == STOPS[0] or name in LONG:
                out.append({"name": f"{name}@{S}", "config": name, "rank": r, "lr": lr, "steps": S})
    return out


def items_of(scr, split):
    return sorted([i for i in scr.values() if i["split"] == split and i["correct"]], key=lambda i: i["key"])


def by_product(its, multi=True):
    by = {}
    for i in its:
        by.setdefault(i["product"], []).append(i)
    return {p: v for p, v in by.items() if len(v) >= 2 or not multi}


# ------------------------------------------------------------------------------------------- ladder
def eval_pilot(m, rowsI, rowsO, lora, tag):
    out = {}
    cfg = {"l1": L1, "fmt": FMT}
    rep = T.eval_reports(m, rowsI, cfg, None, lora=lora, tie_key=(tag, "I"))
    for kind in ("diff", "same"):
        idx = [k for k, e in enumerate(rowsI) if e["kind"] == kind]
        out[f"acc_{kind}"] = rate(rep[k]["pick"] == rowsI[k]["label"] for k in idx)
        out[f"n_{kind}"] = len(idx)
    out["diff_names_recipient_bin"] = rate(rep[k]["pick"] == e["nat"] for k, e in enumerate(rowsI) if e["kind"] == "diff")
    rep2 = T.eval_reports(m, rowsO, cfg, None, lora=lora, tie_key=(tag, "O"))
    out["acc_nat"] = rate(r["pick"] == e["label"] for r, e in zip(rep2, rowsO))
    out["n_nat"] = len(rowsO)
    out["frac_pick0_nat"] = rate(r["pick"] == 0 for r in rep2)
    out["picks_swap"] = [r["pick"] for r in rep]
    out["picks_nat"] = [r["pick"] for r in rep2]
    return out


def calibrate_s_kl(m, A, byA, b):
    """v3's pilot rule P4, for the four-digit R, on half-A different-bin pairs: the scale (in units of the swap norm at
    la) at which random directions' median R-KL matches the training swap's (log-log interpolation)."""
    la = load_json(os.path.join(ROOT, "config_v3.json"))["la"]
    rn = seeded("v4-skl")
    cand = [x for x in A if x["product"] in byA]
    rn.shuffle(cand)
    pairs = []
    for x in cand:
        ds = [y for y in byA[x["product"]] if y["key"] != x["key"] and b(y) != b(x)]
        if ds:
            pairs.append((x, rn.choice(ds)))
        if len(pairs) >= (12 if SMOKE else 120):
            break
    keys = sorted({k for x, y in pairs for k in (x["key"], y["key"])})
    _, caps = m.run_units([unit_natural4(m, k) for k in keys], capture=(la,))
    cap = {k: c[la][0].float() for k, c in zip(keys, caps)}
    bn = rate((cap[y["key"]] - cap[x["key"]]).norm(dim=-1).mean().item() for x, y in pairs)
    res = measure(m, [with_clean4(m, unit_swap4(m, x["key"], y["key"], list(range(L1, m.L))), x["key"]) for x, y in pairs])
    swap_kl = statistics.median(r["kl"] for r in res)
    rnd = {}
    for sc in (0.0625, 0.125, 0.25, 0.5, 1, 2):
        rr = measure(m, [with_clean4(m, unit_add4(m, x["key"], la, random_unit(m.d, x["key"]) * (sc * bn)), x["key"]) for x, _ in pairs])
        rnd[str(sc)] = {"kl_median": statistics.median(r["kl"] for r in rr), "retention": rate(r["correct"] for r in rr)}
    pts = sorted((float(k), v["kl_median"]) for k, v in rnd.items())
    s_kl = pts[0][0]
    for (s0, k0), (s1, k1) in zip(pts, pts[1:]):
        if k0 <= swap_kl <= k1 and k1 > k0 > 0:
            f = (math.log(swap_kl) - math.log(k0)) / (math.log(k1) - math.log(k0))
            s_kl = math.exp(math.log(s0) + f * (math.log(s1) - math.log(s0)))
            break
    else:
        if swap_kl > pts[-1][1]:
            s_kl = pts[-1][0]
    out = {"n_pairs": len(pairs), "la": la, "base_norm_A": bn, "swap_kl_median": swap_kl, "random": rnd, "s_kl": s_kl}
    log("s_kl calibration", out)
    return out


def ladder(m):
    os.makedirs(PIL, exist_ok=True)
    path = os.path.join(PIL, "ladder.json")
    Ld = load_json(path) if os.path.exists(path) else None
    if Ld and Ld.get("done"):
        return log("exists", path)
    scr = load_json(SCREEN)["items"]
    M = medians(scr)
    b = lambda i: bin2(i["lp2"], M["med"])  # noqa: E731
    pb = lambda i: bin2(i["para_lp2"], M["med"])  # noqa: E731
    pil = items_of(scr, "pilot")
    half = {p: ("A" if seeded("v4-half", p).random() < 0.5 else "B") for p in sorted({i["product"] for i in pil})}
    A = [i for i in pil if half[i["product"]] == "A"]
    B = [i for i in pil if half[i["product"]] == "B"]
    if SMOKE:
        B = B[:40]
    byA, byB = by_product(A), by_product(B)
    if Ld is None:
        P, info = predictors(A, B, M)
        tB = [b(i) for i in B]
        accB = {k: rate(int(v[j]) == tB[j] for j in range(len(B))) for k, v in P.items()}
        rB = seeded("v4-ladder-B")
        rowsI = []
        for x in B:
            if x["product"] not in byB:
                continue
            others = [y for y in byB[x["product"]] if y["key"] != x["key"]]
            dd = [y for y in others if b(y) not in (b(x), pb(x))]
            ss = [y for y in others if b(y) == b(x)]
            for kind, pool in (("diff", dd), ("same", ss)):
                if pool:
                    y = rB.choice(pool)
                    rowsI.append({"arm": "swap", "x": x["key"], "y": y["key"], "kind": kind, "label": b(y), "nat": b(x),
                                  "layers": list(range(L1, m.L))})
        rowsO = [{"arm": "nat", "x": i["key"], "label": b(i)} for i in B]
        res = measure(m, [T.example_unit(m, e, {"l1": L1, "fmt": FMT}, None, rotations=()) for e in rowsI])
        for e, r in zip(rowsI, res):
            e["cb_int"] = bin2(r["lp"][1], M["med"])
            e["answer_kept"] = r["correct"]
        dif = [e for e in rowsI if e["kind"] == "diff"]
        Ld = {"medians": M, "n_A": len(A), "n_B": len(B), "Pstar_pilot_acc": accB, "Pstar_pilot": max(accB.values()),
              "Pstar_info": info, "rowsI": rowsI, "rowsO": rowsO,
              "bed": {"PG2_prob_vs_entropy_B": rate(b(i) == ent_bin2(i["ent2"], M["med_ent"]) for i in B),
                      "PG3_swap_carries_donor_bin_diff": rate(e["cb_int"] == e["label"] for e in dif),
                      "PG3_digits_kept": rate(e["answer_kept"] for e in rowsI)},
              "rungs": {}, "order": [R["name"] for R in rungs()], "chosen": None, "done": False}
        Ld["N"] = eval_pilot(m, rowsI, rowsO, lora=False, tag="ladder-N")
        Ld["s_kl"] = calibrate_s_kl(m, A, byA, b)
        save_json(Ld, path)
        log("ladder setup", {k: Ld[k] for k in ("n_A", "n_B", "Pstar_pilot", "bed")}, "N", {k: v for k, v in Ld["N"].items() if "picks" not in k})
    rowsI, rowsO = Ld["rowsI"], Ld["rowsO"]
    if not (Ld["bed"]["PG3_swap_carries_donor_bin_diff"] >= 0.90 and Ld["bed"]["PG2_prob_vs_entropy_B"] >= 0.70):
        Ld["done"] = True
        Ld["stopped"] = "two-bin bed check failed"
        save_json(Ld, path)
        return log("BED CHECK FAILED", Ld["bed"])
    nmax = STOPS[-1] * BS
    exI = T.i_examples(A, byA, b, nmax, seeded("v4-ladder-I"), para_bin=pb)
    rO = seeded("v4-ladder-O")
    exO = [{"arm": "O", "x": i["key"], "label": b(i)} for i in (A[rO.randrange(len(A))] for _ in range(nmax))]
    for R in rungs():
        res = Ld["rungs"].setdefault(R["name"], {"rung": R})
        if res.get("complete"):
            if res["pass"]:
                Ld["chosen"] = res["rung"]
                break
            continue
        cfg = {"bs": BS, "lr": R["lr"], "rank": R["rank"], "l1": L1, "fmt": FMT}
        for arm, ex in (("I", exI), ("O", exO)):
            if arm in res:
                continue
            ck = os.path.join(CKPT, f"ladder_{R['config']}_{arm}.pt")
            t0 = time.time()
            status, hist = T.train(m, arm, ex, cfg, 0, None, R["steps"], ck, deadline=DEADLINE)
            if status == "yield":
                save_json(Ld, path)
                log("deadline near: yielding during", R["name"], arm)
                sys.exit(EXIT_YIELD)
            ev = eval_pilot(m, rowsI, rowsO, lora=True, tag=f"ladder-{R['name']}-{arm}")
            ev["train_minutes_this_call"] = (time.time() - t0) / 60
            ev["loss_hist"] = hist
            res[arm] = ev
            m.remove_lora()
            torch.cuda.empty_cache()
            save_json(Ld, path)
            log(R["name"], arm, {k: v for k, v in ev.items() if k not in ("picks_swap", "picks_nat", "loss_hist")})
        I_, O_ = res["I"], res["O"]
        res["gate_I"] = I_["acc_diff"] >= GATE_I and I_["acc_same"] >= GATE_I
        res["gate_O"] = O_["acc_nat"] >= Ld["Pstar_pilot"]
        res["O_learned"] = O_["acc_nat"] >= GATE_I  # the exploratory fallback's O condition (kappa >= 1/3)
        res["pass"] = bool(res["gate_I"] and res["gate_O"])
        res["complete"] = True
        if res["pass"]:
            Ld["chosen"] = R
        save_json(Ld, path)
        log("RUNG", R["name"], "gate_I", res["gate_I"], "gate_O", res["gate_O"], "O_learned", res["O_learned"])
        if res["pass"]:
            break
    if Ld["chosen"] is None:  # exploratory fallback: the cheapest rung where I passes and O learned (never the verdict)
        for name in Ld["order"]:
            r = Ld["rungs"].get(name)
            if r and r.get("complete") and r["gate_I"] and r["O_learned"]:
                Ld["fallback"] = r["rung"]
                break
    Ld["done"] = True
    save_json(Ld, path)
    log("ladder done; chosen", Ld["chosen"], "fallback", Ld.get("fallback"))


# ------------------------------------------------------------------------------------------- gen
def gen(m):
    path = os.path.join(RES, "gen.json")
    if os.path.exists(path):
        return log("exists", path)
    cfg = load_json(os.path.join(ROOT, "config_v4" + SUF + ".json"))
    os.makedirs(RES, exist_ok=True)
    L = m.L
    scr = load_json(SCREEN)["items"]
    M = medians(scr)
    b = lambda i: bin2(i["lp2"], M["med"])  # noqa: E731
    pb = lambda i: bin2(i["para_lp2"], M["med"])  # noqa: E731
    tr_its = items_of(scr, "pilot" if SMOKE else "train")
    te_its = items_of(scr, "pilot" if SMOKE else "test")
    if SMOKE:
        te_its = te_its[:24]
    tr_by, te_by = by_product(tr_its), by_product(te_its)
    G = {"cfg": cfg, "medians": M, "n_train_items": len(tr_its), "n_test_items": len(te_its)}
    # bin vectors (train items) and the swap norm at la
    la = cfg["la"]
    _, caps = m.run_units([unit_natural4(m, i["key"]) for i in tr_its], capture=(la,))
    capc = {i["key"]: c[la][0].float() for i, c in zip(tr_its, caps)}
    allm = torch.stack([v.mean(0) for v in capc.values()]).mean(0)
    jvec = {}
    for k in range(2):
        d = torch.stack([capc[i["key"]].mean(0) for i in tr_its if b(i) == k]).mean(0) - allm
        jvec[k] = d / d.norm()
    rn = seeded("v4-norm")
    cand = [x for x in tr_its if x["product"] in tr_by]
    rn.shuffle(cand)
    npairs = []
    for x in cand:
        ds = [y for y in tr_by[x["product"]] if y["key"] != x["key"] and b(y) != b(x)]
        if ds:
            npairs.append((x, rn.choice(ds)))
        if len(npairs) >= 300:
            break
    cfg["base_norm"] = rate((capc[y["key"]] - capc[x["key"]]).norm(dim=-1).mean().item() for x, y in npairs)
    torch.save({"jvec": jvec, "base_norm": cfg["base_norm"], "la": la}, os.path.join(RES, "jvec.pt"))
    del caps, capc
    Tt = {}
    r1 = seeded("v4-T1")
    Tt["T1"] = []
    for x in te_its:
        if x["product"] not in te_by:
            continue
        others = [y for y in te_by[x["product"]] if y["key"] != x["key"]]
        dd = [y for y in others if b(y) not in (b(x), pb(x))]
        ss = [y for y in others if b(y) == b(x)]
        for kind, pool in (("diff", dd), ("same", ss)):
            if pool:
                y = r1.choice(pool)
                Tt["T1"].append({"arm": "swap", "x": x["key"], "y": y["key"], "kind": kind, "layers": list(range(cfg["l1"], L)),
                                 "label": b(y)})
    r2 = seeded("v4-T2a")
    Tt["T2a"] = []
    for x in te_its:
        if x["product"] in te_by:
            dd = [y for y in te_by[x["product"]] if y["key"] != x["key"] and b(y) != b(x)]
            if dd:
                Tt["T2a"].append({"arm": "swap", "x": x["key"], "y": r2.choice(dd)["key"], "layers": [cfg["ls"]]})
    n_small = 12 if SMOKE else 400
    Tt["T2c"] = [{"arm": "add", "x": x["key"], "c": 1 - b(x), "scale": cfg["s_add"]} for x in te_its[:n_small] if b(x) == pb(x)]
    Tt["T3"] = [{"arm": "nat", "x": x["key"], "label": b(x)} for x in te_its]
    Tt["T4"] = [{"arm": "add", "x": x["key"], "rand": x["key"], "scale": sc, "kind": kd, "nat": b(x)}
                for x in te_its[:(8 if SMOKE else 400)] for kd, sc in (("kl", cfg["s_kl"]), ("norm", cfg["s_add"]))]
    Tt["T5"] = [{"arm": "mc", "x": te_its[k % len(te_its)]["key"], "q": q, "opts": o, "label": o[0]}
                for k, (q, o) in enumerate(MCQ[:(8 if SMOKE else len(MCQ))])]
    for name in ("T1", "T2a", "T2c", "T4"):
        units = [with_clean4(m, T.example_unit(m, e, cfg, jvec, rotations=()), e["x"]) for e in Tt[name]]
        res = measure(m, units)
        for e, r in zip(Tt[name], res):
            e["cb_int"] = bin2(r["lp"][1], M["med"])
            e["lp2_int"] = r["lp"][1]
            e["answer_kept"] = r["correct"]
            e["kl"] = r["kl"]
        log(name, len(Tt[name]), "rows")
    G["tests"] = Tt
    # training streams (train items), the same length for every arm
    N = cfg["n_train"]
    rO = seeded("v4-trainO")
    O = [{"arm": "O", "x": i["key"], "label": b(i)} for i in (tr_its[rO.randrange(len(tr_its))] for _ in range(N))]
    I_ = T.i_examples(tr_its, tr_by, b, N, seeded("v4-trainI"), para_bin=pb)
    rJ = seeded("v4-trainJ")
    elig = [x for x in tr_its if b(x) == pb(x)]
    J = []
    for t in range(N):
        if t % 2 == 0:
            x = elig[rJ.randrange(len(elig))]
            J.append({"arm": "J", "x": x["key"], "c": 1 - b(x), "label": 1 - b(x), "kind": "diff"})
        else:
            x = tr_its[rJ.randrange(len(tr_its))]
            J.append({"arm": "J", "x": x["key"], "c": b(x), "label": b(x), "kind": "same"})
    assert len(O) == len(I_) == len(J) == N
    G["train"] = {"O": O, "X": [dict(e, arm="X") for e in O], "I": I_, "J": J}  # arm Id trains on I's stream
    save_json(G, path)
    log("gen done", {k: len(v) for k, v in G["train"].items()}, {k: len(v) for k, v in Tt.items()})


# ------------------------------------------------------------------------------------------- train / eval
def setup(arm="O"):
    G = load_json(os.path.join(RES, "gen.json"))
    cfg = dict(G["cfg"])
    cfg["fmt"] = "del" if arm in ("Id", "Nd") else cfg["fmt"]
    jv = torch.load(os.path.join(RES, "jvec.pt"))
    cfg["base_norm"] = jv["base_norm"]
    return G, cfg, jv["jvec"]


def train(m, arm, seed):
    path = os.path.join(RES, f"lora_{arm}_s{seed}.pt")
    if os.path.exists(path):
        return log("exists", path)
    G, cfg, jvec = setup(arm)
    t0 = time.time()
    ck = os.path.join(CKPT, f"main_{cfg['rung']}_{arm}_s{seed}.pt")
    skey = "I" if arm == "Id" else arm  # Id trains on I's examples, in I's order and option orders (paired contrast)
    stream = G["train"][skey]
    order = list(range(len(stream)))
    seeded("v4-order", skey, seed).shuffle(order)  # each seed its own example order, as in v3
    stream = [stream[i] for i in order]
    status, hist = T.train(m, arm, stream, cfg, seed, jvec, cfg["steps"], ck, deadline=DEADLINE, stream_key=skey)
    if status == "yield":
        log("deadline near: yielding during train", arm, seed)
        sys.exit(EXIT_YIELD)
    torch.save(m.lora_state(), path + ".tmp")
    os.replace(path + ".tmp", path)
    save_json({"arm": arm, "seed": seed, "hist": hist, "minutes_last_call": (time.time() - t0) / 60, "rank": cfg["rank"],
               "lr": cfg["lr"], "steps": cfg["steps"], "fmt": cfg["fmt"]}, os.path.join(RES, f"trainlog_{arm}_s{seed}.json"))
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


def evaluate(m, arm, seed):
    tag = arm if arm in ("N", "Nd") else f"{arm}_s{seed}"
    path = os.path.join(RES, f"eval_{tag}.json")
    if os.path.exists(path):
        return log("exists", path)
    G, cfg, jvec = setup(arm)
    lora = arm not in ("N", "Nd")
    if lora:
        m.load_lora(torch.load(os.path.join(RES, f"lora_{arm}_s{seed}.pt")))
    Tt = G["tests"]
    E = {"arm": arm, "seed": seed, "fmt": cfg["fmt"]}
    if arm == "X":
        E["T3"] = T.eval_external(m, Tt["T3"], lora=True, tie_key=("main", tag, "T3"))
    else:
        for name in ("T1", "T2a", "T2c", "T3", "T4"):
            E[name] = T.eval_reports(m, Tt[name], cfg, jvec, lora=lora, tie_key=("main", tag, name))
        units = [add_mc(m, unit_natural(m, e["x"]), e["q"], e["opts"]) for e in Tt["T5"]]
        res = m.run_units(units, lora=lora)
        E["T5"] = []
        for e, r in zip(Tt["T5"], res):
            s = rot_scores([r[k][0] for k in range(4)])
            E["T5"].append({"pick": e["opts"][max(range(4), key=lambda i: s[i])]})
        m.lora_on(False)
        keys = [e["x"] for e in Tt["T3"]][:(8 if SMOKE else 400)]
        res = measure_lora(m, keys, lora)
        E["T5u"] = [{"key": k, "correct": r["correct"]} for k, r in zip(keys, res)]
    save_json(E, path)
    m.remove_lora()


def main():
    torch.set_num_threads(4)
    st = sys.argv[1]
    m = P9v4()
    if st == "ladder":
        ladder(m)
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

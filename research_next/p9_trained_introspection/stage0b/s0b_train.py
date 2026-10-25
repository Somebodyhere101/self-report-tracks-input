"""Stage 0b-train (paper 9 v4; exploratory, pilot products only, before the v4 freeze): can a LoRA reporter learn the
two-bin confidence at all, in the immediate or the delayed report format?
Data: pilot HALF A only (v4's seeded halves; half B stays untouched for v4's frozen learnability gate), split again by
product into A1 (about 60%, training) and A2 (held out). Arm I: v4's examples (same-product swaps of the four digit
positions from layer 16, alternating different-bin and same-bin donors). Arm O: natural labels. Scored on A2:
  I's question: different-bin swap rows (label = donor's bin; following the input scores 0) and same-bin swap rows;
  O's question: natural rows, against the input-only predictors (P*_A2; majority and learned fitted on A1).
Training: trainlib_v4.train (v4's loop: Adam, warm-up, clip 1.0, batch 8, seeded option orders, checkpoints; LoRA in
the report tail only). Writes stage0b/s0b_train.json after every evaluation; exit 3 if the GPU-lock deadline is near."""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import torch

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
P9DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P9DIR, "code_v4"))
from v4lib import P9v4, load_json, log, measure, seeded  # noqa: E402
from v4stats import bin2, medians, predictors  # noqa: E402
import trainlib_v4 as T  # noqa: E402

OUT = os.path.join(HERE, "s0b_train.json")
CKPT = os.path.join(os.environ.get("TEMP", HERE), "p9", "s0b_ckpt")
DEADLINE = float(os.environ["GPU_LOCK_DEADLINE"]) if os.environ.get("GPU_LOCK_DEADLINE") else None
L1 = 16
# (name, format, arm, rank, lr, evaluation stops), most informative first
RUNS = [("imm_I_r8_1e-3", "imm", "I", 8, 1e-3, [150, 300]), ("imm_I_r8_1e-4", "imm", "I", 8, 1e-4, [150, 300]),
        ("del_I_r8_1e-3", "del", "I", 8, 1e-3, [150, 300]), ("del_I_r8_1e-4", "del", "I", 8, 1e-4, [150, 300]),
        ("imm_O_r8_1e-3", "imm", "O", 8, 1e-3, [150, 300]),
        ("imm_I_r32_3e-4", "imm", "I", 32, 3e-4, [300, 600, 900]), ("imm_O_r32_3e-4", "imm", "O", 32, 3e-4, [300, 600, 900])]


def rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def save(R):
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1)
    os.replace(tmp, OUT)


def score(m, rowsI, rowsO, fmt, lora, tag):
    cfg = {"l1": L1, "fmt": fmt}
    rI = T.eval_reports(m, rowsI, cfg, None, lora=lora, tie_key=("s0b", tag, "I"))
    rO = T.eval_reports(m, rowsO, cfg, None, lora=lora, tie_key=("s0b", tag, "O"))
    d = [r["pick"] == e["label"] for r, e in zip(rI, rowsI) if e["kind"] == "diff"]
    s = [r["pick"] == e["label"] for r, e in zip(rI, rowsI) if e["kind"] == "same"]
    return {"acc_diff": rate(d), "acc_same": rate(s), "acc_swap_balanced": (rate(d) + rate(s)) / 2,
            "diff_names_recipient_bin": rate(r["pick"] == e["nat"] for r, e in zip(rI, rowsI) if e["kind"] == "diff"),
            "acc_nat": rate(r["pick"] == e["label"] for r, e in zip(rO, rowsO)),
            "frac_pick0_nat": rate(r["pick"] == 0 for r in rO), "frac_pick0_swap": rate(r["pick"] == 0 for r in rI),
            "n_diff": len(d), "n_same": len(s), "n_nat": len(rO)}


def main():
    t00 = time.time()
    scr = load_json(os.path.join(P9DIR, "res_v3", "screen.json"))["items"]
    M = medians(scr)
    b = lambda i: bin2(i["lp2"], M["med"])  # noqa: E731
    pb = lambda i: bin2(i["para_lp2"], M["med"])  # noqa: E731
    pil = sorted([i for i in scr.values() if i["split"] == "pilot" and i["correct"]], key=lambda i: i["key"])
    half = {p: ("A" if seeded("v4-half", p).random() < 0.5 else "B") for p in sorted({i["product"] for i in pil})}
    A = [i for i in pil if half[i["product"]] == "A"]
    sub = {p: ("A1" if seeded("s0b-split", p).random() < 0.6 else "A2") for p in sorted({i["product"] for i in A})}
    A1 = [i for i in A if sub[i["product"]] == "A1"]
    A2 = [i for i in A if sub[i["product"]] == "A2"]

    def by(its):
        d = {}
        for i in its:
            d.setdefault(i["product"], []).append(i)
        return {p: v for p, v in d.items() if len(v) >= 2}
    by1, by2 = by(A1), by(A2)
    m = P9v4()
    R = load_json(OUT) if os.path.exists(OUT) else None
    if R is None:
        rng = seeded("s0b-A2rows")
        rowsI = []
        for x in A2:
            if x["product"] not in by2:
                continue
            others = [y for y in by2[x["product"]] if y["key"] != x["key"]]
            for kind, pool in (("diff", [y for y in others if b(y) not in (b(x), pb(x))]), ("same", [y for y in others if b(y) == b(x)])):
                if pool:
                    y = rng.choice(pool)
                    rowsI.append({"arm": "swap", "x": x["key"], "y": y["key"], "kind": kind, "label": b(y), "nat": b(x),
                                  "layers": list(range(L1, m.L))})
        rowsO = [{"arm": "nat", "x": i["key"], "label": b(i)} for i in A2]
        res = measure(m, [T.example_unit(m, e, {"l1": L1, "fmt": "imm"}, None, rotations=()) for e in rowsI])
        P, info = predictors(A1, A2, M)
        tA2 = [b(i) for i in A2]
        R = {"n_A1": len(A1), "n_A2": len(A2), "medians": M, "rowsI": rowsI, "rowsO": rowsO,
             "swap_carries_donor_bin_diff": rate(bin2(r["lp"][1], M["med"]) == e["label"] for r, e in zip(res, rowsI) if e["kind"] == "diff"),
             "digits_kept": rate(r["correct"] for r in res),
             "Pstar_A2_acc": {k: rate(int(v[j]) == tA2[j] for j in range(len(A2))) for k, v in P.items()},
             "N": {f: score(m, rowsI, rowsO, f, False, f"N-{f}") for f in ("imm", "del")}, "runs": {}}
        save(R)
        log("setup", {k: R[k] for k in ("n_A1", "n_A2", "swap_carries_donor_bin_diff", "digits_kept", "Pstar_A2_acc")}, "N", R["N"])
    rowsI, rowsO = R["rowsI"], R["rowsO"]
    nmax = max(max(st) for *_, st in RUNS) * 8
    streams = {"I": T.i_examples(A1, by1, b, nmax, seeded("s0b-I"), para_bin=pb)}
    rO = seeded("s0b-O")
    streams["O"] = [{"arm": "O", "x": i["key"], "label": b(i)} for i in (A1[rO.randrange(len(A1))] for _ in range(nmax))]
    for name, fmt, arm, rank, lr, stops in RUNS:
        run = R["runs"].setdefault(name, {"fmt": fmt, "arm": arm, "rank": rank, "lr": lr, "evals": {}})
        for S in stops:
            if str(S) in run["evals"]:
                continue
            t0 = time.time()
            cfg = {"bs": 8, "lr": lr, "rank": rank, "l1": L1, "fmt": fmt}
            status, hist = T.train(m, arm, streams[arm], cfg, 0, None, S, os.path.join(CKPT, name + ".pt"), deadline=DEADLINE)
            if status == "yield":
                save(R)
                log("deadline near: yielding in", name, S)
                sys.exit(3)
            ev = score(m, rowsI, rowsO, fmt, True, f"{name}@{S}")
            ev["loss_hist"] = hist
            ev["minutes"] = (time.time() - t0) / 60
            run["evals"][str(S)] = ev
            m.remove_lora()
            torch.cuda.empty_cache()
            save(R)
            log(name, S, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ev.items() if k != "loss_hist"},
                "last loss", hist[-1]["loss"] if hist else None)
    R["minutes_total_last_call"] = (time.time() - t00) / 60
    R["done"] = True
    save(R)
    log("Stage 0b-train done")


if __name__ == "__main__":
    main()

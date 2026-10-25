"""Paper 9 stage 3: evaluate one reporter (N = untrained, or a trained arm and seed) on T1-T5.
In-place arms report in turn 2 of the conversation (adapter on in turn 2 only). X reads the transcript
as text in a fresh chat (T3 only). T5u applies the adapter to every position (ungated) on the 2-hop and
1-hop questions of the test cities. Resumable: skips a finished evaluation.
python stage_eval.py --arm I --seed 0 [--selftest]"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p9lib import P9, CONT_FIRST, log, rot_scores, save_json  # noqa: E402
from build import add_mc, onehop_fc_prompts, twohop_prompt, unit_natural  # noqa: E402
from items import load_screen  # noqa: E402
from stage_train import setup  # noqa: E402
import trainlib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["N", "O", "I", "J", "X"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    out, G, cfg, jvec = setup(a.selftest)
    tag = "N" if a.arm == "N" else f"{a.arm}_s{a.seed}"
    path = os.path.join(out, f"eval_{tag}.json")
    if os.path.exists(path):
        log("exists, skipping", path)
        return
    t0 = time.time()
    m = P9()
    lora = a.arm != "N"
    if lora:
        m.load_lora(torch.load(os.path.join(out, f"lora_{a.arm}_s{a.seed}.pt")))
    T = G["tests"]
    E = {"arm": a.arm, "seed": a.seed}
    if a.arm == "X":
        E["T3"] = trainlib.eval_external(m, T["T3"], lora=True)
    else:
        for name in ("T1", "T2a", "T2c", "T3", "T4"):
            E[name] = trainlib.eval_reports(m, T[name], cfg, jvec, lora=lora)
            log(name, "done")
        # T5 gated: general questions in turn 2
        units = [add_mc(m, unit_natural(m, e["x"], e["cont"]), e["q"], e["opts"]) for e in T["T5"]]
        res = m.run_units(units, lora=lora)
        E["T5"] = []
        for e, r in zip(T["T5"], res):
            s = rot_scores([r[k][0] for k in range(4)])
            E["T5"].append({"scores": s, "pick": e["opts"][max(range(4), key=lambda i: s[i])]})
        # T5u ungated: the adapter on every position, 2-hop and 1-hop questions of the test cities
        scr = load_screen()
        cities = sorted({e["x"] for e in T["T3"]})
        conts = list(CONT_FIRST.keys())
        r2 = m.score_fresh([twohop_prompt(m, c) for c in cities], lora=lora)
        prompts = []
        for c in cities:
            prompts += onehop_fc_prompts(m, c, scr[c]["cands"])
        r1 = m.score_fresh(prompts, lora=lora)
        E["T5u"] = []
        for k, c in enumerate(cities):
            s = rot_scores([r1[4 * k + q][0] for q in range(4)])
            E["T5u"].append({"city": c, "cont_pred": conts[max(range(5), key=lambda j: r2[k][1][j])],
                             "p1": scr[c]["cands"][max(range(4), key=lambda j: s[j])]})
    E["minutes"] = (time.time() - t0) / 60
    save_json(E, path)
    log("saved", path, f"{E['minutes']:.1f} min")


if __name__ == "__main__":
    main()

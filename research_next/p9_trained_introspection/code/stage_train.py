"""Paper 9 stage 2: train one reporter arm (O, I, J or X) with one seed. Resumable: skips a finished run.
python stage_train.py --arm I --seed 0 [--selftest]"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p9lib import P9, TINY, load_json, log, outdir, save_json  # noqa: E402
import trainlib  # noqa: E402

ROOT = os.path.dirname(HERE)


def setup(selftest):
    out = outdir("res_selftest" if selftest else "res")
    G = load_json(os.path.join(out, "gen.json"))
    cfg = dict(G["cfg"])
    if selftest:
        cfg.update({"steps": 3 if TINY else 12, "bs": 8})
    jv = torch.load(os.path.join(out, "jvec.pt"))
    cfg["base_norm"] = jv["base_norm"]
    return out, G, cfg, jv["jvec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["O", "I", "J", "X"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    out, G, cfg, jvec = setup(a.selftest)
    path = os.path.join(out, f"lora_{a.arm}_s{a.seed}.pt")
    if os.path.exists(path):
        log("exists, skipping", path)
        return
    t0 = time.time()
    m = P9()
    state, hist = trainlib.train(m, a.arm, G["train"][a.arm], cfg, a.seed, jvec=jvec)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    save_json({"arm": a.arm, "seed": a.seed, "hist": hist, "minutes": (time.time() - t0) / 60,
               "steps": cfg["steps"], "bs": cfg["bs"], "lr": cfg["lr"], "n_examples": len(G["train"][a.arm]),
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30},
              os.path.join(out, f"trainlog_{a.arm}_s{a.seed}.json"))
    log("saved", path, f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

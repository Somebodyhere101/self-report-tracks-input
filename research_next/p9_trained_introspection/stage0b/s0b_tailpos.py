"""Stage 0b-tailpos (exploratory, pilot products only): where along the immediate report tail does the swapped-in
two-bin confidence stop being linearly readable? Base model, Stage 0a's rows (590 natural pilot items, 474 swaps of the
four digit positions from layer 16). GPU part: block outputs at the last digit (d4) and at every position of the
immediate tail, layers 12/16/20/24/27. CPU part (--probe): Stage 0a's B2 (a probe trained on natural rows of training
products, tested on held-out products' swapped rows whose two-bin label differs; agreement with the donor vs the
recipient), per position and layer, with Stage 0a's probe code (stage0/s0a_probe.py, imported read-only).
Features go to S0B_FEATDIR/feats_tailpos.pt; results to stage0b/tailpos.json."""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import torch

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
P9DIR = os.path.dirname(HERE)
FEAT = os.path.join(os.environ.get("S0B_FEATDIR", r"C:\Users\<user>\AppData\Local\Temp\p9\stage0b"), "feats_tailpos.pt")
LAYERS = [12, 16, 20, 24, 27]


def extract():
    sys.path.insert(0, os.path.join(P9DIR, "code_v4"))
    import v4lib
    import v3lib
    rows = json.load(open(os.path.join(HERE, "probe_imm", "stage0", "rows.json"), encoding="utf-8"))["rows"]
    m = v4lib.P9v4()
    units = [v4lib.unit_natural4(m, r["x"]) if r["kind"] == "nat" else v4lib.unit_swap4(m, r["x"], r["y"], list(range(16, m.L)))
             for r in rows]
    tail = v4lib.tail_imm(m, v4lib.rotate2(v4lib.BINS, 0))
    n_pos = 1 + len(tail)  # d4, then every tail position
    feats = torch.zeros(len(units), n_pos, len(LAYERS), m.d, dtype=torch.float16)
    with torch.no_grad():
        for s in range(0, len(units), 24):
            batch = list(range(s, min(s + 24, len(units))))
            rws, _, _ = m._flatten(units, batch)
            cache, h, cap = m._turn1_pass(rws, capture_layers=tuple(LAYERS))
            T1 = len(rws[0]["ids"])
            tg, off = [], 0
            for ui in batch:
                tg.append(off + units[ui]["tgt"])
                off += len(units[ui]["rows"])
            for k, (ui, t) in enumerate(zip(batch, tg)):
                for j, l in enumerate(LAYERS):
                    feats[ui, 0, j] = cap[l][t, T1 - 1].to(torch.float16).cpu()
            del h, cap
            c2 = m._select_cache(cache, tg)
            x = torch.tensor([tail] * len(tg), device=m.dev)
            pos = (torch.arange(len(tail), device=m.dev) + T1)[None].expand(len(tg), -1)
            m.capture = {l: None for l in LAYERS}
            m.patch2, m.ops, m.ppos0 = None, None, T1
            try:
                m.hf.model(input_ids=x, past_key_values=c2, position_ids=pos, use_cache=True)
                cap2 = dict(m.capture)
            finally:
                m.capture, m.ppos0 = None, 0
            for k, ui in enumerate(batch):
                for j, l in enumerate(LAYERS):
                    feats[ui, 1:, j] = cap2[l][k].to(torch.float16).cpu()
            del cache, c2, cap2
    os.makedirs(os.path.dirname(FEAT), exist_ok=True)
    torch.save({"feats": feats, "layers": LAYERS, "tail_tokens": [m.tok.decode([t]) for t in tail]}, FEAT)
    v3lib.log("tailpos features", tuple(feats.shape))


def probe():
    sys.path.insert(0, os.path.join(P9DIR, "stage0"))
    import s0a_probe as S  # Stage 0a's probe code, read-only
    rows = json.load(open(os.path.join(HERE, "probe_imm", "stage0", "rows.json"), encoding="utf-8"))["rows"]
    D = torch.load(FEAT)
    F, toks = D["feats"], D["tail_tokens"]
    sw = np.array([r["kind"] == "swap" for r in rows])
    prod = np.array([r["product"] for r in rows])
    own = np.array([r["bin2_x"] for r in rows])
    eff = np.array([r["bin2_y"] if r["kind"] == "swap" else r["bin2_x"] for r in rows])
    diff = sw & (own != eff)
    allg = np.unique(prod)
    dpos = np.where(diff)[0]
    out = {"positions": ["d4"] + toks, "layers": LAYERS, "n_B": int(diff.sum()), "B2_donor": {}, "B2_recipient": {}}
    for j, l in enumerate(LAYERS):
        for p in range(F.shape[1]):
            X = F[:, p, j].float().numpy()
            pred = np.full(diff.sum(), -1)
            for tr, te in S.folds(allg):
                trp, tep = set(allg[tr]), set(allg[te])
                mtr = (~sw) & np.isin(prod, list(trp))
                clf = S.fit(X[mtr], own[mtr], 0.5)  # Stage 0a picked lam = 0.5 almost everywhere
                sel = np.isin(prod[dpos], list(tep))
                pred[sel] = clf.predict(X[dpos[sel]])
            out["B2_donor"][f"L{l}_p{p}"] = float((pred == eff[diff]).mean())
            out["B2_recipient"][f"L{l}_p{p}"] = float((pred == own[diff]).mean())
        print(l, [round(out["B2_donor"][f"L{l}_p{p}"], 2) for p in range(F.shape[1])], flush=True)
    json.dump(out, open(os.path.join(HERE, "tailpos.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    t0 = time.time()
    probe() if "--probe" in sys.argv else extract()
    print("done", round((time.time() - t0) / 60, 1), "min", flush=True)

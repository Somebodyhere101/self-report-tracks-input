"""Stage 0a (paper 9 v4), GPU part: residual-stream features for natural and swapped PILOT rows.

Base model only (no adapter). Turn 1 = v3's teacher-forced multiplication (swapped rows: v3 unit_swap at
layers >= l1 = 16, in-batch exact copy of a same-product donor's R). Turn 2 = v4's two-option report prompt
(rotation 0: A) Very sure, B) Less sure), run on the turn-1 cache. Captured block outputs:
  rep: the final token of the turn-2 prompt (the "(" whose next token is the report letter);
  R0 : the first answer-digit position of turn 1 (the position whose logits give lp2).
Only pilot products are used. code_v3 / code_v4 are imported read-only.
Features go to FEAT (outside P9, large); row metadata goes to stage0/rows.json."""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import torch

torch.set_num_threads(4)

P9DIR = os.environ.get("P9DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(P9DIR, "stage0")
FEAT = os.environ.get("S0_FEAT", r"C:\Users\<user>\AppData\Local\Temp\p6\stage0\feats.pt")
sys.path.insert(0, os.path.join(P9DIR, "code_v3"))
sys.path.insert(0, os.path.join(P9DIR, "code_v4"))
import v3lib  # noqa: E402
import v4lib  # noqa: E402

MAX_SWAP = 1400


def main():
    t0 = time.time()
    scr = v3lib.load_json(os.path.join(P9DIR, "res_v3", "screen.json"))["items"]
    gen_th = [-0.018149960786104202, -0.0013263961300253868, -7.486063259420916e-05]  # res_v3/gen.json "th" (train quartiles)
    med = gen_th[1]  # = median lp2 of correct train items (checked: statistics.median over 1111 train items)
    sp = v3lib.split_of_product()
    its = sorted([i for i in scr.values() if i["correct"] and sp[i["product"]] == "pilot"], key=lambda z: z["key"])
    assert all(i["split"] == "pilot" for i in its)
    b4 = lambda lp: v3lib.bin_of(lp, gen_th)  # noqa: E731
    b2 = lambda lp: v4lib.bin_of(lp, med)  # noqa: E731
    by_p = {}
    for i in its:
        by_p.setdefault(i["product"], []).append(i)
    pairs = [(x, y) for p, v in by_p.items() for x in v for y in v
             if x["key"] != y["key"] and b4(x["lp2"]) != b4(y["lp2"])]
    rng = v3lib.seeded("p9-stage0a")
    rng.shuffle(pairs)
    pairs = pairs[:MAX_SWAP]
    v3lib.log("pilot items", len(its), "products", len(by_p), "4-bin-different ordered pairs", len(pairs),
              "of which 2-bin-different", sum(b2(x["lp2"]) != b2(y["lp2"]) for x, y in pairs))

    m = v3lib.P9()
    L = m.L
    LAYERS = [8, 12, 16, 20, 24, L - 1]
    l1 = 16
    tail = m.tail(v4lib.Q_REPORT, v4lib.rotate2(v4lib.BINS, 0))
    ab = [m.letter_ids[0], m.letter_ids[1]]

    units, meta = [], []
    for i in its:
        units.append(v3lib.unit_natural(m, i["key"]))
        meta.append({"kind": "nat", "x": i["key"], "y": None, "product": i["product"], "lp2_x": i["lp2"], "lp2_y": None})
    for x, y in pairs:
        units.append(v3lib.unit_swap(m, x["key"], y["key"], list(range(l1, L))))
        meta.append({"kind": "swap", "x": x["key"], "y": y["key"], "product": x["product"], "lp2_x": x["lp2"],
                     "lp2_y": y["lp2"]})

    feats = {pos: {l: [None] * len(units) for l in LAYERS} for pos in ("rep", "R0")}
    groups = {}
    for ui, u in enumerate(units):
        groups.setdefault(m._key(u), []).append(ui)
    with torch.no_grad():
        for key, uis in groups.items():
            for s in range(0, len(uis), 24):
                batch = uis[s:s + 24]
                rows, _, _ = m._flatten(units, batch)
                cache, h, cap = m._turn1_pass(rows, capture_layers=tuple(LAYERS))
                T1 = len(rows[0]["ids"])
                tg_rows, off = [], 0
                for ui in batch:
                    tg_rows.append(off + units[ui]["tgt"])
                    off += len(units[ui]["rows"])
                # turn-1 measurement (as v3 measure) and R0 capture
                for ui, t in zip(batch, tg_rows):
                    p0 = rows[t]["r0"]
                    lg = torch.log_softmax(m.hf.lm_head(h[t, p0 - 1:p0 + 3]).float(), -1)
                    tgt = torch.tensor(rows[t]["ids"][p0:p0 + 4], device=m.dev)
                    lp = lg.gather(-1, tgt[:, None])[:, 0]
                    meta[ui]["lp2_meas"] = float(lp[1])
                    meta[ui]["correct_meas"] = bool((lg.argmax(-1) == tgt).all())
                    for l in LAYERS:
                        feats["R0"][l][ui] = cap[l][t, p0].to(torch.float16).cpu()
                del h, cap
                # turn 2 on the cache, target rows only, capturing every chosen block at the last tail token
                c2 = m._select_cache(cache, tg_rows)
                K = len(tg_rows)
                x = torch.tensor([tail] * K, device=m.dev)
                pos = (torch.arange(len(tail), device=m.dev) + T1)[None].expand(K, -1)
                m.capture = {l: None for l in LAYERS}
                m.patch2, m.ops, m.ppos0 = None, None, T1
                m.lora_on(False)
                try:
                    hh = m.hf.model(input_ids=x, past_key_values=c2, position_ids=pos, use_cache=True).last_hidden_state
                    cap2 = dict(m.capture)
                finally:
                    m.capture, m.ppos0 = None, 0
                lg2 = m.hf.lm_head(hh[:, -1]).float()
                lpab = torch.log_softmax(lg2[:, ab], -1).cpu()
                for k, ui in enumerate(batch):
                    meta[ui]["rep_lpA"] = float(lpab[k, 0])
                    for l in LAYERS:
                        feats["rep"][l][ui] = cap2[l][k, -1].to(torch.float16).cpu()
                del cache, c2, hh, cap2
    for ui, r in enumerate(meta):
        r["bin4_x"] = b4(r["lp2_x"])
        r["bin2_x"] = b2(r["lp2_x"])
        if r["kind"] == "swap":
            r["bin4_y"] = b4(r["lp2_y"])
            r["bin2_y"] = b2(r["lp2_y"])
        r["bin4_meas"] = b4(r["lp2_meas"])
        r["bin2_meas"] = b2(r["lp2_meas"])
    out = {pos: {l: torch.stack(v) for l, v in d.items()} for pos, d in feats.items()}
    os.makedirs(os.path.dirname(FEAT), exist_ok=True)
    torch.save({"feats": out, "layers": LAYERS, "L": L}, FEAT)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rows.json"), "w", encoding="utf-8") as f:
        json.dump({"layers": LAYERS, "L": L, "l1": l1, "th4": gen_th, "med": med, "tail_rotation": 0,
                   "rows": meta, "minutes": (time.time() - t0) / 60}, f, indent=0)
    v3lib.log("done", len(meta), "rows in", round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    main()

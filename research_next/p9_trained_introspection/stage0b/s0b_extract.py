"""Stage 0b-probe (paper 9 v4; exploratory, pilot products only), GPU part: Stage 0a's extraction (stage0/s0a_extract.py,
same rows: every correct pilot item, and the same 474 ordered same-product pairs whose four-bin differs) for the two
v4 report formats. Base model only (no adapter).
Turn 1 ends with the four digits; swapped rows copy the donor's block outputs at the four digit positions from layer 16
(R = the digits). Two report tails run on the same turn-1 cache, option order 0 (A) Very sure, B) Less sure):
  imm: the immediate tail, appended to the answer right after the digits (v4lib.tail_imm);
  del: the delayed tail, closing the answer and asking in turn 2 (v4lib.tail_del).
Captured block outputs: rep = the last tail token (whose next token is the report letter), for each tail; R0 = the
first answer-digit position (whose logits give lp2). Writes, for each tail, probe_<fmt>/stage0/rows.json (the layout
Stage 0a's probe script reads) and the features to FEATDIR/feats_<fmt>.pt (outside the study folder, large)."""
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
FEATDIR = os.environ.get("S0B_FEATDIR", r"C:\Users\<user>\AppData\Local\Temp\p9\stage0b")
sys.path.insert(0, os.path.join(P9DIR, "code_v4"))
import v4lib  # noqa: E402
import v3lib  # noqa: E402  (imported by v4lib; v3, frozen)

MAX_SWAP = 1400


def main():
    t0 = time.time()
    scr = v3lib.load_json(os.path.join(P9DIR, "res_v3", "screen.json"))["items"]
    gen_th = [-0.018149960786104202, -0.0013263961300253868, -7.486063259420916e-05]  # res_v3/gen.json "th"
    med = gen_th[1]
    sp = v3lib.split_of_product()
    its = sorted([i for i in scr.values() if i["correct"] and sp[i["product"]] == "pilot"], key=lambda z: z["key"])
    b4 = lambda lp: v3lib.bin_of(lp, gen_th)  # noqa: E731
    b2 = lambda lp: v4lib.bin_of(lp, med)  # noqa: E731
    by_p = {}
    for i in its:
        by_p.setdefault(i["product"], []).append(i)
    pairs = [(x, y) for p, v in by_p.items() for x in v for y in v if x["key"] != y["key"] and b4(x["lp2"]) != b4(y["lp2"])]
    rng = v3lib.seeded("p9-stage0a")  # Stage 0a's seed: the same pairs
    rng.shuffle(pairs)
    pairs = pairs[:MAX_SWAP]
    v3lib.log("pilot items", len(its), "pairs", len(pairs), "2-bin-different", sum(b2(x["lp2"]) != b2(y["lp2"]) for x, y in pairs))
    m = v4lib.P9v4()
    L = m.L
    LAYERS = [8, 12, 16, 20, 24, L - 1]
    l1 = 16
    tails = {"imm": v4lib.tail_imm(m, v4lib.rotate2(v4lib.BINS, 0)), "del": v4lib.tail_del(m, v4lib.rotate2(v4lib.BINS, 0))}
    ab = [m.letter_ids[0], m.letter_ids[1]]
    units, meta = [], []
    for i in its:
        units.append(v4lib.unit_natural4(m, i["key"]))
        meta.append({"kind": "nat", "x": i["key"], "y": None, "product": i["product"], "lp2_x": i["lp2"], "lp2_y": None})
    for x, y in pairs:
        units.append(v4lib.unit_swap4(m, x["key"], y["key"], list(range(l1, L))))
        meta.append({"kind": "swap", "x": x["key"], "y": y["key"], "product": x["product"], "lp2_x": x["lp2"], "lp2_y": y["lp2"]})
    feats = {f: {l: [None] * len(units) for l in LAYERS} for f in ("imm", "del", "R0")}
    lpA = {f: [None] * len(units) for f in ("imm", "del")}
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
                for f, tail in tails.items():
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
                    hl = hh[:, -1]
                    lg2 = m.letter_fix(hl, m.hf.lm_head(hl).float())
                    lpab = torch.log_softmax(lg2[:, ab], -1).cpu()
                    for k, ui in enumerate(batch):
                        lpA[f][ui] = float(lpab[k, 0])
                        for l in LAYERS:
                            feats[f][l][ui] = cap2[l][k, -1].to(torch.float16).cpu()
                    del c2, hh, cap2
                del cache
    for r in meta:
        r["bin4_x"], r["bin2_x"] = b4(r["lp2_x"]), b2(r["lp2_x"])
        if r["kind"] == "swap":
            r["bin4_y"], r["bin2_y"] = b4(r["lp2_y"]), b2(r["lp2_y"])
        r["bin4_meas"], r["bin2_meas"] = b4(r["lp2_meas"]), b2(r["lp2_meas"])
    os.makedirs(FEATDIR, exist_ok=True)
    for f in ("imm", "del"):
        out = {"rep": {l: torch.stack(feats[f][l]) for l in LAYERS}, "R0": {l: torch.stack(feats["R0"][l]) for l in LAYERS}}
        torch.save({"feats": out, "layers": LAYERS, "L": L}, os.path.join(FEATDIR, f"feats_{f}.pt"))
        rows_f = [dict(r, rep_lpA=lpA[f][i]) for i, r in enumerate(meta)]
        od = os.path.join(HERE, f"probe_{f}", "stage0")
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "rows.json"), "w", encoding="utf-8") as fh:
            json.dump({"layers": LAYERS, "L": L, "l1": l1, "th4": gen_th, "med": med, "tail_rotation": 0, "format": f,
                       "R": "the four answer digits", "rows": rows_f, "minutes": (time.time() - t0) / 60}, fh, indent=0)
    v3lib.log("done", len(meta), "rows in", round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    main()

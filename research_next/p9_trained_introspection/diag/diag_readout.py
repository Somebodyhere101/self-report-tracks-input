"""Post-bed-failure diagnosis (pilot-split cities only; not part of any preregistered analysis).
Why did the placeholder readout of R fail calibration (0.12) in the v1 pilot? CPU, Qwen3-1.7B, ~12 items.
D1: probability mass on the letters A-D at the answer position, and the top tokens, for: the fresh 1-hop
    question; turn-2 questions with the real city; the placeholder readout with R copied.
D2: calibration (pick == gazetteer) of readout variants on known pilot cities:
    city   - turn 2 asked with the real city present (no copy)
    prior  - placeholder, nothing copied
    R      - placeholder, R copied at every layer (the v1 readout)
    R+city - placeholder, R AND the city positions copied at every layer (positive control)
    each with letter scoring (4 rotations) and with country-name scoring ("The country is <name>")."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")
sys.path.insert(0, CODE)
from p9lib import P9, Q_READ_COUNTRY, Q_REPORT, rot_scores, rotate, load_json  # noqa: E402
from build import onehop_fc_prompts  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
m = P9(device=dev)
scr = {i["city"]: i for i in load_json(os.path.join(os.path.dirname(HERE), "res", "screen.json"))["items"] if i.get("ok")}
known = sorted(c for c, i in scr.items() if i["split"] == "pilot" and i["acc2"] and i["p1"] == i["country"]
               and i.get("free1") == i["country"])[:12]
L = m.L
letters = m.letter_ids


def top(lg, k=5):
    p = torch.softmax(lg.float(), -1)
    v, ix = p.topk(k)
    return [(m.tok.decode([int(i)]), round(float(x), 3)) for x, i in zip(v, ix)], round(float(p[letters].sum()), 4)


@torch.no_grad()
def turn12(rows, tails, sel):
    cache, _, _ = m._turn1_pass(rows)
    return m._turn2_pass(cache, len(rows[0]["ids"]), sel, tails, lora=False)


def rows_for(city, cont, variant):
    ids, rc, r0 = m.turn1(city, cont)
    src = {"ids": ids, "r0": r0}
    if variant == "city":
        return [src], 0
    pids, _, _ = m.turn1(city, cont, placeholder=True)
    p = {"ids": pids, "r0": r0}
    if variant == "prior":
        return [p], 0
    if variant == "R":
        p["copy"] = {l: 0 for l in range(L)}
        return [src, p], 1
    if variant == "R+city":  # copy the city positions too: patch mask starts at the city span
        p["copy"] = {l: 0 for l in range(L)}
        p["r0"] = r0
        return [src, p], 1
    raise ValueError(variant)


# the city+R variant needs a mask from the city start; emulate by lowering r0 for both rows
def rows_for_cityR(city, cont):
    ids, rc, r0 = m.turn1(city, cont)
    pids, _, _ = m.turn1(city, cont, placeholder=True)
    return [{"ids": ids, "r0": rc}, {"ids": pids, "r0": rc, "copy": {l: 0 for l in range(L)}}], 1


print("== D1: letter mass and top tokens", flush=True)
c0 = known[0]
it = scr[c0]
fresh = onehop_fc_prompts(m, c0, it["cands"])[0]
with torch.no_grad():
    h = m.hf.model(input_ids=torch.tensor([fresh], device=m.dev)).last_hidden_state[0, -1]
    print("fresh 1-hop:", top(m.hf.lm_head(h)))
for name, q in (("report", Q_REPORT), ("readout-q", Q_READ_COUNTRY)):
    for variant in ("city", "R"):
        rows, t = rows_for(c0, it["continent"], variant)
        lg = turn12(rows, [m.tail(q, it["cands"])], [t])[0]
        print(name, variant, top(lg))

print("== D2: calibration on", len(known), "known pilot cities", flush=True)
for variant in ("city", "prior", "R", "R+city"):
    hits_l, hits_n = 0, 0
    for c in known:
        it = scr[c]
        rows, t = rows_for_cityR(c, it["continent"]) if variant == "R+city" else rows_for(c, it["continent"], variant)
        tails = [m.tail(Q_READ_COUNTRY, rotate(it["cands"], r)) for r in range(4)]
        lg = turn12(rows, tails, [t] * 4)
        lp = torch.log_softmax(lg[:, letters], -1).tolist()
        s = rot_scores(lp)
        hits_l += it["cands"][max(range(4), key=lambda k: s[k])] == it["country"]
        # name scoring: the readout question without options, assistant prefill "The country is", then the name
        base = m._enc("<|im_start|>user\n" + Q_READ_COUNTRY + " Answer with the country name only.<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        names = [m._enc(n) for n in it["cands"]]
        tails = [base + nm for nm in names]
        with torch.no_grad():
            cache, _, _ = m._turn1_pass(rows)
            T1 = len(rows[0]["ids"])
            mm = max(len(x) for x in tails)
            x = torch.tensor([tt + [m.pad_id] * (mm - len(tt)) for tt in tails], device=m.dev)
            c2 = m._select_cache(cache, [t] * 4)
            hh = m.hf.model(input_ids=x, past_key_values=c2,
                            position_ids=(torch.arange(mm, device=m.dev) + T1)[None].expand(4, -1)).last_hidden_state
            lpv = torch.log_softmax(m.hf.lm_head(hh).float(), -1)
            sc = []
            for k, nm in enumerate(names):
                st = len(base)
                sc.append(sum(float(lpv[k, st - 1 + j, nm[j]]) for j in range(len(nm))))
        hits_n += it["cands"][max(range(4), key=lambda k: sc[k])] == it["country"]
    print(f"{variant:7s} letters {hits_l}/{len(known)}   names {hits_n}/{len(known)}", flush=True)

"""Post-bed-failure diagnosis 2 (pilot-split cities only; not part of any preregistered analysis). CPU.
A: readout calibration when the placeholder row copies the city span AND R only at layers >= l (l = 0..20):
   is the country readable from the late-layer city states alone (early layers left as the placeholder's)?
B: untrained report (city visible) on city-span+R swaps at layers >= l from a different-country,
   same-continent donor: does it name the recipient (early-layer identity) or the donor (late layers)?
C: readout of those swapped rows (copy from the same l): does the swap carry the donor's country?"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")
sys.path.insert(0, CODE)
from p9lib import P9, Q_READ_COUNTRY, Q_REPORT, rot_scores, rotate, load_json  # noqa: E402
from build import fill  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
m = P9(device="cpu")
scr = {i["city"]: i for i in load_json(os.path.join(os.path.dirname(HERE), "res", "screen.json"))["items"] if i.get("ok")}
known = sorted(c for c, i in scr.items() if i["split"] == "pilot" and i["acc2"] and i["p1"] == i["country"]
               and i.get("free1") == i["country"])
L = m.L
LET = m.letter_ids


@torch.no_grad()
def score(rows, t, q, cands):
    cache, _, _ = m._turn1_pass(rows)
    lg = m._turn2_pass(cache, len(rows[0]["ids"]), [t] * 4, [m.tail(q, rotate(cands, r)) for r in range(4)], lora=False)
    s = rot_scores(torch.log_softmax(lg[:, LET], -1).tolist())
    return cands[max(range(4), key=lambda k: s[k])]


def row(city, cont, ph=False):
    ids, rc, r0 = m.turn1(city, cont, placeholder=ph)
    return {"ids": ids, "r0": rc}  # mask from the city start: city span + R


print("== A: readout calibration, copy city+R at layers >= l (12 known pilot cities)", flush=True)
items = known[:12]
for l in (0, 4, 8, 12, 16, 20):
    hit = 0
    for c in items:
        it = scr[c]
        src, p = row(c, it["continent"]), row(c, it["continent"], ph=True)
        p["copy"] = {k: 0 for k in range(l, L)}
        hit += score([src, p], 1, Q_READ_COUNTRY, it["cands"]) == it["country"]
    print(f"l={l:2d} readout = gazetteer {hit}/{len(items)}", flush=True)

print("== B/C: city+R swaps from a different-country donor (same token count and continent)", flush=True)
pairs = []
for c in known:
    it = scr[c]
    for d in known:
        jt = scr[d]
        if d != c and jt["n"] == it["n"] and jt["continent"] == it["continent"] and jt["country"] != it["country"]:
            pairs.append((c, d))
            break
    if len(pairs) >= 12:
        break
for l in (4, 8, 12, 16, 20):
    rec, don, ro_don = 0, 0, 0
    for c, d in pairs:
        it, jt = scr[c], scr[d]
        cands = fill([jt["country"], it["country"]], "diag" + c)
        y, x = row(d, it["continent"]), row(c, it["continent"])
        x["copy"] = {k: 0 for k in range(l, L)}
        pick = score([y, x], 1, Q_REPORT, cands)
        rec += pick == it["country"]
        don += pick == jt["country"]
        p = row(c, it["continent"], ph=True)
        p["copy"] = {k: 0 for k in range(l, L)}  # readout of the swapped row = donor's states at layers >= l
        ro_don += score([y, x, p], 2, Q_READ_COUNTRY, cands) == jt["country"]
    print(f"l={l:2d} untrained report: recipient {rec}/{len(pairs)} donor {don}/{len(pairs)};"
          f" readout of swapped row = donor {ro_don}/{len(pairs)}", flush=True)

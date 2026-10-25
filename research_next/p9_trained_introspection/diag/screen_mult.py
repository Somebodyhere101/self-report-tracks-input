"""Task-search screen, candidate 3 (paper 8's multiplication; exploratory; CPU <= 4 threads).
Turn 1: user "What is {a} * {b}?" + assistant prefill "{a} * {b} = " + the product's 4 digits (teacher-forced).
Intermediate: the confidence profile over the answer digits, i.e. the model's own probability of each emitted
digit (an output of the computation, not a lens). Label: the least-sure digit (lowest probability).
Gates: PG1 accuracy (greedy) >= 0.70; PG2 the two definitions of least-sure (lowest probability; highest
entropy) agree >= 0.70 and the label is not degenerate; PG3 swapping the answer positions (digits 1-3 and
the closing tokens: they predict digits 2-4) at layers >= l from a same-product donor moves the least-sure
digit (among digits 2-4) to the donor's in >= 0.90 of rows where they differ, with the digits unchanged.
Also: the untrained report "Which digit of your answer were you least sure of?" on natural and swapped rows."""
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
from p9lib import P9, rot_scores, rotate  # noqa: E402

torch.set_num_threads(4)
m = P9(device="cpu")
L = m.L
LET = m.letter_ids
OPTS = ["the first digit", "the second digit", "the third digit", "the fourth digit"]
REP = "Which digit of your answer were you least sure of?"


def turn1(a, b):
    p = m.fresh(f"What is {a} * {b}?") + m._enc(f"{a} * {b} = ")
    d = m._enc(str(a * b))
    assert len(d) == 4
    return p + d + m._enc("<|im_end|>\n"), len(p)


@torch.no_grad()
def profile(rows):
    """Per row: log-prob and entropy of each of the 4 emitted digits (positions p0-1 .. p0+2)."""
    _, h, _ = m._turn1_pass(rows)
    out = []
    for b, r in enumerate(rows):
        p0 = r["p0"]
        lg = torch.log_softmax(m.hf.lm_head(h[b, p0 - 1:p0 + 3]).float(), -1)
        tg = torch.tensor(r["ids"][p0:p0 + 4], device=m.dev)
        lp = lg.gather(-1, tg[:, None])[:, 0]
        ent = -(lg.exp() * lg).sum(-1)
        am = lg.argmax(-1)
        out.append((lp.tolist(), ent.tolist(), bool((am == tg).all())))
    return out


# same-product pairs with 4-digit products and two-digit factors
prods = {}
for a in range(11, 100):
    for b in range(a, 100):
        if 1000 <= a * b <= 9999:
            prods.setdefault(a * b, []).append((a, b))
multi = sorted(p for p, v in prods.items() if len(v) >= 2)
rng = random.Random(9)
rng.shuffle(multi)
pairs = [tuple(rng.sample(prods[p], 2)) for p in multi[:20]]
acc, agree, lsd, nat_rep = 0, 0, [], []
rows_nat = []
for (a, b), _ in pairs:
    ids, p0 = turn1(a, b)
    rows_nat.append({"ids": ids, "r0": p0, "p0": p0})
prof = profile(rows_nat)
for (lp, ent, ok) in prof:
    acc += ok
    k1 = min(range(4), key=lambda k: lp[k])
    k2 = max(range(4), key=lambda k: ent[k])
    agree += k1 == k2
    lsd.append(k1)
from collections import Counter  # noqa: E402
print(f"PG1 accuracy (all 4 digits argmax) {acc}/{len(pairs)}")
print(f"PG2 least-sure by probability vs by entropy agree {agree}/{len(pairs)}; distribution {Counter(lsd)}")
print("   example min log-probs:", [round(min(p[0]), 4) for p in prof[:8]])
# swaps: answer positions p0 .. end at layers >= l from the donor (same product -> identical tokens there)
for l in (0, 14, 21):
    moved, differ, kept = 0, 0, 0
    for ((a, b), (c, d)) in pairs:
        x_ids, p0 = turn1(a, b)
        y_ids, p0y = turn1(c, d)
        assert p0 == p0y and len(x_ids) == len(y_ids)
        y = {"ids": y_ids, "r0": p0, "p0": p0}
        x = {"ids": x_ids, "r0": p0, "p0": p0, "copy": {k: 0 for k in range(l, L)}}
        xn = {"ids": x_ids, "r0": p0, "p0": p0}
        (ly, _, _), (lx, _, okx), (ln, _, _) = profile([y, x, xn])
        dy = 1 + min(range(3), key=lambda k: ly[1 + k])
        dn = 1 + min(range(3), key=lambda k: ln[1 + k])
        dx = 1 + min(range(3), key=lambda k: lx[1 + k])
        kept += okx
        if dy != dn:
            differ += 1
            moved += dx == dy
    print(f"PG3 swap from layer {l:2d}: least-sure digit (2-4) moves to the donor's {moved}/{differ} "
          f"(rows where they differ); digits unchanged {kept}/{len(pairs)}", flush=True)
# untrained report on natural rows (4 rotations)
hits = 0
with torch.no_grad():
    for r, (lp, _, _) in zip(rows_nat, prof):
        cache, _, _ = m._turn1_pass([r])
        lg = m._turn2_pass(cache, len(r["ids"]), [0] * 4, [m.tail(REP, rotate(OPTS, q)) for q in range(4)], lora=False)
        s = rot_scores(torch.log_softmax(lg[:, LET], -1).tolist())
        nat_rep.append(max(range(4), key=lambda k: s[k]))
        hits += nat_rep[-1] == min(range(4), key=lambda k: lp[k])
print(f"untrained report names the least-sure digit {hits}/{len(pairs)}; report picks {Counter(nat_rep)}")

"""Task-search screen, candidate 3b (exploratory; CPU <= 4 threads). Intermediate: the model's confidence in
the SECOND digit of its product (probability of the emitted digit at the answer position that predicts it,
i.e. computed after the input). Label: unsure = p2 below the median of the screened items.
PG2: label from probability vs label from entropy (median split) agree >= 0.70.
PG3: same-product swaps of the answer positions at layers >= l: when donor and recipient labels differ, the
swapped run's label equals the donor's >= 0.90, digits unchanged >= 0.95."""
import os, random, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
from p9lib import P9, rot_scores, rotate  # noqa: E402
torch.set_num_threads(4)
m = P9(device="cpu")
L = m.L
LET = m.letter_ids


def turn1(a, b):
    p = m.fresh(f"What is {a} * {b}?") + m._enc(f"{a} * {b} = ")
    return p + m._enc(str(a * b)) + m._enc("<|im_end|>\n"), len(p)


@torch.no_grad()
def p2(rows):
    _, h, _ = m._turn1_pass(rows)
    out = []
    for bi, r in enumerate(rows):
        p0 = r["r0"]
        lg = torch.log_softmax(m.hf.lm_head(h[bi, p0 - 1:p0 + 3]).float(), -1)
        tg = torch.tensor(r["ids"][p0:p0 + 4], device=m.dev)
        out.append((float(lg[1, tg[1]]), float(-(lg[1].exp() * lg[1]).sum()), bool((lg.argmax(-1) == tg).all())))
    return out


prods = {}
for a in range(11, 100):
    for b in range(a, 100):
        if 1000 <= a * b <= 9999:
            prods.setdefault(a * b, []).append((a, b))
rng = random.Random(21)
nat = [rng.choice(prods[p]) for p in rng.sample(sorted(prods), 48)]
rows = []
for a, b in nat:
    ids, p0 = turn1(a, b)
    rows.append({"ids": ids, "r0": p0})
res = []
for i in range(0, len(rows), 16):
    res += p2(rows[i:i + 16])
lps = sorted(r[0] for r in res)
ents = sorted(r[1] for r in res)
tp, th = lps[len(lps) // 2], ents[len(ents) // 2]
lab_p = [r[0] < tp for r in res]
lab_h = [r[1] > th for r in res]
print(f"PG1 accuracy {sum(r[2] for r in res)}/{len(res)}")
print(f"PG2 median log p2 {tp:.4f} (p2 {2.718281828 ** tp:.4f}); quartiles {[round(lps[k], 4) for k in (12, 24, 36)]}")
print(f"PG2 label by probability vs by entropy agree {sum(x == y for x, y in zip(lab_p, lab_h))}/{len(res)}")
multi = [p for p, v in prods.items() if len(v) >= 2]
pairs = [tuple(rng.sample(prods[p], 2)) for p in rng.sample(sorted(multi), 32)]
for l in (0, 8, 14, 21):
    moved, differ, kept = 0, 0, 0
    for (a, b), (c, d) in pairs:
        xi, p0 = turn1(a, b)
        yi, _ = turn1(c, d)
        y = {"ids": yi, "r0": p0}
        x = {"ids": xi, "r0": p0, "copy": {k: 0 for k in range(l, L)}}
        xn = {"ids": xi, "r0": p0}
        (ly, _, _), (lx, _, okx), (ln, _, _) = p2([y, x, xn])
        kept += okx
        if (ly < tp) != (ln < tp):
            differ += 1
            moved += (lx < tp) == (ly < tp)
    print(f"PG3 swap from layer {l:2d}: label moves to the donor's {moved}/{differ}; digits unchanged {kept}/{len(pairs)}", flush=True)
Q = "Were you sure of the second digit of your answer?"
hits, said_unsure = 0, 0
with torch.no_grad():
    for r, (lp, _, _) in zip(rows[:24], res[:24]):
        cache, _, _ = m._turn1_pass([r])
        opts = ["Yes, I was sure of it.", "No, I was unsure of it."]
        lg = m._turn2_pass(cache, len(r["ids"]), [0, 0], [m.tail(Q, [opts[q], opts[1 - q], "-", "-"][:2] + ["", ""])[:0] or
                                                         m._enc("<|im_start|>user\n" + Q + "\nA) " + opts[q] + "\nB) " + opts[1 - q] + "\nAnswer with the letter.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe answer is (")
                                                         for q in (0, 1)], lora=False)
        l2 = torch.log_softmax(lg[:, LET[:2]], -1).tolist()
        s_unsure = (l2[0][1] + l2[1][0]) / 2
        s_sure = (l2[0][0] + l2[1][1]) / 2
        u = s_unsure > s_sure
        said_unsure += u
        hits += u == (lp < tp)
print(f"untrained report matches the label {hits}/24; says unsure {said_unsure}/24")

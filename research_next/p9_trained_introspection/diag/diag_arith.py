"""Bed screen for a replacement task (exploratory; CPU; Qwen3-1.7B). Not part of any preregistered analysis.
Task: "What is {a} + {b}? Reply only with whether the sum is even or odd." (single digits). The intermediate is
the sum; the answer (parity) is many-to-one, so a donor with another sum of the same parity is answer-fixed.
Question 1: is the sum held AFTER the operands (region R), readable by the model's own downstream question
answering? Readout in a placeholder context ("a + b") with nothing / R / operands+R copied at every layer.
Question 2: does the untrained report ("What was the sum you computed?") follow an R-only swap from a donor
(computation) or the visible operands (input)?"""
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
from p9lib import P9, rot_scores, rotate  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
m = P9(device="cpu")
L = m.L
LET = m.letter_ids
Q = "What is {a} + {b}? Reply only with whether the sum is even or odd."


def prefix_len(a, b):
    full = m.chat(Q.format(a=a, b=b))
    cut = full.index(f"{a} + {b}") + len(f"{a} + {b}")
    return len(m._enc(full[:cut]))


def ops_start(a, b):
    full = m.chat(Q.format(a=a, b=b))
    return len(m._enc(full[:full.index(f"{a} + {b}")]))


def turn1(a, b, ans):
    return m._enc(m.chat(Q.format(a=a, b=b)) + ans + "<|im_end|>\n")


def answer_word(a, b):
    ids = m._enc(m.chat(Q.format(a=a, b=b)))
    with torch.no_grad():
        lg = m.hf(input_ids=torch.tensor([ids], device=m.dev)).logits[0, -1].float()
    words = ["Even", "Odd", "even", "odd"]
    wid = [m._enc(w)[0] for w in words]
    return words[max(range(4), key=lambda k: float(lg[wid[k]]))]


def cands_for(s, seed):
    pool = [v for v in range(0, 19) if v % 2 == s % 2 and v != s]
    rng = random.Random(seed)
    c = [s] + rng.sample(pool, 3)
    rng.shuffle(c)
    return [str(v) for v in c]


@torch.no_grad()
def pick(rows, t, q, cands):
    cache, _, _ = m._turn1_pass(rows)
    lg = m._turn2_pass(cache, len(rows[0]["ids"]), [t] * 4, [m.tail(q, rotate(cands, r)) for r in range(4)], lora=False)
    s = rot_scores(torch.log_softmax(lg[:, LET], -1).tolist())
    return cands[max(range(4), key=lambda k: s[k])]


rng = random.Random(7)
items = [(rng.randint(1, 9), rng.randint(1, 9)) for _ in range(16)]
PA, PB = " X", " Y"  # "What is  X +  Y?" tokenises like "What is 3 + 4?"
assert prefix_len(3, 4) == prefix_len(PA, PB) and ops_start(3, 4) == ops_start(PA, PB), "placeholder layout differs"
hits = {"prior": 0, "R": 0, "ops+R": 0}
rep = {"recipient": 0, "donor": 0}
ok = 0
for i, (a, b) in enumerate(items):
    s = a + b
    ans = answer_word(a, b)
    ok += (ans.lower() == ("even" if s % 2 == 0 else "odd"))
    n0 = prefix_len(a, b)
    cands = cands_for(s, i)
    src = {"ids": turn1(a, b, ans), "r0": n0}
    ph_ids = turn1(PA, PB, ans)
    assert len(ph_ids) == len(src["ids"])
    for v in hits:
        if v == "prior":
            rows, t = [{"ids": ph_ids, "r0": n0}], 0
        elif v == "R":
            rows, t = [src, {"ids": ph_ids, "r0": n0, "copy": {k: 0 for k in range(L)}}], 1
        else:  # operands too: mask from the first operand
            r_ops = ops_start(a, b)
            rows, t = [{"ids": src["ids"], "r0": r_ops}, {"ids": ph_ids, "r0": r_ops, "copy": {k: 0 for k in range(L)}}], 1
        hits[v] += pick(rows, t, "What was the sum?", cands) == str(s)
    # donor: another single-digit pair with a different sum of the same parity
    while True:
        a2, b2 = rng.randint(1, 9), rng.randint(1, 9)
        if a2 + b2 != s and (a2 + b2) % 2 == s % 2:
            break
    dc = cands_for(s, 100 + i)
    if str(a2 + b2) not in dc:
        dc[[c for c in range(4) if dc[c] != str(s)][0]] = str(a2 + b2)
    y = {"ids": turn1(a2, b2, ans), "r0": n0}
    x = {"ids": src["ids"], "r0": n0, "copy": {k: 0 for k in range(L)}}
    p = pick([y, x], 1, "What was the sum you computed?", dc)
    rep["recipient"] += p == str(s)
    rep["donor"] += p == str(a2 + b2)
    print(i, a, b, s, ans, "| R-swap report:", p, "(donor", a2 + b2, ")", flush=True)
n = len(items)
print(f"parity answers correct {ok}/{n}")
print("readout of the sum:", {k: f"{v}/{n}" for k, v in hits.items()})
print("untrained report under an R-only swap:", {k: f"{v}/{n}" for k, v in rep.items()})

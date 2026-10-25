"""Task-search screen, candidate 1 ("a+b compared to c"; exploratory; never analysed later; CPU <= 4 threads).
Turn 1: "Compute: {a} + {b} = ? Is the result greater than {c}? Answer yes or no."  (a, b two-digit)
Intermediate: the sum s, computed after the operands; answer = s > c (many-to-one, so a donor with another
sum on the same side of c and the same c is answer-fixed). R = every turn-1 position from " =" onward.
Gates as in v1 (same thresholds): PG1 yes/no accuracy >= 0.70; PG2 placeholder readout of the sum from R
(operands replaced by " # #") calibration >= 0.80 and two readouts (sum; its last digit) agree >= 0.70;
PG3 the readout of R-swapped rows names the donor's sum >= 0.90. Also: untrained report under the swap."""
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
Q = "Compute: {a} + {b} = ? Is the result greater than {c}? Answer yes or no."
PH = " # #"
RQ1 = "What was the result of the addition?"
RQ2 = "What was the last digit of the result of the addition?"
REP = "What was the result of the addition you computed?"
YES = [m._enc(w)[0] for w in ("Yes", "No")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16


def chat(a, b, c):
    return m.chat(Q.format(a=a, b=b, c=c))


def r0_of(a, b, c):
    t = chat(a, b, c)
    return len(m._enc(t[:t.index(f"{a} + {b}") + len(f"{a} + {b}")]))


def ops0(a, b, c):
    t = chat(a, b, c)
    return len(m._enc(t[:t.index(f"{a} + {b}")]))


@torch.no_grad()
def answer(a, b, c):
    ids = m._enc(chat(a, b, c))
    lg = m.hf(input_ids=torch.tensor([ids], device=m.dev)).logits[0, -1].float()
    return "Yes" if lg[YES[0]] > lg[YES[1]] else "No"


def ids1(a, b, c, ans):
    return m._enc(chat(a, b, c) + ans + "<|im_end|>\n")


def cands(s, c, rng, must=None):
    same = [v for v in range(20, 99) if (v > c) == (s > c) and v != s and v % 10 != s % 10]
    out = [s] + ([must] if must is not None else [])
    for v in rng.sample(same, len(same)):
        if len(out) == 4:
            break
        if v % 10 not in [x % 10 for x in out] and v not in out:
            out.append(v)
    rng.shuffle(out)
    return out


def score(units):
    res = m.run_units(units)
    return [rot_scores([r[k][0] for k in range(4)]) for r in res]


rng = random.Random(11)
items = []
while len(items) < N:
    a, b = rng.randint(10, 49), rng.randint(10, 49)
    s = a + b
    c = s + rng.choice([d for d in range(-15, 16) if abs(d) >= 3])
    if 10 <= c <= 99:
        items.append((a, b, c))
assert r0_of(47, 38, 60) == r0_of(PH, PH, 60) and ops0(47, 38, 60) == ops0(PH, PH, 60)
acc, cal, rel, tr, rep_r, rep_d, n_sw = 0, {"prior": 0, "R": 0, "ops+R": 0}, {"prior": 0, "R": 0, "ops+R": 0}, 0, 0, 0, 0
for i, (a, b, c) in enumerate(items):
    s = a + b
    ans = answer(a, b, c)
    acc += ans == ("Yes" if s > c else "No")
    r0, o0 = r0_of(a, b, c), ops0(a, b, c)
    cs = cands(s, c, rng)
    src = ids1(a, b, c, ans)
    ph = ids1(PH, PH, c, ans)
    assert len(ph) == len(src)
    units = []
    for v in cal:
        st = r0 if v != "ops+R" else o0
        rows = [{"ids": ph, "r0": st}] if v == "prior" else \
            [{"ids": src, "r0": st}, {"ids": ph, "r0": st, "copy": {k: 0 for k in range(L)}}]
        t = len(rows) - 1
        units.append({"rows": rows, "reqs": [(t, m.tail(RQ1, rotate([str(x) for x in cs], r)), r) for r in range(4)]})
        units.append({"rows": rows, "reqs": [(t, m.tail(RQ2, rotate([str(x % 10) for x in cs], r)), r) for r in range(4)]})
    sc = score(units)
    for j, v in enumerate(cal):
        k1 = max(range(4), key=lambda k: sc[2 * j][k])
        k2 = max(range(4), key=lambda k: sc[2 * j + 1][k])
        cal[v] += cs[k1] == s
        rel[v] += k1 == k2
    # donor: same c, same side, different sum and last digit, two-digit operands
    for _ in range(1000):
        a2, b2 = rng.randint(10, 49), rng.randint(10, 49)
        s2 = a2 + b2
        if s2 != s and (s2 > c) == (s > c) and s2 % 10 != s % 10:
            break
    else:
        continue
    n_sw += 1
    cs2 = cands(s, c, rng, must=s2)
    don = ids1(a2, b2, c, ans)
    sw = {"ids": src, "r0": r0, "copy": {k: 0 for k in range(L)}}
    p = {"ids": ph, "r0": r0, "copy": {k: 0 for k in range(L)}}
    units = [{"rows": [{"ids": don, "r0": r0}, sw, p],
              "reqs": [(2, m.tail(RQ1, rotate([str(x) for x in cs2], r)), r) for r in range(4)]},
             {"rows": [{"ids": don, "r0": r0}, sw],
              "reqs": [(1, m.tail(REP, rotate([str(x) for x in cs2], r)), r) for r in range(4)]}]
    sc = score(units)
    tr += cs2[max(range(4), key=lambda k: sc[0][k])] == s2
    pk = cs2[max(range(4), key=lambda k: sc[1][k])]
    rep_r += pk == s
    rep_d += pk == s2
    print(i, f"{a}+{b}={s} vs {c}: {ans}; donor {a2}+{b2}={s2}; report under swap -> {pk}", flush=True)
n = len(items)
print(f"PG1 yes/no accuracy {acc}/{n}")
print("PG2 readout calibration (= true sum):", {k: f"{v}/{n}" for k, v in cal.items()})
print("PG2 readout agreement (sum vs last digit):", {k: f"{v}/{n}" for k, v in rel.items()})
print(f"PG3 R-swap readout names the donor's sum {tr}/{n_sw}")
print(f"untrained report under R-swap: input sum {rep_r}/{n_sw}, donor sum {rep_d}/{n_sw}")

"""Task-search screen, step 1 (exploratory; CPU <= 4 threads): does Qwen3-1.7B (thinking off) do the task?
Two-digit addends 10..49; greedy free answer (<= 8 tokens), first integer or word parsed."""
import os
import random
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
from p9lib import P9  # noqa: E402

torch.set_num_threads(4)
m = P9(device="cpu")
FORMATS = {
    "F0_sum": ("Compute: {a} + {b} = ? Answer with the number only.", lambda s: str(s)),
    "F3_first_digit": ("Compute: {a} + {b}. What is the first digit of the sum? Answer with the digit only.",
                       lambda s: str(s)[0]),
    "F4_last_digit": ("Compute: {a} + {b}. What is the last digit of the sum? Answer with the digit only.",
                      lambda s: str(s)[-1]),
    "F5_above50": ("Compute: {a} + {b} = ? Is the result above or below 50? Answer with one word.",
                   lambda s: "above" if s > 50 else "below"),
    "F7_round10": ("Compute: {a} + {b}. Round the sum to the nearest ten and answer with that number only.",
                   lambda s: str(int(round(s / 10.0)) * 10) if s % 10 != 5 else None),
    "F8_tens": ("Compute: {a} + {b} = ? How many tens are in the result? Answer with the digit only.",
                lambda s: str(s // 10)),
}
rng = random.Random(5)
items = [(rng.randint(10, 49), rng.randint(10, 49)) for _ in range(24)]
for name, (q, truth) in FORMATS.items():
    outs = m.generate_free([m.fresh(q.format(a=a, b=b)) for a, b in items], n_new=8)
    ok, n = 0, 0
    for (a, b), o in zip(items, outs):
        t = truth(a + b)
        if t is None:
            continue
        n += 1
        if t.isdigit():
            g = re.findall(r"\d+", o)
            ok += bool(g) and g[0] == t
        else:
            ok += t in o.lower()
    print(f"{name:16s} {ok}/{n}   e.g. {items[0]} -> {outs[0]!r}", flush=True)

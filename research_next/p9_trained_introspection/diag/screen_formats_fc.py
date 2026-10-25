"""Task-search screen step 1b (exploratory; CPU <= 4 threads): forced choice over the valid answer tokens after
an assistant prefill "Answer: " (the study teacher-forces turn-1 answers, as v1 did with continents)."""
import os, random, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
from p9lib import P9  # noqa: E402
torch.set_num_threads(4)
m = P9(device="cpu")
DIG = [str(d) for d in range(10)]
FORMATS = {
    "F3_first_digit": ("Compute: {a} + {b}. What is the first digit of the sum? Answer with the digit only.", DIG, lambda s: str(s)[0]),
    "F4_last_digit": ("Compute: {a} + {b}. What is the last digit of the sum? Answer with the digit only.", DIG, lambda s: str(s)[-1]),
    "F5_above50": ("Compute: {a} + {b} = ? Is the result above or below 50? Answer with one word.", ["above", "below"], lambda s: "above" if s > 50 else "below"),
    "F8_tens": ("Compute: {a} + {b} = ? How many tens are in the result? Answer with the digit only.", DIG, lambda s: str(s // 10)),
    "F1_greater": ("Compute: {a} + {b} = ? Is the result greater than {c}? Answer yes or no.", ["yes", "no"], None),
}
rng = random.Random(5)
items = [(rng.randint(10, 49), rng.randint(10, 49)) for _ in range(40)]
cs = [a + b + rng.choice([d for d in range(-15, 16) if abs(d) >= 3]) for a, b in items]
for name, (q, opts, truth) in FORMATS.items():
    oid = [m._enc(o)[0] for o in opts]
    prompts = [m.fresh(q.format(a=a, b=b, c=c)) + m._enc("Answer: ") for (a, b), c in zip(items, cs)]
    res = m.score_fresh(prompts)
    ok = 0
    picks = []
    for (a, b), c, p in zip(items, cs, prompts):
        pass
    with torch.no_grad():
        for i, ((a, b), c) in enumerate(zip(items, cs)):
            x = torch.tensor([prompts[i]], device=m.dev)
            lg = m.hf(input_ids=x).logits[0, -1].float()
            k = max(range(len(opts)), key=lambda j: float(lg[oid[j]]))
            t = truth(a + b) if truth else ("yes" if a + b > c else "no")
            ok += opts[k] == t
            picks.append(opts[k])
    from collections import Counter
    print(f"{name:16s} {ok}/{len(items)}  picks {Counter(picks).most_common(4)}", flush=True)

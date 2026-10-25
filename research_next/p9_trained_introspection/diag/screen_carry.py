"""Task-search screen, candidate 2 (carry route; exploratory; CPU <= 4 threads).
Turn 1: user "Compute: {a} + {b}." assistant prefill "{a} + {b} = " then the model writes the sum.
(1) sum accuracy (greedy, 3 tokens); (2) with the operands visible, can the model answer in turn 2
"Did this addition require carrying a one from the ones place?" (A/B, both orders)? If (2) fails, no
readout of the hidden route can work."""
import os, random, re, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
from p9lib import P9, rot_scores  # noqa: E402
torch.set_num_threads(4)
m = P9(device="cpu")
rng = random.Random(3)
items = []
while len(items) < 24:
    a, b = rng.randint(10, 49), rng.randint(10, 49)
    carry = (a % 10 + b % 10) >= 10
    if sum(1 for x in items if x[2] == carry) < 12:
        items.append((a, b, carry))
prompts = [m.fresh(f"Compute: {a} + {b}.") + m._enc(f"{a} + {b} = ") for a, b, _ in items]
outs = m.generate_free(prompts, n_new=3)
ok = sum(bool(re.match(r"\s*%d\b" % (a + b), o)) for (a, b, _), o in zip(items, outs))
print(f"(1) sum accuracy {ok}/{len(items)}  e.g. {items[0][:2]} -> {outs[0]!r}", flush=True)
Q = "Did this addition require carrying a one from the ones place?"
LET = m.letter_ids[:2]
hits = 0
picks = []
with torch.no_grad():
    for (a, b, carry), o in zip(items, outs):
        t1 = m.fresh(f"Compute: {a} + {b}.") + m._enc(f"{a} + {b} = {a + b}<|im_end|>\n")
        lp = []
        for opts in (["Yes", "No"], ["No", "Yes"]):
            q = Q + "\nA) " + opts[0] + "\nB) " + opts[1] + "\nAnswer with the letter."
            ids = t1 + m._enc("<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe answer is (")
            lg = m.hf(input_ids=torch.tensor([ids], device=m.dev)).logits[0, -1].float()
            l2 = torch.log_softmax(lg[LET], -1).tolist()
            lp.append(l2 if opts[0] == "Yes" else l2[::-1])
        s_yes = (lp[0][0] + lp[1][0]) / 2
        s_no = (lp[0][1] + lp[1][1]) / 2
        pick = s_yes > s_no
        picks.append(pick)
        hits += pick == carry
print(f"(2) carry question, operands visible: {hits}/{len(items)}; says yes {sum(picks)}/{len(items)} (truth: 12 yes)", flush=True)

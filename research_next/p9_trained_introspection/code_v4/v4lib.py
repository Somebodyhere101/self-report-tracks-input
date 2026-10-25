"""Paper 9 v4 task layer (PREDICTIONS_V4.md): v3's multiplication task with a TWO-bin label.

Everything is v3's (code_v3/v3lib.py, imported read-only): items, prompts, region R (answer positions), the
measure() of each run's own second-digit log-probability lp2, swaps, adds. v4 changes only the label and the
report format:
  bin 0 = "Very sure" (lp2 >= the median of lp2 over correct TRAIN items), bin 1 = "Less sure";
  the report has two options, scored over both orders (two rotations), softmax over the letters A and B.
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code_v3"))
from v3lib import (P9, PREFILL, TAIL_END, all_items, key, load_json, log, measure, nat_row, options_text,  # noqa: E402,F401
                   paraphrase_lp2, parse, save_json, seeded, split_of_product, unit_add, unit_natural, unit_swap,
                   with_clean)

K = 2
BINS = ["Very sure", "Less sure"]  # bin 0: lp2 >= median (more confident), bin 1: below
Q_REPORT = "How sure were you of the second digit of your answer?"
Q_EXTERNAL = ("Here is a question that was put to an assistant, and the assistant's answer.\n\n"
              "Question: What is {a} * {b}?\nAnswer: {a} * {b} = {p}\n\n"
              "How sure do you think the assistant was of the second digit of its answer?")


def bin_of(lp2, med):
    return 0 if lp2 >= med else 1


def rotate2(opts, r):
    """Rotation r shows opts[(j + r) % 2] at letter j."""
    return [opts[(j + r) % 2] for j in range(2)]


def lp_ab(lp4):
    """Renormalise a log-softmax over the letters A-D to A and B only (the two offered options)."""
    a, b = lp4[0], lp4[1]
    m = max(a, b)
    z = m + math.log(math.exp(a - m) + math.exp(b - m))
    return [a - z, b - z]


def rot_scores2(lp_rot):
    """lp_rot: [2 rotations][2 letters]. Score of option k = mean over r of lp_r[(k - r) % 2]."""
    return [sum(lp_rot[r][(k - r) % 2] for r in range(2)) / 2 for k in range(2)]


def add_report(m, unit, rotations=(0, 1)):
    t = unit["tgt"]
    return dict(unit, reqs=list(unit["reqs"]) + [(t, m.tail(Q_REPORT, rotate2(BINS, r)), ("rep", r)) for r in rotations])


def external_prompt(m, k, opts):
    a, b = parse(k)
    q = Q_EXTERNAL.format(a=a, b=b, p=a * b) + "\n" + options_text(opts) + TAIL_END
    return m.fresh(q) + m._enc(PREFILL)


# ------------------------------------------------------------------------------------------- v4 engine
import torch  # noqa: E402


def pick2(scores, *tie_key):
    """The option with the higher score; an exact tie is broken by a seeded coin (v3 broke ties to option 0)."""
    if scores[0] == scores[1]:
        return seeded("v4-tie", *tie_key).randrange(2)
    return 0 if scores[0] > scores[1] else 1


class P9v4(P9):
    """v3's engine (P9, read-only) with two v4 changes:
      the LoRA rank is set per run (alpha = 2 x rank; v1-v3 used rank 8, alpha 16);
      the four letter logits are computed in fp32 (final hidden state in fp32 times the letter rows of lm_head),
      so bf16 rounding of the logits can no longer tie the options (v3 had exact ties on 49-63% of T3 rows).
    Every other logit stays as in v3 (bf16 lm_head, cast to fp32)."""
    rank = 8

    def install_lora(self, seed, r=None, alpha=None):
        r = r or self.rank
        return P9.install_lora(self, seed, r=r, alpha=alpha or 2 * r)

    def load_lora(self, state, seed=0):
        self.rank = int(next(iter(state.values()))[0].shape[0])
        P9.load_lora(self, state, seed)

    def letter_fix(self, hl, full):
        idx = torch.tensor(self.letter_ids, device=full.device)
        let = hl.float() @ self.hf.lm_head.weight[idx].float().t()
        return full.index_copy(1, idx, let)

    def _turn2_pass(self, cache, T1, sel, tails, lora=True):
        K_ = len(sel)
        mx = max(len(t) for t in tails)
        x = torch.tensor([t + [self.pad_id] * (mx - len(t)) for t in tails], device=self.dev)
        pos = (torch.arange(mx, device=self.dev) + T1)[None].expand(K_, -1)
        c2 = self._select_cache(cache, sel)
        self.lora_on(lora)
        try:
            h = self.hf.model(input_ids=x, past_key_values=c2, position_ids=pos, use_cache=True).last_hidden_state
        finally:
            self.lora_on(False)
        last = torch.tensor([len(t) - 1 for t in tails], device=self.dev)
        hl = h[torch.arange(K_, device=self.dev), last]
        return self.letter_fix(hl, self.hf.lm_head(hl).float())

    @torch.no_grad()
    def score_fresh(self, prompts, max_b=64, lora=False):
        """Single-turn letter log-probs (softmax over the four letters, fp32 letter logits) for each prompt."""
        self.lora_on(lora)
        out = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        try:
            for i in range(0, len(order), max_b):
                idx = order[i:i + max_b]
                mx = max(len(prompts[j]) for j in idx)
                x = torch.tensor([prompts[j] + [self.pad_id] * (mx - len(prompts[j])) for j in idx], device=self.dev)
                h = self.hf.model(input_ids=x, use_cache=False).last_hidden_state
                last = torch.tensor([len(prompts[j]) - 1 for j in idx], device=self.dev)
                hl = h[torch.arange(len(idx), device=self.dev), last]
                lg = self.letter_fix(hl, self.hf.lm_head(hl).float())
                lp4 = torch.log_softmax(lg[:, self.letter_ids], -1).cpu().tolist()
                am = lg.argmax(-1).cpu().tolist()
                for k, j in enumerate(idx):
                    out[j] = (lp4[k], None, am[k])
        finally:
            self.lora_on(False)
        return out


# ------------------------------------------------------------------------------------------- report formats
# Coordinator decision (2026-10-22, after Stage 0a): the primary report is IMMEDIATE, a confidence question appended to
# the assistant's answer right after its four digits; the turn-2 DELAYED report is kept as a secondary arm. Both read
# the same turn-1 run, which now ends with the four digits: region R = the four digit positions (interventions act on
# them only). The report positions are never inside R: if they were swapped, a swapped run's report would be the
# donor's own report.
Q_IMM = "How sure am I of the second digit of this answer?"


def turn1_digits(m, k):
    """v3's turn 1 without the closing <|im_end|>\n: the prompt and the teacher-forced four digits; R starts at p0."""
    a, b = parse(k)
    p = m.fresh(f"What is {a} * {b}?") + m._enc(f"{a} * {b} = ")
    d = m._enc(str(a * b))
    assert len(d) == 4, k
    return p + d, len(p)


def nat_row4(m, k):
    ids, p0 = turn1_digits(m, k)
    return {"ids": ids, "r0": p0}


def unit_natural4(m, k):
    return {"rows": [nat_row4(m, k)], "reqs": [], "tgt": 0, "kind": "nat"}


def unit_swap4(m, x, y, layers):
    rx = nat_row4(m, x)
    rx["copy"] = {l: 0 for l in layers}
    return {"rows": [nat_row4(m, y), rx], "reqs": [], "tgt": 1, "kind": "swap"}


def unit_add4(m, x, layer, vec):
    rx = nat_row4(m, x)
    rx["add"] = (layer, vec)
    return {"rows": [rx], "reqs": [], "tgt": 0, "kind": "add"}


def with_clean4(m, unit, x):
    rows = unit["rows"] + [nat_row4(m, x)]
    return dict(unit, rows=rows, clean=len(rows) - 1)


def tail_imm(m, opts):
    """Immediate report: continues the assistant's answer right after its four digits (no turn boundary)."""
    return m._enc("\n\n" + Q_IMM + "\n" + options_text(opts) + "\n" + PREFILL)


def tail_del(m, opts):
    """Delayed report: closes the answer (<|im_end|>\n), then the turn-2 question of v4 (as P9.tail)."""
    return m._enc("<|im_end|>\n") + m.tail(Q_REPORT, opts)


def add_report_fmt(m, unit, fmt, rotations=(0, 1)):
    t = unit["tgt"]
    mk = tail_imm if fmt == "imm" else tail_del
    return dict(unit, reqs=list(unit["reqs"]) + [(t, mk(m, rotate2(BINS, r)), ("rep", r)) for r in rotations])

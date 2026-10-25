"""Paper 9 v3 task layer (PREDICTIONS_V3.md): two-digit multiplication with the answer prefilled, as in paper 8.

Turn 1: user "What is {a} * {b}?"; assistant "{a} * {b} = " + the product's four digits + <|im_end|>\n
(teacher-forced; items are those whose four digits the model gets right).
Intermediate: the model's confidence in the SECOND digit of the product, i.e. its own probability of the
emitted second digit at the answer position that predicts it (the first answer-digit position, after the
input). Label: the quartile bin of log p2 (thresholds from train items), four ordered options.
Region R = the answer positions (the four digits and the closing <|im_end|>\n): interventions act on R only;
the operands are never touched. Ground truth for any (intervened) run = the bin of its own measured p2.
The v1 engine (../code/p9lib.P9: in-batch exact copy, additive op, cache-based turn 2, gated LoRA) is reused
read-only."""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
from p9lib import P9, TAIL_END, PREFILL, options_text, rotate, seeded, load_json, save_json, log, rot_scores  # noqa: E402,F401

BINS = ["Completely sure", "Very sure", "Fairly sure", "Not very sure"]  # bin 0 = highest p2
Q_REPORT = "How sure were you of the second digit of your answer?"
Q_EXTERNAL = ("Here is a question that was put to an assistant, and the assistant's answer.\n\n"
              "Question: What is {a} * {b}?\nAnswer: {a} * {b} = {p}\n\n"
              "How sure do you think the assistant was of the second digit of its answer?")
SPLITS = ["train", "test", "pilot", "train", "test"]


def all_items():
    """Every a * b with 11 <= a <= b <= 99 and a four-digit product."""
    return [(a, b) for a in range(11, 100) for b in range(a, 100) if 1000 <= a * b <= 9999]


def key(a, b):
    return f"{a}*{b}"


def parse(k):
    a, b = k.split("*")
    return int(a), int(b)


def split_of_product():
    """Products (not items) are split, so every factorisation of a product lands in the same split."""
    prods = sorted({a * b for a, b in all_items()})
    rng = seeded("v3-split")
    order = prods[:]
    rng.shuffle(order)
    return {p: SPLITS[i % len(SPLITS)] for i, p in enumerate(order)}


def turn1(m, k):
    a, b = parse(k)
    p = m.fresh(f"What is {a} * {b}?") + m._enc(f"{a} * {b} = ")
    d = m._enc(str(a * b))
    assert len(d) == 4, k
    return p + d + m._enc("<|im_end|>\n"), len(p)


def paraphrase_prompt(m, k):
    """Input-only predictor (the 1-hop analogue): the same product asked in a different chat format."""
    a, b = parse(k)
    return m.fresh(f"Multiply {a} by {b}.") + m._enc(f"{a} x {b} = ")


def bin_of(lp2, th):
    """th = 3 ascending thresholds on log p2 (quartile cut points); bin 0 = most sure."""
    if lp2 >= th[2]:
        return 0
    if lp2 >= th[1]:
        return 1
    if lp2 >= th[0]:
        return 2
    return 3


# ---------------------------------------------------------------------------- rows and units
def nat_row(m, k):
    ids, p0 = turn1(m, k)
    return {"ids": ids, "r0": p0}


def unit_natural(m, k):
    return {"rows": [nat_row(m, k)], "reqs": [], "tgt": 0, "kind": "nat"}


def unit_swap(m, x, y, layers):
    rx = nat_row(m, x)
    rx["copy"] = {l: 0 for l in layers}
    return {"rows": [nat_row(m, y), rx], "reqs": [], "tgt": 1, "kind": "swap"}


def unit_add(m, x, layer, vec):
    rx = nat_row(m, x)
    rx["add"] = (layer, vec)
    return {"rows": [rx], "reqs": [], "tgt": 0, "kind": "add"}


def add_report(m, unit, rotations=(0, 1, 2, 3)):
    t = unit["tgt"]
    return dict(unit, reqs=list(unit["reqs"]) + [(t, m.tail(Q_REPORT, rotate(BINS, r)), ("rep", r)) for r in rotations])


def add_mc(m, unit, question, opts, rotations=(0, 1, 2, 3)):
    t = unit["tgt"]
    return dict(unit, reqs=list(unit["reqs"]) + [(t, m.tail(question, rotate(opts, r)), ("mc", r)) for r in rotations])


def external_prompt(m, k, opts):
    a, b = parse(k)
    q = Q_EXTERNAL.format(a=a, b=b, p=a * b) + "\n" + options_text(opts) + TAIL_END
    return m.fresh(q) + m._enc(PREFILL)


@torch.no_grad()
def measure(m, units, max_rows=48):
    """For each unit's target row: log p of each emitted digit, entropy at the second digit, all four argmax
    correct, and the KL (summed over R) from the unit's clean row if the unit has one (key 'clean')."""
    out = [None] * len(units)
    groups = {}
    for i, u in enumerate(units):
        groups.setdefault(m._key(u), []).append(i)
    for g, idx in groups.items():
        batch, n = [], 0
        for i in idx + [None]:
            if i is not None and (not batch or n + len(units[i]["rows"]) <= max_rows):
                batch.append(i)
                n += len(units[i]["rows"])
                continue
            rows, _, _ = m._flatten(units, batch)
            _, h, _ = m._turn1_pass(rows)
            off = 0
            for j in batch:
                u = units[j]
                t = off + u["tgt"]
                p0 = rows[t]["r0"]
                lg = torch.log_softmax(m.hf.lm_head(h[t, p0 - 1:p0 + 3]).float(), -1)
                tg = torch.tensor(rows[t]["ids"][p0:p0 + 4], device=m.dev)
                lp = lg.gather(-1, tg[:, None])[:, 0]
                ent2 = float(-(lg[1].exp() * lg[1]).sum())
                res = {"lp": lp.tolist(), "ent2": ent2, "correct": bool((lg.argmax(-1) == tg).all())}
                if "clean" in u:
                    c = off + u["clean"]
                    a_ = torch.log_softmax(m.hf.lm_head(h[c, p0:]).float(), -1)
                    b_ = torch.log_softmax(m.hf.lm_head(h[t, p0:]).float(), -1)
                    res["kl"] = float((a_.exp() * (a_ - b_)).sum())
                out[j] = res
                off += len(u["rows"])
            del h
            batch, n = ([i], len(units[i]["rows"])) if i is not None else ([], 0)
    return out


def with_clean(m, unit, x):
    """Append the recipient's clean row and point 'clean' at it (for KL and retention)."""
    rows = unit["rows"] + [nat_row(m, x)]
    return dict(unit, rows=rows, clean=len(rows) - 1)


@torch.no_grad()
def paraphrase_lp2(m, keys, max_b=64):
    """log p of the true second digit after the paraphrase prompt (the first digit teacher-forced)."""
    out = {}
    for i in range(0, len(keys), max_b):
        ks = keys[i:i + max_b]
        seqs = []
        for k in ks:
            a, b = parse(k)
            seqs.append(paraphrase_prompt(m, k) + m._enc(str(a * b))[:1])
        mlen = max(len(s) for s in seqs)
        x = torch.tensor([s + [m.pad_id] * (mlen - len(s)) for s in seqs], device=m.dev)
        h = m.hf.model(input_ids=x, use_cache=False).last_hidden_state
        for j, (k, s) in enumerate(zip(ks, seqs)):
            a, b = parse(k)
            lg = torch.log_softmax(m.hf.lm_head(h[j, len(s) - 1]).float(), -1)
            out[k] = float(lg[m._enc(str(a * b))[1]])
    return out

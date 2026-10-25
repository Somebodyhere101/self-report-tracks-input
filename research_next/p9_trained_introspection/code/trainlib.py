"""Paper 9: reporter training (same budget for every arm) and report evaluation helpers.

Every arm: rank-8 LoRA (alpha 16) on the seven linear modules of all 28 blocks (s2lib.LoRALinear), Adam,
linear warm-up over 10 steps, gradient clipping at 1.0, batch 8, the same number of steps and examples.
In-place arms (O, I, J): the adapter is on only in turn 2. Arm X (external observer): one fresh chat
containing the transcript as text, adapter on throughout.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p9lib import log, rotate, rot_scores, seeded  # noqa: E402
from build import external_prompt  # noqa: E402
from items import example_unit  # noqa: E402


def fresh_loss(m, prompts, targets):
    mlen = max(len(p) for p in prompts)
    x = torch.tensor([p + [m.pad_id] * (mlen - len(p)) for p in prompts], device=m.dev)
    m.lora_on(True)
    try:
        h = m.hf.model(input_ids=x, use_cache=False).last_hidden_state
    finally:
        m.lora_on(False)
    last = torch.tensor([len(p) - 1 for p in prompts], device=m.dev)
    lg = m.hf.lm_head(h[torch.arange(len(prompts), device=m.dev), last]).float()
    tgt = torch.tensor([m.letter_ids[t] for t in targets], device=m.dev)
    return F.cross_entropy(lg, tgt, reduction="sum"), len(prompts)


def train(m, arm, examples, cfg, seed, jvec=None, steps=None, log_every=25):
    steps = steps or cfg["steps"]
    bs, lr = cfg["bs"], cfg["lr"]
    m.install_lora(seed=seed)
    params = [p for _, _, mod in m.lora for p in (mod.A, mod.B)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    rng = seeded("trainorder", arm, seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    ptr, hist, t0 = 0, [], time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 10)
        if ptr + bs > len(order):
            rng.shuffle(order)
            ptr = 0
        batch = [examples[i] for i in order[ptr:ptr + bs]]
        ptr += bs
        rots = [rng.randrange(4) for _ in batch]
        tg = [rotate(ex["cands"], r).index(ex["label"]) for ex, r in zip(batch, rots)]
        if arm == "X":
            prompts = [external_prompt(m, ex["x"], ex["cont"], rotate(ex["cands"], r)) for ex, r in zip(batch, rots)]
            loss, n = fresh_loss(m, prompts, tg)
        else:
            units = [example_unit(m, ex, cfg, jvec, rotations=(r,)) for ex, r in zip(batch, rots)]
            loss, n = m.train_step_loss(units, tg)
        loss = loss / n
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(loss.detach())})
            log(f"{arm} seed {seed} step {step} loss {float(loss.detach()):.4f} ({(time.time() - t0) / 60:.1f} min)")
    for p in params:
        p.requires_grad_(False)
    return m.lora_state(), hist


@torch.no_grad()
def eval_reports(m, examples, cfg, jvec, lora, max_rows=48):
    """Rotation-averaged report scores for each example (4 rotations). Returns list of dicts."""
    units = [example_unit(m, ex, cfg, jvec) for ex in examples]
    res = m.run_units(units, lora=lora, max_rows=max_rows)
    out = []
    for ex, r in zip(examples, res):
        lp = [r[k][0] for k in range(4)]
        s = rot_scores(lp)
        out.append({"scores": s, "pick": ex["cands"][max(range(4), key=lambda i: s[i])], "lp": lp})
    return out


@torch.no_grad()
def eval_external(m, examples, lora=True):
    prompts = []
    for ex in examples:
        prompts += [external_prompt(m, ex["x"], ex["cont"], rotate(ex["cands"], r)) for r in range(4)]
    res = m.score_fresh(prompts, lora=lora)
    out = []
    for k, ex in enumerate(examples):
        lp = [res[4 * k + r][0] for r in range(4)]
        s = rot_scores(lp)
        out.append({"scores": s, "pick": ex["cands"][max(range(4), key=lambda i: s[i])], "lp": lp})
    return out

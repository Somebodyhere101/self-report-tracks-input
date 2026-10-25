"""Paper 9 v3 reporter training and evaluation (same budget and loop as v1's trainlib; labels are bin indices
0..3 over the fixed option list BINS, shown in four cyclic rotations)."""
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v3lib import BINS, add_report, external_prompt, log, rot_scores, rotate, seeded, unit_add, unit_natural, unit_swap  # noqa: E402
from items_v3 import random_unit  # noqa: E402


def example_unit(m, ex, cfg, jvec, rotations=(0, 1, 2, 3)):
    x = ex["x"]
    if ex["arm"] in ("O", "X", "N", "nat"):
        u = unit_natural(m, x)
    elif ex["arm"] in ("I", "swap"):
        u = unit_swap(m, x, ex["y"], ex.get("layers") or list(range(cfg["l1"], m.L)))
    elif ex["arm"] in ("J", "add"):
        if "rand" in ex:
            vec = random_unit(m.d, ex["rand"]) * (ex["scale"] * cfg["base_norm"])
        else:
            vec = jvec[ex["c"]] * (ex.get("scale", cfg["s_add"]) * cfg["base_norm"])
        u = unit_add(m, x, ex["layer"] if "layer" in ex else cfg["la"], vec)
    else:
        raise ValueError(ex["arm"])
    return add_report(m, u, rotations)


def i_examples(its, by_p, binf, n, rng, para_bin=None):
    """Alternate different-bin and same-bin donors (same product). Recipients are drawn from those that have a
    donor of the needed kind. A different-bin donor's bin also differs from the recipient's paraphrase bin
    (the input-implied bin), when para_bin is given."""
    pools = []
    for x in its:
        if x["product"] not in by_p:
            continue
        others = [y for y in by_p[x["product"]] if y["key"] != x["key"]]
        dd = [y for y in others if binf(y) != binf(x) and (para_bin is None or binf(y) != para_bin(x))]
        ss = [y for y in others if binf(y) == binf(x)]
        pools.append((x, dd, ss))
    xs = {0: [p for p in pools if p[1]], 1: [p for p in pools if p[2]]}
    assert xs[0], "no recipients with a different-bin donor"
    out = []
    for t in range(n):
        k = t % 2 if xs[1] else 0
        x, dd, ss = xs[k][rng.randrange(len(xs[k]))]
        y = rng.choice(dd if k == 0 else ss)
        out.append({"arm": "I", "x": x["key"], "y": y["key"], "label": binf(y), "kind": ("diff", "same")[k]})
    return out


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
    return F.cross_entropy(lg, torch.tensor([m.letter_ids[t] for t in targets], device=m.dev), reduction="sum"), len(prompts)


def train(m, arm, examples, cfg, seed, jvec=None, steps=None, log_every=25):
    steps = steps or cfg["steps"]
    bs, lr = cfg["bs"], cfg["lr"]
    if m.lora:
        m.remove_lora()
    m.install_lora(seed=seed)
    params = [p for _, _, mod in m.lora for p in (mod.A, mod.B)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    rng = seeded("v3-trainorder", arm, seed)
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
        tg = [(ex["label"] - r) % 4 for ex, r in zip(batch, rots)]  # rotation r shows BINS[(j + r) % 4] at letter j
        if arm == "X":
            loss, n = fresh_loss(m, [external_prompt(m, ex["x"], rotate(BINS, r)) for ex, r in zip(batch, rots)], tg)
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
    units = [example_unit(m, ex, cfg, jvec) for ex in examples]
    res = m.run_units(units, lora=lora, max_rows=max_rows)
    out = []
    for r in res:
        lp = [r[k][0] for k in range(4)]
        s = rot_scores(lp)
        out.append({"scores": s, "pick": max(range(4), key=lambda i: s[i]), "lp": lp})
    return out


@torch.no_grad()
def eval_external(m, examples, lora=True):
    prompts = []
    for ex in examples:
        prompts += [external_prompt(m, ex["x"], rotate(BINS, r)) for r in range(4)]
    res = m.score_fresh(prompts, lora=lora)
    out = []
    for k in range(len(examples)):
        lp = [res[4 * k + r][0] for r in range(4)]
        s = rot_scores(lp)
        out.append({"scores": s, "pick": max(range(4), key=lambda i: s[i]), "lp": lp})
    return out

"""Paper 9 v4 reporter training and evaluation. As v3 (trainlib_v3: Adam, 10-step linear warm-up then a constant lr,
gradient clip 1.0, batch 8, one random option order per example, cross-entropy over the full vocabulary), with v4's
changes:
  two options (v4lib.BINS) in two orders; target letter (label - r) % 2 for order r;
  LoRA rank and lr per budget rung (alpha = 2 x rank);
  the examples are a fixed stream read in order (step t uses examples[bs*t : bs*(t+1)]) and the option order of each
  example comes from seeded('v4-rot', arm, seed, t), so a run of S steps is an exact prefix of a longer run;
  a checkpoint (adapter, optimiser state, step, history) every 200 steps and at the end; when the GPU-lock deadline
  is near, the loop saves and returns 'yield' (the caller exits 3 and the job is re-queued).
Evaluation scores both orders and averages them (rotation-averaged), picks with a seeded tie-break."""
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v4lib import (BINS, add_report, add_report_fmt, external_prompt, log, lp_ab, pick2, rot_scores2, rotate2, seeded,  # noqa: E402
                   unit_add, unit_add4, unit_natural, unit_natural4, unit_swap, unit_swap4)
from items_v3 import random_unit  # noqa: E402  (v3, frozen)
from trainlib_v3 import i_examples  # noqa: E402,F401  (v3, frozen)

CKPT_EVERY = 200
MARGIN_S = 240


def example_unit(m, ex, cfg, jvec, rotations=(0, 1)):
    """cfg['fmt'] = 'imm' or 'del': turn 1 ends with the four digits (R = the digits) and the report is the immediate
    or the delayed tail; no 'fmt': v3's turn 1 (R = digits + <|im_end|>\\n) and the turn-2 report."""
    x, fmt = ex["x"], cfg.get("fmt")
    nat, swap, add = (unit_natural4, unit_swap4, unit_add4) if fmt else (unit_natural, unit_swap, unit_add)
    if ex["arm"] in ("O", "X", "N", "nat"):
        u = nat(m, x)
    elif ex["arm"] in ("I", "swap"):
        u = swap(m, x, ex["y"], ex.get("layers") or list(range(cfg["l1"], m.L)))
    elif ex["arm"] in ("J", "add"):
        if "rand" in ex:
            vec = random_unit(m.d, ex["rand"]) * (ex["scale"] * cfg["base_norm"])
        else:
            vec = jvec[ex["c"]] * (ex.get("scale", cfg["s_add"]) * cfg["base_norm"])
        u = add(m, x, ex["layer"] if "layer" in ex else cfg["la"], vec)
    else:
        raise ValueError(ex["arm"])
    return add_report_fmt(m, u, fmt, rotations) if fmt else add_report(m, u, rotations)


def fresh_loss(m, prompts, targets):
    mlen = max(len(p) for p in prompts)
    x = torch.tensor([p + [m.pad_id] * (mlen - len(p)) for p in prompts], device=m.dev)
    m.lora_on(True)
    try:
        h = m.hf.model(input_ids=x, use_cache=False).last_hidden_state
    finally:
        m.lora_on(False)
    last = torch.tensor([len(p) - 1 for p in prompts], device=m.dev)
    hl = h[torch.arange(len(prompts), device=m.dev), last]
    lg = m.letter_fix(hl, m.hf.lm_head(hl).float())
    return F.cross_entropy(lg, torch.tensor([m.letter_ids[t] for t in targets], device=m.dev), reduction="sum"), len(prompts)


def _save(path, m, opt, step, hist, lr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"lora": m.lora_state(), "opt": opt.state_dict(), "step": step, "hist": hist, "rank": m.rank, "lr": lr},
               path + ".tmp")
    os.replace(path + ".tmp", path)


def train(m, arm, examples, cfg, seed, jvec, until, ckpt_path, deadline=None, log_every=50, stream_key=None):
    """Train arm's adapter (or continue it from ckpt_path) up to step `until`. Leaves the adapter installed.
    Option orders are seeded by (stream_key or arm, seed, step). Returns ('done', hist) or ('yield', hist); in both
    cases the checkpoint holds the current state."""
    bs, lr = cfg["bs"], cfg["lr"]
    assert len(examples) >= until * bs, (len(examples), until, bs)
    m.rank = cfg["rank"]
    if m.lora:
        m.remove_lora()
    m.install_lora(seed=seed)
    params = [p for _, _, mod in m.lora for p in (mod.A, mod.B)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    step, hist = 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path)
        assert ck["rank"] == cfg["rank"] and ck.get("lr") == lr and ck["step"] <= until, \
            (ckpt_path, ck["rank"], ck.get("lr"), ck["step"], cfg["rank"], lr, until)
        for b, p, mod in m.lora:
            A, B = ck["lora"][f"{b}.{p}"]
            mod.A.data.copy_(A.to(mod.A.device))
            mod.B.data.copy_(B.to(mod.B.device))
        opt.load_state_dict(ck["opt"])
        step, hist = ck["step"], ck["hist"]
        log(f"{arm} seed {seed}: resumed at step {step} from {ckpt_path}")
    t0 = time.time()
    status = "done"
    while step < until:
        if deadline and time.time() > deadline - MARGIN_S:
            status = "yield"
            break
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 10)
        batch = examples[step * bs:(step + 1) * bs]
        rr = seeded("v4-rot", stream_key or arm, seed, step)
        rots = [rr.randrange(2) for _ in batch]
        tg = [(ex["label"] - r) % 2 for ex, r in zip(batch, rots)]  # order r shows BINS[(j + r) % 2] at letter j
        if arm == "X":
            loss, n = fresh_loss(m, [external_prompt(m, ex["x"], rotate2(BINS, r)) for ex, r in zip(batch, rots)], tg)
        else:
            units = [example_unit(m, ex, cfg, jvec, rotations=(r,)) for ex, r in zip(batch, rots)]
            loss, n = m.train_step_loss(units, tg)
        loss = loss / n
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % log_every == 0 or step == until - 1:
            hist.append({"step": step, "loss": float(loss.detach())})
            log(f"{arm} seed {seed} r{cfg['rank']} lr {lr:g} step {step} loss {float(loss.detach()):.4f} "
                f"({(time.time() - t0) / 60:.1f} min)")
        step += 1
        if step % CKPT_EVERY == 0 and step < until:
            _save(ckpt_path, m, opt, step, hist, lr)
    _save(ckpt_path, m, opt, step, hist, lr)
    for p in params:
        p.requires_grad_(False)
    return status, hist


@torch.no_grad()
def eval_reports(m, examples, cfg, jvec, lora, tie_key, max_rows=48):
    units = [example_unit(m, ex, cfg, jvec) for ex in examples]
    res = m.run_units(units, lora=lora, max_rows=max_rows)
    out = []
    for k, r in enumerate(res):
        lp = [lp_ab(r[j][0]) for j in range(2)]
        s = rot_scores2(lp)
        out.append({"scores": s, "pick": pick2(s, *tie_key, k), "lp": lp})
    return out


@torch.no_grad()
def eval_external(m, examples, lora, tie_key):
    prompts = []
    for ex in examples:
        prompts += [external_prompt(m, ex["x"], rotate2(BINS, r)) for r in range(2)]
    res = m.score_fresh(prompts, lora=lora)
    out = []
    for k in range(len(examples)):
        lp = [lp_ab(res[2 * k + r][0]) for r in range(2)]
        s = rot_scores2(lp)
        out.append({"scores": s, "pick": pick2(s, *tie_key, k), "lp": lp})
    return out

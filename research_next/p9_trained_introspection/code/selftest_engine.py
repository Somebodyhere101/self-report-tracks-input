"""Paper 9 engine self-test (no analysis data). Checks, on a given model:
 1. a swap at every layer makes the recipient's R states equal the donor's at every layer (bit-exact);
 2. a placeholder readout row reproduces the intervened row's R states at every layer for natural, swap
    (all-layer and single-layer) and add sources (bit-exact);
 3. turn 2 on the selected turn-1 cache gives the same letter log-probs as one full pass over turn 1 + turn 2
    (natural rows; tolerance for kernel differences);
 4. an installed LoRA with non-zero weights leaves turn 1 unchanged and changes turn 2; disabling it restores turn 2;
 5. one training step lowers the loss on a repeated batch;
 6. rot_scores maps rotations back to candidates.
Usage: python selftest_engine.py [--tiny] [--device cpu|cuda]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9lib  # noqa: E402
import p6lib  # noqa: E402  (paper 8 copy of paper 6's engine, on sys.path via p9lib)
from build import (unit_natural, unit_swap, unit_add, add_readout, add_report, fill)  # noqa: E402
from p9lib import P9, rot_scores, rotate  # noqa: E402


def r_states(m, rows, layers):
    _, _, cap = m._turn1_pass(rows, capture_layers=layers)
    r0 = rows[0]["r0"]
    return {l: cap[l][:, r0:].clone() for l in layers}


MARKER = os.path.join(os.path.dirname(HERE), "res_pilot", "ENGINE_SELFTEST_PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if not a.tiny and os.path.exists(MARKER):
        print("engine self-test already passed on this model; skipping")
        return
    key = "q3-1.7b"
    if a.tiny:
        p6lib.MODELS["tiny"] = "HuggingFaceTB/SmolLM2-135M-Instruct"
        key = "tiny"
    torch.manual_seed(0)
    m = P9(device=a.device, key=key)
    L = m.L
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, detail, flush=True)

    # cities from the pilot split only (never used in the analysis); same token count and continent
    cont = "Europe"
    pairs = [("Venice", "Porto"), ("Nancy", "Basel"), ("Liverpool", "Venice"), ("Bristol", "Nancy"), ("Porto", "Basel")]
    x, y = next((a_, b_) for a_, b_ in pairs if len(m._enc(" " + a_)) == len(m._enc(" " + b_)))
    print("pair", x, y, len(m._enc(" " + x)))
    allL = list(range(L))
    with torch.no_grad():
        # 1. all-layer swap
        u = unit_swap(m, x, y, cont, allL)
        st = r_states(m, u["rows"], allL)
        check("swap all layers: recipient R == donor R at every layer",
              all(torch.equal(st[l][1], st[l][0]) for l in allL))
        nat = r_states(m, [unit_natural(m, x, cont)["rows"][0]], allL)
        check("swap changes R (differs from natural recipient)", not torch.equal(nat[L - 1][0], st[L - 1][1]))
        # 2. readouts reproduce the intervened row
        for name, uu in [("natural", unit_natural(m, x, cont)), ("swap all", unit_swap(m, x, y, cont, allL)),
                         ("swap one", unit_swap(m, x, y, cont, [L // 2])),
                         ("swap from mid", unit_swap(m, x, y, cont, list(range(L // 2, L)))),
                         ("add", unit_add(m, x, cont, L // 2, torch.randn(m.d) * 4))]:
            ur = add_readout(m, uu, x, cont, fill(["France"], "t"))
            st = r_states(m, ur["rows"], allL)
            t, pidx = ur["tgt"], len(ur["rows"]) - 1
            same = all(torch.equal(st[l][pidx], st[l][t]) for l in allL)
            check(f"readout reproduces R ({name})", same)
        # 3. cache turn 2 == full pass. In float32 (tiny model) the match must be exact up to 1e-3; in bf16
        # the two kernel paths differ by rounding (0.18 nats seen on the tiny model on CPU), so only a bound.
        cands = fill(["France"], "t3")
        uu = add_report(m, unit_natural(m, x, cont), cands)

        def cache_vs_full():
            res = m.run_units([uu])[0]
            full = []
            for r in range(4):
                ids = uu["rows"][0]["ids"] + m.tail(p9lib.Q_REPORT, rotate(cands, r))
                full.append(m.score_fresh([ids])[0][0])
            return res, max(abs(a_ - b_) for r in range(4) for a_, b_ in zip(res[r][0], full[r]))
        res, diff = cache_vs_full()
        # exactness of this code path is checked in float32 on the tiny model; in bf16 only gross errors fail
        check("turn 2 from cache close to one full pass (bf16 rounding bound 1.0)", diff < 1.0, f"max |dlogp| = {diff:.4f}")
        # 4. LoRA gating
        m.install_lora(seed=0)
        for _, _, mod in m.lora:
            mod.B.data.normal_(0, 0.02)
        st_l = r_states(m, uu["rows"], [L - 1])
        st_0 = nat[L - 1]
        check("LoRA installed: turn 1 unchanged", torch.equal(st_l[L - 1][0], st_0[0]))
        res_on = m.run_units([uu], lora=True)[0]
        res_off = m.run_units([uu], lora=False)[0]
        d_on = max(abs(a_ - b_) for r in range(4) for a_, b_ in zip(res_on[r][0], res[r][0]))
        d_off = max(abs(a_ - b_) for r in range(4) for a_, b_ in zip(res_off[r][0], res[r][0]))
        check("LoRA on changes turn 2", d_on > 1e-3, f"{d_on:.4f}")
        check("LoRA off restores turn 2", d_off < 1e-6, f"{d_off:.2e}")
    # 5. a training step lowers the loss
    for _, _, mod in m.lora:
        mod.B.data.zero_()
    params = [p for _, _, mod in m.lora for p in (mod.A, mod.B)]
    for p_ in params:
        p_.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=1e-3)
    units = []
    targets = []
    for i in range(3):
        cs = fill(["Portugal", "Italy"], f"tr{i}")
        u2 = add_report(m, unit_swap(m, x, y, cont, list(range(L // 2, L))), cs, rotations=(i % 4,))
        units.append(u2)
        targets.append(rotate(cs, i % 4).index("Portugal"))
    losses = []
    for step in range(6):
        loss, n = m.train_step_loss(units, targets)
        loss = loss / n
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    check("training lowers the loss", losses[-1] < losses[0], " ".join(f"{v:.3f}" for v in losses))
    m.remove_lora()
    if a.tiny:  # last, because the dtype round trip also casts non-weight buffers (rotary frequencies)
        with torch.no_grad():
            m.hf.to(torch.float32)
            _, d32 = cache_vs_full()
        check("turn 2 from cache matches one full pass (float32)", d32 < 1e-3, f"max |dlogp| = {d32:.2e}")
    # 6. rotation scoring
    lp = [[0.0] * 4 for _ in range(4)]
    for r in range(4):
        lp[r][(2 - r) % 4] = 5.0  # candidate 2 is always chosen
    s = rot_scores(lp)
    check("rot_scores picks the consistent candidate", max(range(4), key=lambda k: s[k]) == 2)
    print("ALL PASS" if ok else "SOME FAILED")
    if ok and not a.tiny:
        os.makedirs(os.path.dirname(MARKER), exist_ok=True)
        open(MARKER, "w").write("passed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

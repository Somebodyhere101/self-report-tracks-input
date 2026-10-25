"""Paper 8 patching arm: whole-state interchange on top of paper 8's engine (p8lib.P8Model, unchanged).

A patch replaces the output of one block, at a per-row set of PROMPT positions, with a donor's block
output at the same positions (prompts have equal token counts, so positions align). It applies in the
pass that processes the prompt (step 0 of generation, or the teacher-forced pass); generated positions
are never patched, they are computed from the patched prompt states through the KV cache.

Rows of one batch may carry different prompts of the same length (no padding), so donors' states and
clean outcomes for many items can be computed in one pass.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from p8lib import P8Model  # noqa: E402


class PatchModel(P8Model):
    def __init__(self, *a, **kw):
        self.patch = None  # {layer: (mask [B, n] bool, donor [B, n, d] bf16)} while the prompt pass runs
        super().__init__(*a, **kw)

    def _hook(self, i):
        p8f = P8Model._hook(self, i)  # paper 8's add/mask/capture hook, applied after the patch

        def f(mod, inp, out):
            pt = self.patch
            if pt is not None and i in pt:
                h = out[0] if isinstance(out, tuple) else out
                mask, donor = pt[i]
                n = mask.shape[1]
                assert h.shape[1] >= n, "patch applied outside the prompt pass"
                hp = torch.where(mask[..., None], donor.to(h.dtype), h[:, :n])
                h = torch.cat([hp, h[:, n:]], 1) if h.shape[1] > n else hp
                out = (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return p8f(mod, inp, out)
        return f

    def _set_patch(self, patch):
        self.patch = None if not patch else {l: (m.to(self.dev, torch.bool), d.to(self.dev, torch.bfloat16))
                                             for l, (m, d) in patch.items()}

    @torch.no_grad()
    def states(self, ids_rows, layers):
        """Block outputs of `layers` at every position, one row per prompt (equal lengths), bf16 on GPU."""
        x = torch.tensor(ids_rows, device=self.dev)
        self.capture = {l: None for l in layers}
        try:
            self.hf.model(input_ids=x, use_cache=False)
            return {l: self.capture[l].detach().clone() for l in layers}
        finally:
            self.capture = None

    @torch.no_grad()
    def gen(self, ids_rows, patch, n_new, lens=True):
        """Greedy decoding of B rows (equal-length prompts) with a prompt-pass patch; the loop of
        p6lib.Model.generate / p8lib.P8Model.gen8. Returns tokens [B, steps], first-step log-probs and,
        with lens, the logit-lens top-1 [B, steps, L]."""
        B = len(ids_rows)
        x = torch.tensor(ids_rows, device=self.dev)
        T = x.shape[1]
        cache = transformers.DynamicCache()
        out_tok, top1s, lp0 = [], [], None
        done = torch.zeros(B, dtype=torch.bool, device=self.dev)
        eos_t = torch.tensor(sorted(self.eos), device=self.dev)
        self._set_patch(patch)
        try:
            for step in range(n_new):
                if lens:
                    self.capture = {l: None for l in range(self.L)}
                p0 = 0 if step == 0 else T - 1
                pos = torch.arange(p0, p0 + x.shape[1], device=self.dev)[None].expand(B, -1)
                o = self.hf(input_ids=x, past_key_values=cache, use_cache=True, position_ids=pos, logits_to_keep=1)
                self.patch = None  # the KV cache now holds the patched prompt
                logits = o.logits[:, -1].float()
                nxt = logits.argmax(-1)
                if step == 0:
                    lp0 = torch.log_softmax(logits, -1)
                if lens:
                    top1s.append(self._lens_top1())
                nxt = torch.where(done, eos_t[0], nxt)
                out_tok.append(nxt)
                done |= torch.isin(nxt, eos_t)
                x = nxt[:, None]
                T += 1
                if bool(done.all()):
                    break
        finally:
            self.patch, self.capture = None, None
        del cache
        res = dict(tokens=torch.stack(out_tok, 1).cpu(), lp0=lp0.cpu())
        if lens:
            res["top1"] = torch.stack(top1s, 1).cpu()
        return res

    @torch.no_grad()
    def cont_lp(self, ids_rows, patch, conts, cap_layers=None):
        """Teacher-forced log-probability of each row's continuation conts[b] (list of token ids) after
        its prompt, with the prompt-pass patch. Rows are right-padded (causal attention: padding after a
        token does not affect it). Returns [B] summed log-probs and, with cap_layers, the block outputs
        of those layers at the prompt positions."""
        n = len(ids_rows[0])
        m = max(len(c) for c in conts)
        pad = sorted(self.eos)[0]
        x = torch.tensor([r + c + [pad] * (m - len(c)) for r, c in zip(ids_rows, conts)], device=self.dev)
        self._set_patch(patch)
        self.capture = {l: None for l in cap_layers} if cap_layers else None
        try:
            h = self.hf.model(input_ids=x, use_cache=False).last_hidden_state
            cap = {l: v[:, :n].float() for l, v in self.capture.items()} if cap_layers else None
        finally:
            self.patch, self.capture = None, None
        lp = torch.log_softmax(self.hf.lm_head(h[:, n - 1:n - 1 + m]).float(), -1)  # [B, m, V]
        tgt = x[:, n:n + m]
        g = lp.gather(-1, tgt[..., None])[..., 0]
        keep = torch.tensor([[j < len(c) for j in range(m)] for c in conts], device=self.dev)
        out = (g * keep).sum(1).cpu().tolist()
        del h, lp
        return (out, cap) if cap_layers else out

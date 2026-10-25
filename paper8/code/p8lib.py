"""Paper 8 engine: paper 6's Model (p6lib.py, copied unchanged) with pushes restricted to a per-row set
of token positions, so that one batch can hold rows pushed at the reading position onward, at every
prompt position, or at the source tokens only.

A row's push mask is `pmask[b, t]` for prompt positions t < n_prompt and `gmask[b]` for every
generated (or teacher-forced continuation) position. Paper 6's protocol is pmask = (t >= start),
gmask = 1, which reproduces p6lib.Model.generate exactly (checked by run8.py selftest).

The teacher-forced pass also captures the hidden states (block outputs) of selected layers at selected
positions, for the persistence measure.
"""

from __future__ import annotations

import torch
import transformers

from p6lib import Model


class P8Model(Model):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cap_pos = None

    def _hook(self, i):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            o = self.ops
            if o is not None:
                add = o["add"].get(i)
                if add is not None:
                    T = h.shape[1]
                    pos = o["pos0"] + torch.arange(T, device=h.device)
                    n = o["n_prompt"]
                    inside = (pos < n)[None, :]
                    pm = o["pmask"][:, pos.clamp(max=n - 1)]
                    m = torch.where(inside, pm, o["gmask"][:, None].expand_as(pm))
                    h = h + m.to(h.dtype)[..., None] * add[:, None, :]
            if self.capture is not None and i in self.capture:
                cp = getattr(self, "cap_pos", None)
                self.capture[i] = (h[:, cp] if cp is not None else h).detach()
            if isinstance(out, tuple):
                return (h,) + tuple(out[1:])
            return h
        return f

    def _ops(self, n_prompt, add, pmask, gmask):
        return dict(add={l: v.to(self.dev, torch.bfloat16) for l, v in add.items()},
                    pmask=pmask.to(self.dev, torch.float32), gmask=gmask.to(self.dev, torch.float32),
                    n_prompt=n_prompt, pos0=0)

    @torch.no_grad()
    def gen8(self, ids, add, pmask, gmask, n_new, lens=False, stop_all_eos=True):
        """p6lib.Model.generate with position masks in place of `start` (same loop otherwise)."""
        B = pmask.shape[0]
        x = torch.tensor([ids] * B, device=self.dev)
        T = len(ids)
        cache = transformers.DynamicCache()
        out_tok, top1s, lp0 = [], [], None
        done = torch.zeros(B, dtype=torch.bool, device=self.dev)
        eos_t = torch.tensor(sorted(self.eos), device=self.dev)
        self.ops = self._ops(len(ids), add, pmask, gmask)
        try:
            for step in range(n_new):
                if lens:
                    self.capture = {l: None for l in range(self.L)}
                p0 = 0 if step == 0 else T - 1
                self.ops["pos0"] = p0
                pos = torch.arange(p0, p0 + x.shape[1], device=self.dev)[None].expand(B, -1)
                o = self.hf(input_ids=x, past_key_values=cache, use_cache=True, position_ids=pos, logits_to_keep=1)
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
                if stop_all_eos and bool(done.all()):
                    break
        finally:
            self.ops, self.capture = None, None
        del cache
        res = dict(tokens=torch.stack(out_tok, 1).cpu(), lp0=lp0)
        torch.cuda.empty_cache()
        if lens:
            res["top1"] = torch.stack(top1s, 1).cpu()
        return res

    @torch.no_grad()
    def tf8(self, ids, start, add, pmask, gmask, cont, cap_layers, cap_pos):
        """Teacher-forced on ids + cont (the clean row's continuation). KL(row 0 || row b) summed over
        positions start .. end, as p6lib.Model.tf_kl's kl_all (the output window is the same for every
        push condition). Also returns the block outputs of `cap_layers` at absolute positions `cap_pos`
        as {layer: [B, len(cap_pos), d] float32 on the GPU}."""
        full = ids + cont
        B = pmask.shape[0]
        x = torch.tensor([full] * B, device=self.dev)
        self.ops = self._ops(len(ids), add, pmask, gmask)
        self.capture = {l: None for l in cap_layers}
        self.cap_pos = torch.tensor(cap_pos, device=self.dev)
        try:
            h = self.hf.model(input_ids=x, use_cache=False).last_hidden_state
            cap = {l: v.float() for l, v in self.capture.items()}
        finally:
            self.ops, self.capture, self.cap_pos = None, None, None
        kls = []
        last = len(full) - 1 if cont else len(full)
        V = self.hf.lm_head.weight.shape[0]
        step = max(1, int(0.5e9 // (4 * B * V)))
        for c0 in range(start, last, step):
            lp = torch.log_softmax(self.hf.lm_head(h[:, c0:min(last, c0 + step)]).float(), -1)
            kls.append((lp[0:1].exp() * (lp[0:1] - lp)).sum(-1).clamp_min(0))
            del lp
        del h
        K = torch.cat(kls, 1).cpu()
        return K.sum(1).tolist(), cap

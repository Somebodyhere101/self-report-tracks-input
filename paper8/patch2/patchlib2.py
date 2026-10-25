"""Paper 8 patching arm 2: whole-state interchange at prompt AND generated (answer) positions.

Extends ../patch/patchlib.PatchModel (unchanged, imported). The interchange is done inside one batch:

- donor rows run their own prompt, teacher-forced on the answer tokens they share with the recipient
  (arm S: same product, same answer tokens), so the donor's state at every answer position is defined;
- a patched row b names a source row src[b] and a mask over ABSOLUTE positions 0 .. W-1; at block l the
  hook replaces row b's output at the masked positions with row src[b]'s output at the same positions,
  in the prompt pass (step 0) and at each generation step that processes a masked position.

Because the donor's states are computed in the same forward pass as the patched rows, the copy is
bit-exact (no batch-shape numerics between the donor's run and the copy), and with every position
patched, every block >= l of the patched row equals the donor row's.

paper 8's hook (capture for the logit lens) runs after the replacement, so the lens reads the patched
state. The old prompt-only patch (PatchModel.patch) is left in place and unused.
"""

from __future__ import annotations

import os
import sys

import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "patch"))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from patchlib import PatchModel  # noqa: E402


class PatchModel2(PatchModel):
    def __init__(self, *a, **kw):
        self.patch2 = None  # {layer: (mask [B, W] bool, src [B] long)}, absolute positions 0..W-1
        self.ppos0 = 0      # absolute position of the first token of the current forward pass
        self.keep = None    # {layer: [block outputs per step]} for the selftest
        super().__init__(*a, **kw)

    def _hook(self, i):
        base = PatchModel._hook(self, i)  # old prompt patch (inactive here), then paper 8's hook

        def f(mod, inp, out):
            pt = self.patch2
            if pt is not None and i in pt:
                h = out[0] if isinstance(out, tuple) else out
                mask, src = pt[i]
                T, p0, W = h.shape[1], self.ppos0, mask.shape[1]
                if p0 < W:
                    k = min(W, p0 + T) - p0  # leading positions of this pass that the patch covers
                    seg = torch.where(mask[:, p0:p0 + k, None], h[src, :k], h[:, :k])
                    h = torch.cat([seg, h[:, k:]], 1) if T > k else seg
                    out = (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            if self.keep is not None and i in self.keep:
                hh = out[0] if isinstance(out, tuple) else out
                self.keep[i].append(hh.detach().clone())
            return base(mod, inp, out)
        return f

    @torch.no_grad()
    def gen2(self, ids_rows, patch, n_new, force=None, lens=True, keep_layers=None):
        """Greedy decoding of B rows (equal-length prompts, which may differ between rows) with an in-batch
        patch {layer: (mask [B, W], src [B])}; the loop of PatchModel.gen. force = (rows [B] bool, tokens
        [K]): those rows are fed tokens[step] instead of their argmax for the first K steps (their lens and
        argmax are still recorded). Returns tokens [B, steps] (argmax), lens top-1 [B, steps, L] and,
        with keep_layers, those blocks' outputs at every processed position [B, n + steps - 1, d]."""
        B = len(ids_rows)
        x = torch.tensor(ids_rows, device=self.dev)
        T = x.shape[1]
        cache = transformers.DynamicCache()
        out_tok, top1s = [], []
        done = torch.zeros(B, dtype=torch.bool, device=self.dev)
        eos_t = torch.tensor(sorted(self.eos), device=self.dev)
        self.patch2 = None if not patch else {l: (mk.to(self.dev, torch.bool), sr.to(self.dev, torch.long))
                                              for l, (mk, sr) in patch.items()}
        if force is not None:
            frows = force[0].to(self.dev, torch.bool)
            ftok = [int(t) for t in force[1]]
        self.keep = {l: [] for l in keep_layers} if keep_layers else None
        try:
            for step in range(n_new):
                if lens:
                    self.capture = {l: None for l in range(self.L)}
                p0 = 0 if step == 0 else T - 1
                self.ppos0 = p0
                pos = torch.arange(p0, p0 + x.shape[1], device=self.dev)[None].expand(B, -1)
                o = self.hf(input_ids=x, past_key_values=cache, use_cache=True, position_ids=pos, logits_to_keep=1)
                logits = o.logits[:, -1].float()
                nxt = logits.argmax(-1)
                if lens:
                    top1s.append(self._lens_top1())
                nxt = torch.where(done, eos_t[0], nxt)
                out_tok.append(nxt)
                feed = nxt
                if force is not None and step < len(ftok):
                    feed = torch.where(frows, torch.full_like(nxt, ftok[step]), nxt)
                done |= torch.isin(feed, eos_t)
                x = feed[:, None]
                T += 1
                if bool(done.all()):
                    break
            kept = {l: torch.cat(v, 1) for l, v in self.keep.items()} if self.keep else None
        finally:
            self.patch2, self.capture, self.keep, self.ppos0 = None, None, None, 0
        del cache
        res = dict(tokens=torch.stack(out_tok, 1).cpu())
        if lens:
            res["top1"] = torch.stack(top1s, 1).cpu()
        if kept is not None:
            res["kept"] = kept
        return res

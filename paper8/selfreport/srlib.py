"""Paper 8 self-report arm: engine (selfreport/PREDICTIONS.md).

SRModel extends ../patch2/patchlib2.PatchModel2 (unchanged, imported) with one teacher-forced pass over
whole sequences (prompt + shared answer [+ follow-up turn]), in which:
- the in-batch interchange of PatchModel2 applies at its masked absolute positions (ppos0 = 0, one pass);
- paper 8's additive push (P8Model ops: per-row add at one layer, per-row position mask) applies too,
  for the Gaussian control rows;
- every block output is captured, so the logit lens and the tuned lens can be read at the digit
  positions, and the final logits give the follow-up's letter readout and the KL window.

The interchange happens at fixed absolute positions in a single pass, so every later position (the rest
of the answer and the whole follow-up turn) attends to the patched states at layers >= l: the patch stays
in force through the follow-up, as a patched KV cache would.
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "patch2"))
sys.path.insert(0, os.path.join(HERE, "..", "patch"))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from patchlib2 import PatchModel2  # noqa: E402

QUESTION = "Did you work that out step by step, or did you just know it? Answer A or B."
OPTS = ("I worked it out step by step.", "I just knew it.")  # order 0: A = step by step; order 1: reversed
PREFILL = "The answer is ("


def tail_text(order):
    a, b = OPTS if order == 0 else OPTS[::-1]
    q = f"{QUESTION}\n\nA) {a}\nB) {b}"
    return "<|im_end|>\n<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" + PREFILL


class SRModel(PatchModel2):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.tuned = None  # (W [L-1, d+1, d] float32, on GPU) : tuned-lens translators for layers 0..L-2

    def tail_ids(self, order):
        return self.tok(tail_text(order), add_special_tokens=False)["input_ids"]

    def letter_ids(self):
        return self.tok.convert_tokens_to_ids("A"), self.tok.convert_tokens_to_ids("B")

    @torch.no_grad()
    def tf_pass(self, rows_ids, patch=None, add=None, pmask=None):
        """One teacher-forced pass over equal-length rows. patch: {layer: (mask [B, W] bool, src [B] long)};
        add: {layer: [B, d]} with pmask [B, T] (positions where the add applies). Returns the final hidden
        state [B, T, d] (after the final norm) and every block output {l: [B, T, d]} (bf16, GPU)."""
        x = torch.tensor(rows_ids, device=self.dev)
        B, T = x.shape
        self.patch2 = None if not patch else {l: (mk.to(self.dev, torch.bool), sr.to(self.dev, torch.long))
                                              for l, (mk, sr) in patch.items()}
        self.ppos0 = 0
        if add:
            self.ops = self._ops(T, add, pmask, torch.zeros(B))
        self.capture = {l: None for l in range(self.L)}
        try:
            h = self.hf.model(input_ids=x, use_cache=False).last_hidden_state
            cap = dict(self.capture)
        finally:
            self.patch2, self.ops, self.capture, self.ppos0 = None, None, None, 0
        return h, cap

    @torch.no_grad()
    def lens_top1(self, cap, pos, tuned=False):
        """Top-1 token of the logit lens (or the tuned lens) at positions pos for every layer: [B, len(pos), L]."""
        out = []
        for l in range(self.L):
            hl = cap[l][:, pos].float()
            if tuned and l < self.L - 1:
                W = self.tuned[l]
                hl = hl @ W[:-1] + W[-1]
            out.append(self.hf.lm_head(self.norm(hl.to(torch.bfloat16))).argmax(-1))
        return torch.stack(out, -1)

    @torch.no_grad()
    def window_logp(self, h, p0, p1):
        """log-softmax of the next-token logits at positions p0 .. p1-1: [B, p1-p0, V] float32."""
        return torch.log_softmax(self.hf.lm_head(h[:, p0:p1]).float(), -1)

"""S2 shared library: model, prompts, minimal LoRA, clamp/push interventions,
KV-cache decoding, activation collection, logistic probes, DAS, AUC.

Conventions
- Layer l means the residual stream after l decoder blocks (output of
  model.model.layers[l-1]); the probe layer is l = 14 of 28.
- Every teacher-forced sequence is left-padded: [pad][prompt][48 response tokens].
  Response read positions are the 48 positions T-49..T-2 (each predicts one
  response token). Mode "resp" reads/intervenes there. Mode "all" uses every
  real position except the first real token and the final position.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, math, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
RES = os.path.join(ROOT, "res")
sys.path.insert(0, CODE)
from scorers import SCORERS  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"
NRESP = 48
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- model / data
def load_model():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return tok, model


def eos_ids(tok):
    ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    return sorted(i for i in ids if i is not None and i >= 0)


def load_items():
    with open(os.path.join(CODE, "items.json"), encoding="utf-8") as fh:
        return json.load(fh)


def prompt_ids(tok, instruction, question):
    text = tok.apply_chat_template([{"role": "user", "content": f"{instruction} {question}"}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def left_pad(seqs, pad_id):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s, dtype=torch.long)
        attn[i, T - len(s):] = 1
    pos = (attn.cumsum(1) - 1).clamp(min=0)
    return ids.to(DEV), attn.to(DEV), pos.to(DEV)


def read_mask(attn, mode):
    """(B,T) float mask of read/intervention positions for a teacher-forced batch."""
    B, T = attn.shape
    if mode == "resp":
        m = torch.zeros((B, T), device=attn.device)
        m[:, T - NRESP - 1:T - 1] = 1
        return m
    m = attn.float().clone()
    first = (attn.cumsum(1) == 1) & (attn == 1)
    m[first] = 0
    m[:, T - 1] = 0
    return m


# ---------------------------------------------------------------- LoRA
LORA_TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32))
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale
        return y


def _get(mod, path):
    for p in path.split("."):
        mod = getattr(mod, p)
    return mod


def _set(mod, path, val):
    parts = path.split(".")
    for p in parts[:-1]:
        mod = getattr(mod, p)
    setattr(mod, parts[-1], val)


def install_lora(model, blocks, r=8, alpha=16, seed=0):
    torch.manual_seed(seed)
    mods = []
    for b in blocks:
        layer = model.model.layers[b]
        for path in LORA_TARGETS:
            m = LoRALinear(_get(layer, path), r, alpha)
            _set(layer, path, m)
            mods.append((b, path, m))
    return mods


def remove_lora(model, mods):
    for b, path, m in mods:
        _set(model.model.layers[b], path, m.base)


def lora_enabled(mods, flag):
    for _, _, m in mods:
        m.enabled = flag


def lora_params(mods):
    return [p for _, _, m in mods for p in (m.A, m.B)]


# ---------------------------------------------------------------- hooks
class Capture:
    def __init__(self):
        self.h = None

    def __call__(self, mod, inp, out):
        self.h = out[0] if isinstance(out, tuple) else out


class Edit:
    """Edit the residual after a block at masked positions.

    kind="clamp": set the coordinates along orthonormal Q (d,k) to target (B,k).
    kind="add":   add vec (B,d) (already scaled).
    Teacher-forced mode: mask (B,T). Generation mode: prefill_mask (B,Tp) for the
    first call, then every later single-token call is edited at its position.
    """

    def __init__(self, kind, Q=None, target=None, vec=None, mask=None, prefill_mask=None):
        self.kind, self.Q, self.target, self.vec = kind, Q, target, vec
        self.mask, self.prefill_mask = mask, prefill_mask

    def __call__(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.mask is not None:
            m = self.mask
        elif h.shape[1] > 1:
            m = self.prefill_mask
        else:
            m = torch.ones(h.shape[0], 1, device=h.device)
        hf = h.float()
        if self.kind == "clamp":
            coord = hf @ self.Q
            delta = (self.target[:, None, :] - coord) @ self.Q.t()
        else:
            delta = self.vec[:, None, :].expand_as(hf)
        hn = (hf + delta * m[..., None]).to(h.dtype)
        return (hn,) + tuple(out[1:]) if isinstance(out, tuple) else hn


def hook(model, layer, fn):
    return model.model.layers[layer - 1].register_forward_hook(fn)


# ---------------------------------------------------------------- forward helpers
def tf_forward(model, ids, attn, pos, layer=None, n_logits=NRESP, grad=False):
    """Teacher-forced forward. Returns (h_layer or None, logits over the response read positions)."""
    cap = Capture()
    hd = hook(model, layer, cap) if layer is not None else None
    try:
        with torch.set_grad_enabled(grad):
            out = model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
            T = ids.shape[1]
            logits = model.lm_head(out.last_hidden_state[:, T - NRESP - 1:T - NRESP - 1 + n_logits])
    finally:
        if hd is not None:
            hd.remove()
    return cap.h, logits


def kl_rows(lp_ref, logits):
    """Per-position KL(ref || new); lp_ref is a log-softmax (float), logits raw."""
    lp = F.log_softmax(logits.float(), -1)
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1)


# ---------------------------------------------------------------- decoding
@torch.no_grad()
def decode(model, tok, prompts, greedy=True, temp=0.7, top_p=0.95, seed=0, bs=48, layer=None,
           edit_factory=None, n=NRESP):
    """KV-cache decoding of exactly n tokens (end-of-turn tokens banned).
    edit_factory(i0, attn) -> Edit (generation mode) or None."""
    ban = eos_ids(tok)
    outs = []
    g = torch.Generator(device=DEV)
    for i0 in range(0, len(prompts), bs):
        ids, attn, pos = left_pad(prompts[i0:i0 + bs], tok.pad_token_id)
        g.manual_seed(seed * 100003 + i0)
        hd = None
        if edit_factory is not None:
            hd = hook(model, layer, edit_factory(i0, attn))
        try:
            out = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            logits = out.logits[:, -1].float()
            cur = pos[:, -1:]
            toks = []
            for t in range(n):
                logits[:, ban] = -float("inf")
                if greedy:
                    nxt = logits.argmax(-1)
                else:
                    probs = F.softmax(logits / temp, -1)
                    sp, si = probs.sort(-1, descending=True)
                    sp[(sp.cumsum(-1) - sp) > top_p] = 0
                    nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=g)).squeeze(-1)
                toks.append(nxt)
                if t == n - 1:
                    break
                attn = torch.cat([attn, torch.ones_like(attn[:, :1])], 1)
                cur = cur + 1
                out = model(input_ids=nxt[:, None], attention_mask=attn, position_ids=cur,
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1].float()
            outs += torch.stack(toks, 1).tolist()
        finally:
            if hd is not None:
                hd.remove()
    return outs


def prefill_mask(attn, mode):
    if mode == "resp":
        m = torch.zeros_like(attn, dtype=torch.float)
        m[:, -1] = 1
        return m
    m = attn.float().clone()
    first = (attn.cumsum(1) == 1) & (attn == 1)
    m[first] = 0
    return m


def concept_rate(tok, concept, outs):
    f = SCORERS[concept]
    return np.array([bool(f(tok.decode(o, skip_special_tokens=True))[0]) for o in outs])


def clamp_factory(Q, target, mode):
    """Edit factory for decode(): clamp along Q to a fixed target (k,) for every row."""
    def fac(i0, attn):
        return Edit("clamp", Q=Q, target=target[None].expand(attn.shape[0], -1),
                    prefill_mask=prefill_mask(attn, mode))
    return fac


# ---------------------------------------------------------------- activations
@torch.no_grad()
def collect(model, tok, prompts, resps, layer, mode, bs=24, ref_logp=False):
    """Layer-l activations at read positions. Returns A (N,R,d) bf16, M (N,R) bool
    (right-aligned, R = 48 for mode resp), and optionally nothing else."""
    As, Ms = [], []
    for i0 in range(0, len(prompts), bs):
        seqs = [p + r for p, r in zip(prompts[i0:i0 + bs], resps[i0:i0 + bs])]
        ids, attn, pos = left_pad(seqs, tok.pad_token_id)
        h, _ = tf_forward(model, ids, attn, pos, layer, n_logits=1)
        m = read_mask(attn, mode).bool()
        As.append((h, m))
    R = max(int(m.sum(1).max()) for _, m in As)
    N = len(prompts)
    d = As[0][0].shape[-1]
    A = torch.zeros((N, R, d), dtype=torch.bfloat16, device=DEV)
    M = torch.zeros((N, R), dtype=torch.bool, device=DEV)
    k = 0
    for h, m in As:
        for j in range(h.shape[0]):
            v = h[j][m[j]]
            A[k, R - v.shape[0]:] = v
            M[k, R - v.shape[0]:] = True
            k += 1
    return A, M


def token_scores(A, w):
    return (A.float() @ w.float().to(A.device))


def pool(S, M, how):
    if how == "mean":
        return (S * M).sum(1) / M.sum(1)
    return S.masked_fill(~M, -float("inf")).max(1).values


def class_means(A, M):
    Af = A.float()
    return (Af * M[..., None]).sum((0, 1)) / M.sum()


# ---------------------------------------------------------------- probes / stats
def auc(pos, neg):
    """Mann-Whitney AUC (ties count 1/2)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    from scipy.stats import rankdata
    r = rankdata(allv)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def fit_logreg(X, y, l2=1e-3, iters=300):
    """Token-level L2 logistic regression on standardised features; returns raw-space w (d,), b."""
    X = X.float()
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Z = (X - mu) / sd
    y = y.float().to(X.device)
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Z @ w + b, y) + 0.5 * l2 * (w ** 2).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    w_raw = (w / sd).detach()
    b_raw = (b - (mu / sd) @ w).detach()
    return w_raw, b_raw


def flat_tokens(A, M):
    return A[M].float()


# ---------------------------------------------------------------- DAS
def orth(v):
    if v.shape[1] == 1:
        return v / v.norm()
    return torch.linalg.qr(v).Q


def fit_das(model, tok, k, layer, mode, PG, PE, RG, RE, mG, mE, init, steps=150, lr=1e-2, bs=4,
            n_loss=8, seed=0):
    """Fit a k-dim DAS subspace with class-mean interchange.

    Pairs share a question: rows "E prompt + G response" are clamped to the G class
    mean (target: produce the G response), rows "G prompt + E response" to the E
    mean. Loss: cross-entropy on the first n_loss response tokens."""
    v = init.detach().clone().float().to(DEV)
    v = (v / v.norm(dim=0, keepdim=True) * math.sqrt(v.shape[0])).requires_grad_(True)  # entries ~1, so lr is relative
    opt = torch.optim.Adam([v], lr=lr)
    rng = np.random.default_rng(seed)
    n = len(PG)
    order = rng.permutation(n)
    ptr = 0
    hist = []
    for step in range(steps):
        if ptr + bs > n:
            order, ptr = rng.permutation(n), 0
        idx = order[ptr:ptr + bs]
        ptr += bs
        seqs = [PE[i] + RG[i] for i in idx] + [PG[i] + RE[i] for i in idx]
        ids, attn, pos = left_pad(seqs, tok.pad_token_id)
        Q = orth(v)
        tgt = torch.cat([(mG @ Q)[None].expand(bs, -1), (mE @ Q)[None].expand(bs, -1)], 0)
        ed = Edit("clamp", Q=Q, target=tgt, mask=read_mask(attn, mode))
        hd = hook(model, layer, ed)
        try:
            _, logits = tf_forward(model, ids, attn, pos, None, n_logits=n_loss, grad=True)
        finally:
            hd.remove()
        T = ids.shape[1]
        tgt_tok = ids[:, T - NRESP:T - NRESP + n_loss]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt_tok.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            hist.append((step, float(loss.detach())))
    return orth(v.detach()), hist


def gate_eval(model, tok, concept, layer, mode, Q, mG, mE, PG, PE, base_G=None):
    """Sufficiency (both directions) and necessity for subspace Q on prompts PG/PE.
    Returns dict with per-item flags."""
    tG, tE = mG @ Q, mE @ Q
    tM = 0.5 * (tG + tE)
    if base_G is None:
        base_G = concept_rate(tok, concept, decode(model, tok, PG))
    e2g = concept_rate(tok, concept, decode(model, tok, PE, layer=layer, edit_factory=clamp_factory(Q, tG, mode)))
    g2e = ~concept_rate(tok, concept, decode(model, tok, PG, layer=layer, edit_factory=clamp_factory(Q, tE, mode)))
    abl = concept_rate(tok, concept, decode(model, tok, PG, layer=layer, edit_factory=clamp_factory(Q, tM, mode)))
    return {"suff": float(np.concatenate([e2g, g2e]).mean()), "suff_e2g": float(e2g.mean()),
            "suff_g2e": float(g2e.mean()), "base_rate": float(base_G.mean()), "abl_rate": float(abl.mean()),
            "nec_pp": float(100 * (base_G.mean() - abl.mean())),
            "items": {"e2g": e2g.tolist(), "g2e": g2e.tolist(), "abl": abl.tolist(), "base": base_G.tolist()}}


@torch.no_grad()
def push_kl(model, tok, layer, mode, direction, alpha, PG, RG, PE, RE, bs=16):
    """Mean per-token KL(orig || pushed) at the response positions when the unit
    direction is added with -alpha on G rows and +alpha on E rows at read positions."""
    u = (direction / direction.norm()).float().to(DEV)
    tot, cnt = 0.0, 0
    for P, R, sgn in ((PG, RG, -1.0), (PE, RE, 1.0)):
        for i0 in range(0, len(P), bs):
            seqs = [p + r for p, r in zip(P[i0:i0 + bs], R[i0:i0 + bs])]
            ids, attn, pos = left_pad(seqs, tok.pad_token_id)
            _, l0 = tf_forward(model, ids, attn, pos)
            vec = (sgn * alpha * u)[None].expand(ids.shape[0], -1)
            hd = hook(model, layer, Edit("add", vec=vec, mask=read_mask(attn, mode)))
            try:
                _, l1 = tf_forward(model, ids, attn, pos)
            finally:
                hd.remove()
            kl = kl_rows(F.log_softmax(l0.float(), -1), l1)
            tot += float(kl.sum())
            cnt += kl.numel()
    return tot / cnt

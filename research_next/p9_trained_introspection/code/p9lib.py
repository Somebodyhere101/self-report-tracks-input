"""Paper 9 engine: interventions on the turn-1 computation, forced-choice reports in turn 2, gated LoRA.

Reused read-only:
- paper 8's in-batch exact interchange engine (../../../paper8/selfreport/srlib.SRModel, which extends
  patch2/patchlib2.PatchModel2 -> patch/patchlib.PatchModel -> code/p8lib.P8Model -> code/p6lib.Model).
  Its block hook copies row src[b]'s block output into row b at masked absolute positions (bit-exact,
  same pass), and then applies paper 8's per-row additive push at masked positions.
- the S2 session's LoRA (../../s2_monitors/code/s2lib.LoRALinear; no peft installed).

A conversation is
  turn 1: <user> Which continent is the country containing {city} in? ... <assistant> {continent}<|im_end|>\n
  turn 2: <user> {question with four lettered options} <assistant> The answer is (   -> next token = letter
Turn 1 runs first (no gradient, LoRA off) with the interventions, and returns its key/value cache.
Turn 2 runs on top of that cache with the LoRA on (the report adapter is active only while reporting),
so the computation being reported is always the unmodified base model's.

Region R = every turn-1 position after the city span (the rest of the question, the chat template, the
answer and the closing <|im_end|>\n). Interventions act on R only; the city tokens are never touched.
Layer l means the OUTPUT of block l (0-indexed), as in paper 8.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WS = os.path.abspath(os.path.join(ROOT, "..", ".."))
EXT = [os.path.abspath(p) for p in (os.path.join(WS, "paper8", "selfreport"), os.path.join(WS, "paper8", "patch2"),
                                    os.path.join(WS, "paper8", "patch"), os.path.join(WS, "paper8", "code"),
                                    os.path.join(WS, "research_next", "s2_monitors", "code"))]
for p in EXT:
    if p not in sys.path:
        sys.path.append(p)

from srlib import SRModel  # noqa: E402  (paper 8, read-only)
import s2lib  # noqa: E402  (S2 LoRA, read-only)

# paper 8's and S2's modules put their own folders at the front of sys.path when imported; move them behind
# this folder so that S2's items.py / analyze.py / report_tables.py can never shadow paper 9's modules
sys.path[:] = [HERE] + [p for p in sys.path if os.path.abspath(p) not in EXT + [HERE]] + EXT

from cities import COUNTRIES  # noqa: E402

# P9_TINY=1: code dry run on CPU with SmolLM2-135M-Instruct, tiny item counts, separate *_tiny output folders
TINY = bool(os.environ.get("P9_TINY"))


def outdir(name):
    return os.path.join(ROOT, name + ("_tiny" if TINY else ""))

LETTERS = ["A", "B", "C", "D"]
CONT_FIRST = {"Europe": "Europe", "Asia": "Asia", "Africa": "Africa", "North America": "North",
              "South America": "South"}
Q1 = "Which continent is the country containing {city} in? Answer with the continent name only."
Q_REPORT = "Which country did you use to work out your answer?"
Q_READ_COUNTRY = "Which country was the question about?"
Q_READ_CAPITAL = "What is the capital of the country the question was about?"
Q_ONEHOP_FC = "Which country is {city} in?"
Q_ONEHOP_FREE = "Which country is {city} in? Answer with the country name only."
PREFILL = "The answer is ("
TAIL_END = "\nAnswer with the letter."
# placeholders for the readout context, by number of city tokens (verified at load time)
PLACEHOLDERS = [" X", " city X", " the city X", " the city of X", " the city called X Y"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def seeded(*keys):
    h = hashlib.sha256("|".join(str(k) for k in keys).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def options_text(opts):
    return "\n".join(f"{L}) {o}" for L, o in zip(LETTERS, opts))


def rotate(cands, r):
    """Rotation r shows cands[(j + r) % 4] at letter j."""
    return [cands[(j + r) % 4] for j in range(4)]


def rot_scores(lp_rot):
    """lp_rot: [4 rotations][4 letters] log-probs (softmax over the four letters). Returns the
    rotation-averaged score of each candidate in base order: s[k] = mean_r lp_r[(k - r) % 4]."""
    return [sum(lp_rot[r][(k - r) % 4] for r in range(4)) / 4 for k in range(4)]


class P9(SRModel):
    def __init__(self, device="cuda", key="q3-1.7b"):
        if TINY:
            import p6lib
            p6lib.MODELS["tiny"] = "HuggingFaceTB/SmolLM2-135M-Instruct"
            key, device = "tiny", "cpu"
            torch.set_num_threads(2)
        super().__init__(key, device=device, thinking=False)
        self.letter_ids = [self.tok.convert_tokens_to_ids(L) for L in LETTERS]
        self.cont_ids = {c: self.tok(f, add_special_tokens=False)["input_ids"][0] for c, f in CONT_FIRST.items()}
        self.pad_id = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.lora = None
        self._pre = self._enc(self.chat(Q1.format(city="\x00")).split("\x00")[0].rstrip(" "))
        self.ph = {}
        for p in PLACEHOLDERS:
            n = len(self._enc(p))
            self.ph.setdefault(n, p)

    # ------------------------------------------------------------------ text
    def _enc(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def answer_text(self, continent):
        return continent

    def turn1(self, city, continent, placeholder=False):
        """Turn-1 ids for a city (or the placeholder of the same token count), the city span start r_c and
        R's start r0. Asserts that the city tokens do not merge with their neighbours."""
        n_city = len(self._enc(" " + city))
        name = self.ph[n_city].lstrip(" ") if placeholder else city
        text = self.chat(Q1.format(city=name)) + self.answer_text(continent) + "<|im_end|>\n"
        ids = self._enc(text)
        pre = self._pre
        span = self._enc(" " + name)
        assert ids[:len(pre)] == pre and ids[len(pre):len(pre) + len(span)] == span, (city, placeholder)
        assert len(span) == n_city, (city, placeholder, len(span), n_city)
        if not TINY:  # the answer starts exactly where the 2-hop prompt ends (used by kl_units)
            q = self._enc(self.chat(Q1.format(city=name)))
            assert ids[:len(q)] == q, (city, "answer merges with the prompt")
        return ids, len(pre), len(pre) + len(span)

    def tail(self, question, opts):
        q = question + "\n" + options_text(opts) + TAIL_END
        return self._enc("<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                         + PREFILL)

    def fresh(self, user_text):
        return self._enc(self.chat(user_text))

    # ------------------------------------------------------------------ LoRA
    def install_lora(self, seed, r=8, alpha=16):
        torch.manual_seed(seed)
        mods = []
        for b, layer in enumerate(self.layers):
            for path in s2lib.LORA_TARGETS:
                m = s2lib.LoRALinear(s2lib._get(layer, path), r, alpha)
                s2lib._set(layer, path, m)
                mods.append((b, path, m))
        self.lora = mods
        self.lora_on(False)
        return mods

    def remove_lora(self):
        if self.lora:
            for b, path, m in self.lora:
                s2lib._set(self.layers[b], path, m.base)
        self.lora = None

    def lora_on(self, flag):
        if self.lora:
            s2lib.lora_enabled(self.lora, flag)

    def lora_state(self):
        return {f"{b}.{p}": (m.A.detach().cpu(), m.B.detach().cpu()) for b, p, m in self.lora}

    def load_lora(self, state, seed=0):
        self.install_lora(seed)
        for b, p, m in self.lora:
            A, B = state[f"{b}.{p}"]
            m.A.data.copy_(A.to(m.A.device))
            m.B.data.copy_(B.to(m.B.device))
            m.A.requires_grad_(False)
            m.B.requires_grad_(False)

    # ------------------------------------------------------------------ passes
    def _turn1_pass(self, rows, capture_layers=None, lora_turn1=False):
        """rows: list of dicts with ids (equal length), r0, copy {layer: src index in rows}, add (layer, vec)
        or None. Returns (cache, final hidden [B, T, d], captured {l: [B, T, d]})."""
        B, T = len(rows), len(rows[0]["ids"])
        assert all(len(r["ids"]) == T and r["r0"] == rows[0]["r0"] for r in rows)
        r0 = rows[0]["r0"]
        x = torch.tensor([r["ids"] for r in rows], device=self.dev)
        patch = {}
        for b, r in enumerate(rows):
            for l, s in r.get("copy", {}).items():
                if l not in patch:
                    patch[l] = (torch.zeros(B, T, dtype=torch.bool), torch.arange(B))
                patch[l][0][b, r0:] = True
                patch[l][1][b] = s
        self.patch2 = {l: (m.to(self.dev), s.to(self.dev)) for l, (m, s) in patch.items()} or None
        self.ppos0 = 0
        adds = [r.get("add") for r in rows]
        if any(a is not None for a in adds):
            layers = sorted({a[0] for a in adds if a is not None})
            addd = {l: torch.zeros(B, self.d) for l in layers}
            pm = torch.zeros(B, T)
            for b, a in enumerate(adds):
                if a is not None:
                    addd[a[0]][b] = a[1].float().cpu()
                    pm[b, r0:] = 1.0
            self.ops = self._ops(T, addd, pm, torch.zeros(B))
        self.capture = {l: None for l in capture_layers} if capture_layers else None
        self.lora_on(lora_turn1)
        try:
            cache = transformers.DynamicCache()
            out = self.hf.model(input_ids=x, past_key_values=cache, use_cache=True)
            cap = {l: v for l, v in self.capture.items()} if capture_layers else None
        finally:
            self.patch2, self.ops, self.capture, self.ppos0 = None, None, None, 0
        return out.past_key_values, out.last_hidden_state, cap

    def _select_cache(self, cache, sel):
        new = transformers.DynamicCache()
        idx = torch.tensor(sel, device=self.dev)
        for i, layer in enumerate(cache.layers):
            new.update(layer.keys[idx], layer.values[idx], i)
        return new

    def _turn2_pass(self, cache, T1, sel, tails, lora=True):
        """Continue rows sel of a turn-1 cache with right-padded tails; returns logits [K, V] (float) at
        each tail's last token."""
        K = len(sel)
        m = max(len(t) for t in tails)
        x = torch.tensor([t + [self.pad_id] * (m - len(t)) for t in tails], device=self.dev)
        pos = (torch.arange(m, device=self.dev) + T1)[None].expand(K, -1)
        c2 = self._select_cache(cache, sel)
        self.lora_on(lora)
        try:
            h = self.hf.model(input_ids=x, past_key_values=c2, position_ids=pos, use_cache=True).last_hidden_state
        finally:
            self.lora_on(False)
        last = torch.tensor([len(t) - 1 for t in tails], device=self.dev)
        hl = h[torch.arange(K, device=self.dev), last]
        return self.hf.lm_head(hl).float()

    # ------------------------------------------------------------------ units
    # A unit is {"rows": [row, ...], "reqs": [(row index, tail ids, tag), ...]} whose rows share T1 and r0.
    # A row's copy map and add refer to row indices inside its unit.

    @staticmethod
    def _key(unit):
        r = unit["rows"][0]
        return (len(r["ids"]), r["r0"])

    @torch.no_grad()
    def run_units(self, units, lora=False, lora_turn1=False, max_rows=48, max_reqs=96, capture=None):
        """Score every request of every unit. Returns, per unit, a list of (letter log-probs (softmax over the
        four letters), full-vocab argmax) in request order. capture=(layer, ...) also returns each row's
        block outputs at the R positions, {layer: [rows, |R|, d]} (CPU float16)."""
        groups = {}
        for ui, u in enumerate(units):
            groups.setdefault(self._key(u), []).append(ui)
        out = [None] * len(units)
        caps = [None] * len(units)
        for key, uis in groups.items():
            batch, nrow, nreq = [], 0, 0
            for ui in uis + [None]:
                u = units[ui] if ui is not None else None
                if u is not None and (not batch or (nrow + len(u["rows"]) <= max_rows
                                                    and nreq + len(u["reqs"]) <= max_reqs)):
                    batch.append(ui)
                    nrow += len(u["rows"])
                    nreq += len(u["reqs"])
                    continue
                self._run_batch(units, batch, out, caps, lora, lora_turn1, capture)
                batch, nrow, nreq = ([ui], len(u["rows"]), len(u["reqs"])) if u is not None else ([], 0, 0)
        return (out, caps) if capture else out

    def _flatten(self, units, batch):
        rows, reqs, owner = [], [], []
        for ui in batch:
            off = len(rows)
            for r in units[ui]["rows"]:
                rr = dict(r)
                rr["copy"] = {l: s + off for l, s in r.get("copy", {}).items()}
                rows.append(rr)
            for (ri, t, tag) in units[ui]["reqs"]:
                reqs.append((ri + off, t))
                owner.append(ui)
        return rows, reqs, owner

    def _run_batch(self, units, batch, out, caps, lora, lora_turn1, capture):
        if not batch:
            return
        rows, reqs, owner = self._flatten(units, batch)
        cache, h, cap = self._turn1_pass(rows, capture_layers=capture, lora_turn1=lora_turn1)
        T1 = len(rows[0]["ids"])
        res = []
        if reqs:
            lg = self._turn2_pass(cache, T1, [r for r, _ in reqs], [t for _, t in reqs], lora=lora)
            lp4 = torch.log_softmax(lg[:, self.letter_ids], -1).cpu().tolist()
            amax = lg.argmax(-1).cpu().tolist()
            res = list(zip(lp4, amax))
        for ui in batch:
            out[ui] = []
        for (lpv, am), ui in zip(res, owner):
            out[ui].append((lpv, am))
        if capture:
            r0 = rows[0]["r0"]
            off = 0
            for ui in batch:
                n = len(units[ui]["rows"])
                caps[ui] = {l: cap[l][off:off + n, r0:].to(torch.float16).cpu() for l in capture}
                off += n
        del cache, h, cap

    @torch.no_grad()
    def kl_units(self, units, max_rows=24):
        """For units whose rows[0] is the clean row and rows[1] the intervened row: KL(clean || intervened)
        of the next-token distributions, summed over R positions (base model). Also returns the turn-1
        continent argmax at the answer-predicting position (the last prompt position) for both rows."""
        out = []
        groups = {}
        for ui, u in enumerate(units):
            groups.setdefault(self._key(u), []).append(ui)
        res = [None] * len(units)
        cont_ids = list(self.cont_ids.values())
        conts = list(self.cont_ids.keys())
        for key, uis in groups.items():
            for i in range(0, len(uis), max(1, max_rows // 3)):
                chunk = uis[i:i + max(1, max_rows // 3)]
                rows, _, _ = self._flatten(units, chunk)
                _, h, _ = self._turn1_pass(rows)
                r0 = rows[0]["r0"]
                off = 0
                for ui in chunk:
                    n = len(units[ui]["rows"])
                    a = torch.log_softmax(self.hf.lm_head(h[off, r0:]).float(), -1)
                    b = torch.log_softmax(self.hf.lm_head(h[off + 1, r0:]).float(), -1)
                    kl = float((a.exp() * (a - b)).sum())
                    ap = units[ui]["ans_pos"] - 1  # position predicting the first answer token
                    ca = conts[int(self.hf.lm_head(h[off, ap]).float()[cont_ids].argmax())]
                    cb = conts[int(self.hf.lm_head(h[off + 1, ap]).float()[cont_ids].argmax())]
                    res[ui] = (kl, ca, cb)
                    off += n
                del h
        return res

    # ------------------------------------------------------------------ single-turn scoring
    @torch.no_grad()
    def score_fresh(self, prompts, max_b=64, lora=False):
        """prompts: list of (ids ending with the prefill). Returns letter log-probs (4), the log-softmax over
        the continent first tokens, and the full-vocab argmax, for each prompt (right-padded batches)."""
        self.lora_on(lora)
        out = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        cont_ids = list(self.cont_ids.values())
        for i in range(0, len(order), max_b):
            idx = order[i:i + max_b]
            m = max(len(prompts[j]) for j in idx)
            x = torch.tensor([prompts[j] + [self.pad_id] * (m - len(prompts[j])) for j in idx], device=self.dev)
            h = self.hf.model(input_ids=x, use_cache=False).last_hidden_state
            last = torch.tensor([len(prompts[j]) - 1 for j in idx], device=self.dev)
            lg = self.hf.lm_head(h[torch.arange(len(idx), device=self.dev), last]).float()
            lp4 = torch.log_softmax(lg[:, self.letter_ids], -1).cpu().tolist()
            lc = torch.log_softmax(lg[:, cont_ids], -1).cpu().tolist()
            am = lg.argmax(-1).cpu().tolist()
            for k, j in enumerate(idx):
                out[j] = (lp4[k], lc[k], am[k])
        self.lora_on(False)
        return out

    @torch.no_grad()
    def generate_free(self, prompts, n_new=10, max_b=64):
        """Greedy continuation of prompts (grouped by equal length); returns decoded strings."""
        self.lora_on(False)
        out = [None] * len(prompts)
        groups = {}
        for i, p in enumerate(prompts):
            groups.setdefault(len(p), []).append(i)
        for L, idx in groups.items():
            for i in range(0, len(idx), max_b):
                sub = idx[i:i + max_b]
                res = self.gen2([prompts[j] for j in sub], None, n_new, lens=False)
                for k, j in enumerate(sub):
                    toks = res["tokens"][k].tolist()
                    toks = [t for t in toks if t not in self.eos]
                    out[j] = self.tok.decode(toks, skip_special_tokens=True).strip()
        return out

    # ------------------------------------------------------------------ training
    def train_step_loss(self, units, targets):
        """units: one request each (the report), targets: letter index per unit. Returns the summed CE
        (full vocabulary) over units and the number of units; turn 1 without gradient, turn 2 with."""
        groups = {}
        for ui, u in enumerate(units):
            groups.setdefault(self._key(u), []).append(ui)
        total, n = 0.0, 0
        for key, uis in groups.items():
            rows, reqs, owner = self._flatten(units, uis)
            with torch.no_grad():
                cache, h, _ = self._turn1_pass(rows)
            del h
            T1 = len(rows[0]["ids"])
            with torch.enable_grad():
                lg = self._turn2_pass(cache, T1, [r for r, _ in reqs], [t for _, t in reqs], lora=True)
                tgt = torch.tensor([self.letter_ids[targets[ui]] for ui in owner], device=self.dev)
                loss = F.cross_entropy(lg, tgt, reduction="sum")
            total = total + loss
            n += len(owner)
        return total, n


def country_parse(text):
    """Map a free answer to a gazetteer country (longest alias match, case-insensitive), else None."""
    t = " " + text.lower().replace(".", " ").replace(",", " ") + " "
    best, bl = None, 0
    for c, (_, _, _, al) in COUNTRIES.items():
        for a in [c] + al:
            k = " " + a.lower().replace(".", " ").replace(",", " ").strip() + " "
            if k in t and len(k) > bl:
                best, bl = c, len(k)
    return best


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

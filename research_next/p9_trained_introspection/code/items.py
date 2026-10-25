"""Paper 9: item pools, donors, pairing and the training/test example lists (shared by pilot, gen, selftest)."""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p9lib import TINY, load_json, outdir, seeded  # noqa: E402
from build import (add_readout, add_report, cont_of, fill, readout_result, unit_add, unit_natural,  # noqa: E402
                   unit_swap)

ROOT = os.path.dirname(HERE)


def load_screen(path=None):
    d = load_json(path or os.path.join(outdir("res"), "screen.json"))
    return {i["city"]: i for i in d["items"] if i.get("ok")}


def usable(it):
    return it.get("ok") and (it.get("acc2") or TINY)


def ref(it, cb):
    """The recipient's reference country: its causal natural bridge when reliable, else the gazetteer."""
    c = cb.get(it["city"])
    return c if c else it["country"]


def is_known(it, cb):
    if TINY:
        return usable(it)
    return usable(it) and it["p1"] == it["country"] and it["free1"] == it["country"] and cb.get(it["city"]) == it["country"]


def natural_readouts(m, its, max_rows=48):
    """Causal natural bridge of each item: placeholder readout of its natural run. Returns {city: result}."""
    units = [add_readout(m, unit_natural(m, it["city"], it["continent"]), it["city"], it["continent"], it["cands"])
             for it in its]
    res = m.run_units(units, max_rows=max_rows)
    return {it["city"]: readout_result(r, it["cands"]) for it, r in zip(its, res)}


def donor_index(donors):
    idx = {}
    for d in donors:
        idx.setdefault((d["n"], d["continent"]), []).append(d)
    for v in idx.values():
        v.sort(key=lambda d: d["city"])
    return idx


def pick_donor(x, idx, cb, kind, rng, exclude=()):
    """A known donor with the same token count and continent; kind 'diff' (its bridge differs from both the
    recipient's reference country and its gazetteer country) or 'same' (bridge equal to the reference
    country, different city). None if there is none."""
    rx = ref(x, cb)
    pool = [d for d in idx.get((x["n"], x["continent"]), []) if d["city"] != x["city"] and d["city"] not in exclude
            and ((d["country"] not in (rx, x["country"])) if kind == "diff" else (d["country"] == rx))]
    return rng.choice(pool) if pool else None


def jvec_vectors(caps_by_city, its, min_items):
    """Country concept vectors at one layer: mean over R positions and over the country's items, minus the
    mean over its continent's items. caps_by_city: {city: [|R|, d] tensor}. Returns {country: unit vector}."""
    by_c, by_cont = {}, {}
    for it in its:
        v = caps_by_city[it["city"]].float().mean(0)
        by_c.setdefault(it["country"], []).append(v)
        by_cont.setdefault(it["continent"], []).append(v)
    cont_mean = {k: torch.stack(v).mean(0) for k, v in by_cont.items()}
    out = {}
    for c, vs in by_c.items():
        if len(vs) >= min_items:
            d = torch.stack(vs).mean(0) - cont_mean[cont_of(c)]
            out[c] = d / d.norm()
    return out


def base_norm(caps_by_city, pairs):
    """Mean over (recipient, donor) pairs and R positions of ||h_donor - h_recipient||."""
    vals = []
    for x, y in pairs:
        a, b = caps_by_city[x].float(), caps_by_city[y].float()
        if a.shape == b.shape:
            vals.append((b - a).norm(dim=-1).mean().item())
    return sum(vals) / len(vals)


def random_unit(d, key):
    g = torch.Generator().manual_seed(int(seeded("rand", key).random() * 2 ** 31))
    v = torch.randn(d, generator=g)
    return v / v.norm()


# ---------------------------------------------------------------------------- example -> unit
def example_unit(m, ex, cfg, jvec, rotations=(0, 1, 2, 3), readout=False):
    x, cont = ex["x"], ex["cont"]
    if ex["arm"] in ("O", "X", "N", "nat"):
        u = unit_natural(m, x, cont)
    elif ex["arm"] in ("I", "swap"):
        u = unit_swap(m, x, ex["y"], cont, ex.get("layers") or list(range(cfg["l1"], m.L)))
    elif ex["arm"] in ("J", "add"):
        vec = ex.get("vec")
        if vec is None and "rand" in ex:  # random direction (tamper control), scale in base-norm units
            vec = random_unit(m.d, ex["rand"]) * (ex["scale"] * cfg["base_norm"])
        elif vec is None:  # country concept vector
            vec = jvec[ex["c"]] * (ex.get("scale", cfg["s_add"]) * cfg["base_norm"])
        u = unit_add(m, x, cont, ex["layer"] if "layer" in ex else cfg["la"], vec)
    else:
        raise ValueError(ex["arm"])
    if readout:
        return add_readout(m, u, x, cont, ex["cands"])
    return add_report(m, u, ex["cands"], rotations)

"""Paper 9: splits, candidate sets and the units (rows + requests) used by every stage.

Unit layouts (row 0 is always first; the intervened row is the one the report or readout continues):
  natural      rows [x]                                        report on row 0
  swap         rows [donor y, x <- y at R for layers in Ls]     report on row 1
  add          rows [x + vec at R at layer la]                  report on row 0
  readout      the same rows plus a placeholder row P whose R copies the intervened row at EVERY layer
               (from the donor directly at the swapped layers, so no copy is chained within a layer;
               for an add, P copies the pre-add output and receives the same add, which reproduces the
               intervened row's state exactly). P's eight requests are four rotations of the country
               question and four of the capital question.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cities import CITIES, COUNTRIES, CONTINENTS  # noqa: E402
from p9lib import (Q_ONEHOP_FC, Q_ONEHOP_FREE, Q_READ_CAPITAL, Q_READ_COUNTRY, Q_REPORT, Q1,  # noqa: E402
                   PREFILL, TAIL_END, options_text, rotate, seeded)

SPLIT_PATTERN = ["train", "test", "pilot", "train", "test"]
Q_EXTERNAL = ("Here is a question that was put to an assistant, and the assistant's answer.\n\n"
              "Question: " + Q1 + "\nAnswer: {continent}\n\n"
              "Which country did the assistant use to work out its answer?")


def cont_of(c):
    return COUNTRIES[c][0]


def capital(c):
    return COUNTRIES[c][2]


def split_cities():
    """city -> split; within each country a seeded order cycles train, test, pilot, train, test."""
    out = {}
    for c in sorted(CITIES):
        xs = sorted(CITIES[c])
        seeded("split", c).shuffle(xs)
        for i, x in enumerate(xs):
            out[x] = SPLIT_PATTERN[i % len(SPLIT_PATTERN)]
    return out


def gaz(city):
    for c, xs in CITIES.items():
        if city in xs:
            return c
    raise KeyError(city)


def fill(core, key):
    """Four candidate countries: the core countries (in order, unique), then countries of the first core
    country's sub-region, then of its continent, each in a seeded order; returned in a seeded order."""
    out = []
    for c in core:
        if c and c not in out:
            out.append(c)
    cont, sub = COUNTRIES[out[0]][0], COUNTRIES[out[0]][1]
    rng = seeded("fill", key)
    same_sub = sorted(c for c, v in COUNTRIES.items() if v[0] == cont and v[1] == sub and c not in out)
    same_cont = sorted(c for c, v in COUNTRIES.items() if v[0] == cont and v[1] != sub and c not in out)
    rng.shuffle(same_sub)
    rng.shuffle(same_cont)
    for c in same_sub + same_cont:
        if len(out) >= 4:
            break
        out.append(c)
    assert len(out) == 4 and all(cont_of(c) == cont for c in out), (core, out)
    seeded("order", key).shuffle(out)
    return out


# ---------------------------------------------------------------------------- prompts
def onehop_fc_prompts(m, city, cands):
    return [m.fresh(Q_ONEHOP_FC.format(city=city) + "\n" + options_text(rotate(cands, r)) + TAIL_END)
            + m._enc(PREFILL) for r in range(4)]


def onehop_free_prompt(m, city):
    return m.fresh(Q_ONEHOP_FREE.format(city=city))


def twohop_prompt(m, city):
    return m.fresh(Q1.format(city=city))


def external_prompt(m, city, continent, opts):
    q = Q_EXTERNAL.format(city=city, continent=continent) + "\n" + options_text(opts) + TAIL_END
    return m.fresh(q) + m._enc(PREFILL)


def mc_tail(m, question, opts):
    return m.tail(question, opts)


# ---------------------------------------------------------------------------- rows and units
def nat_row(m, city, cont):
    ids, _, r0 = m.turn1(city, cont)
    return {"ids": ids, "r0": r0}


def ans_pos(m, city):
    return len(twohop_prompt(m, city))


def unit_natural(m, x, cont):
    return {"rows": [nat_row(m, x, cont)], "reqs": [], "tgt": 0, "kind": "nat", "ans_pos": ans_pos(m, x)}


def unit_swap(m, x, y, cont, layers):
    rx = nat_row(m, x, cont)
    rx["copy"] = {l: 0 for l in layers}
    return {"rows": [nat_row(m, y, cont), rx], "reqs": [], "tgt": 1, "kind": "swap", "layers": sorted(layers),
            "donor": 0, "ans_pos": ans_pos(m, x)}


def unit_add(m, x, cont, layer, vec):
    rx = nat_row(m, x, cont)
    rx["add"] = (layer, vec)
    return {"rows": [rx], "reqs": [], "tgt": 0, "kind": "add", "ans_pos": ans_pos(m, x)}


def with_clean(m, unit, x, cont):
    """Prepend the clean recipient row (for kl_units: rows[0] clean, rows[1] intervened)."""
    rows = [nat_row(m, x, cont)]
    for r in unit["rows"]:
        rr = dict(r)
        rr["copy"] = {l: s + 1 for l, s in r.get("copy", {}).items()}
        rows.append(rr)
    u = dict(unit, rows=rows, tgt=unit["tgt"] + 1)
    if unit["kind"] == "swap":  # kl_units compares rows 0 and 1: put the intervened row second
        u["rows"] = [rows[0], rows[2], rows[1]]
        u["rows"][1] = dict(rows[2], copy={l: 2 for l in unit["layers"]})
        u["tgt"], u["donor"] = 1, 2
    return u


def add_readout(m, unit, x, cont, cands):
    """Append the placeholder row and its eight readout requests (country x4, capital x4)."""
    t = unit["tgt"]
    ids, _, r0 = m.turn1(x, cont, placeholder=True)
    L = m.L
    if unit["kind"] == "swap":
        copy = {l: (unit["donor"] if l in unit["layers"] else t) for l in range(L)}
    else:
        copy = {l: t for l in range(L)}
    p = {"ids": ids, "r0": r0, "copy": copy}
    src = unit["rows"][t]
    if src.get("add") is not None:
        p["add"] = src["add"]
    rows = unit["rows"] + [p]
    pi = len(rows) - 1
    caps = [capital(c) for c in cands]
    reqs = list(unit["reqs"])
    for r in range(4):
        reqs.append((pi, m.tail(Q_READ_COUNTRY, rotate(cands, r)), ("rc", r)))
    for r in range(4):
        reqs.append((pi, m.tail(Q_READ_CAPITAL, rotate(caps, r)), ("rk", r)))
    return dict(unit, rows=rows, reqs=reqs)


def add_report(m, unit, cands, rotations=(0, 1, 2, 3)):
    t = unit["tgt"]
    reqs = list(unit["reqs"])
    for r in rotations:
        reqs.append((t, m.tail(Q_REPORT, rotate(cands, r)), ("rep", r)))
    return dict(unit, reqs=reqs)


def add_mc(m, unit, question, opts, rotations=(0, 1, 2, 3)):
    t = unit["tgt"]
    reqs = list(unit["reqs"])
    for r in rotations:
        reqs.append((t, m.tail(question, rotate(opts, r)), ("mc", r)))
    return dict(unit, reqs=reqs)


def strip_tensors(unit):
    """A JSON-safe copy of a unit's description (no ids, no vectors)."""
    return {k: v for k, v in unit.items() if k not in ("rows", "reqs")}


def readout_result(res, cands):
    """res: the unit's request results; the last eight are the readout (rc x4 then rk x4)."""
    from p9lib import rot_scores
    lp_c = [res[-8 + r][0] for r in range(4)]
    lp_k = [res[-4 + r][0] for r in range(4)]
    rc, rk = rot_scores(lp_c), rot_scores(lp_k)
    ic = max(range(4), key=lambda k: rc[k])
    ik = max(range(4), key=lambda k: rk[k])
    tot = [(a + b) / 2 for a, b in zip(rc, rk)]
    srt = sorted(tot, reverse=True)
    return {"rc": rc, "rk": rk, "cb_country": cands[ic], "cb_capital": cands[ik], "reliable": ic == ik,
            "cb": cands[ic] if ic == ik else None, "margin": srt[0] - srt[1], "lp_country": lp_c, "lp_capital": lp_k}


def report_result(res, cands, first=0, n=4):
    from p9lib import rot_scores
    s = rot_scores([res[first + r][0] for r in range(n)])
    k = max(range(4), key=lambda i: s[i])
    return {"scores": s, "pick": cands[k]}

"""Paper 9 stage 0 (before the freeze): screen every city. No report, no training, no readout.

For each city: number of tokens, the 2-hop continent answer (forced choice over the five continent first
tokens at the answer position), the free 1-hop country answer, the four candidate countries, and the
fresh-chat 1-hop forced choice over them (four rotations). Output: res/screen.json.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p9lib import P9, CONT_FIRST, TINY, country_parse, log, outdir, rot_scores, save_json  # noqa: E402
from build import (cont_of, fill, gaz, onehop_fc_prompts, onehop_free_prompt, split_cities,  # noqa: E402
                   twohop_prompt)
from cities import CITIES  # noqa: E402

ROOT = os.path.dirname(HERE)


def main():
    out_path = os.path.join(outdir("res"), "screen.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        log("exists, skipping", out_path)
        return
    t0 = time.time()
    m = P9()
    sp = split_cities()
    conts = list(CONT_FIRST.keys())
    items = []
    tiny_c = ["France", "Germany", "Italy", "Spain", "Poland", "Netherlands", "United Kingdom", "Sweden"]
    for c in (tiny_c if TINY else sorted(CITIES)):
        for x in sorted(CITIES[c])[:(14 if TINY else None)]:
            it = {"city": x, "country": c, "continent": cont_of(c), "split": sp[x],
                  "n": len(m._enc(" " + x))}
            it["ok"] = it["n"] in m.ph
            if it["ok"]:
                try:
                    m.turn1(x, it["continent"])
                    m.turn1(x, it["continent"], placeholder=True)
                except AssertionError:
                    it["ok"] = False
            items.append(it)
    log(len(items), "cities;", sum(i["ok"] for i in items), "tokenise cleanly with a placeholder")
    good = [i for i in items if i["ok"]]
    # 2-hop
    res = m.score_fresh([twohop_prompt(m, i["city"]) for i in good])
    for i, (lp4, lc, am) in zip(good, res):
        k = max(range(5), key=lambda j: lc[j])
        i["cont_lp"] = lc
        i["cont_pred"] = conts[k]
        i["acc2"] = conts[k] == i["continent"]
        srt = sorted(lc, reverse=True)
        i["cont_margin"] = srt[0] - srt[1]
        i["top_token"] = m.tok.decode([am])
    log("2-hop accuracy", sum(i["acc2"] for i in good) / len(good))
    # free 1-hop
    texts = m.generate_free([onehop_free_prompt(m, i["city"]) for i in good], n_new=10)
    for i, t in zip(good, texts):
        i["free1_text"] = t
        i["free1"] = country_parse(t)
    log("free 1-hop = gazetteer", sum(i["free1"] == i["country"] for i in good) / len(good))
    # candidates and the fresh 1-hop forced choice
    prompts = []
    for i in good:
        f1 = i["free1"] if i["free1"] and cont_of(i["free1"]) == i["continent"] else None
        i["cands"] = fill([i["country"], f1], i["city"])
        prompts += onehop_fc_prompts(m, i["city"], i["cands"])
    res = m.score_fresh(prompts)
    for k, i in enumerate(good):
        s = rot_scores([res[4 * k + r][0] for r in range(4)])
        j = max(range(4), key=lambda q: s[q])
        i["p1_scores"] = s
        i["p1"] = i["cands"][j]
        srt = sorted(s, reverse=True)
        i["p1_margin"] = srt[0] - srt[1]
    log("1-hop forced choice = gazetteer", sum(i["p1"] == i["country"] for i in good) / len(good))
    save_json({"items": items, "minutes": (time.time() - t0) / 60}, out_path)
    log("wrote", out_path)


if __name__ == "__main__":
    main()

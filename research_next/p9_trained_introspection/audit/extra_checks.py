"""Auditor's supplementary checks (no study modules): exact McNemar tests on discordant cells and the
composition of the v1 pilot's known-donor pool. Reads audit/recomputed_v1v2.json and the raw files."""
import json, os
from math import comb
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load = lambda rel: json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))  # noqa: E731


def mcnemar_exact(b, c):
    n = b + c
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


R = load("audit/recomputed_v1v2.json")
t = R["v2"]["A"]["R_vs_prior_2x2"]
print("Part A R vs prior discordant:", t["R_only"], "vs", t["prior_only"], "exact McNemar p =",
      round(mcnemar_exact(t["R_only"], t["prior_only"]), 4))
pb = R["v2"]["agreement"]["per_B"]
lo = [b for b in ("4", "6", "8", "9", "10", "11")]
hi = [b for b in ("12", "14", "16", "20")]
for name, Bs in (("B 4-11", lo), ("B 12-20", hi)):
    a = sum(pb[b]["rep_donor_ro_not"] for b in Bs)
    c = sum(pb[b]["ro_donor_rep_not"] for b in Bs)
    print(f"Part B {name}: report names donor without readout {a}, readout without report {c}, exact McNemar p =",
          round(mcnemar_exact(a, c), 5))
disc = sum(round((1 - pb[b]["observed"]) * 200) for b in pb)
print("Part B discordant rows (any pick) over all 2000:", disc)

# v1 pilot known-donor pool (is_known as frozen: usable, p1 == free1 == gazetteer, reliable CB == gazetteer)
scr = {i["city"]: i for i in load("res/screen.json")["items"]}
per = load("res_pilot/pilot.json")["P1"]["per_item"]
known = [c for c in per if scr[c]["p1"] == scr[c]["country"] and scr[c]["free1"] == scr[c]["country"]
         and per[c]["reliable"] and per[c]["cb_country"] == scr[c]["country"]]
print("v1 pilot known donors:", len(known), "countries:", Counter(scr[c]["country"] for c in known).most_common())
print("their continents:", Counter(scr[c]["continent"] for c in known).most_common())
# what the (unreliable-inclusive) natural readout names most often among pilot cities
print("most frequent natural CB country picks (all 194 pilot):", Counter(per[c]["cb_country"] for c in per).most_common(8))

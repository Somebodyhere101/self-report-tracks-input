import json, os, sys
R = sys.argv[1]
M = json.load(open(os.path.join(R, "audit", "recomputed_v3.json"), encoding="utf-8"))
A = json.load(open(os.path.join(R, "res_v3", "analysis.json"), encoding="utf-8"))
pairs = []
for arm in "NOIJ":
    pairs.append(("T3 " + arm, M["T3"][arm]["ci"], A["T3"][arm]["ci"]))
    pairs.append(("T1 " + arm, M["T1"][arm]["ci_diff"], A["T1"][arm]["ci_diff"]))
    pairs.append(("T2 " + arm, M["T2"][arm]["pooled_ci"], A["T2"][arm]["pooled_ci"]))
    for k in ("kl", "norm"):
        pairs.append((f"T4 {arm} {k}", M["T4"][arm][k]["ci"], A["T4"][arm][k]["ci"]))
mx = max((abs(a - b), n) for n, m, r in pairs for a, b in zip(m, r))
print("CI bounds compared:", 2 * len(pairs), "max abs diff:", round(mx[0], 4), "at", mx[1])

import json, os, statistics as st, sys
R = sys.argv[1]
G = json.load(open(os.path.join(R, "res_v3", "gen.json"), encoding="utf-8"))
T = G["tests"]
print("base_norm main", G["cfg"]["base_norm"])
for name in ("T1", "T2a", "T2c"):
    print(name, "median kl", round(st.median(e["kl"] for e in T[name]), 4), "kept", sum(e["answer_kept"] for e in T[name]) / len(T[name]))
for kind in ("kl", "norm"):
    rows = [e for e in T["T4"] if e["kind"] == kind]
    print("T4", kind, "median kl", round(st.median(e["kl"] for e in rows), 4), "kept", sum(e["answer_kept"] for e in rows) / len(rows))
d = [e for e in T["T1"] if e["kind"] == "diff"]
print("T1 diff median kl", round(st.median(e["kl"] for e in d), 4))

"""Auditor: print the structure of the v3 raw files (keys, lengths, first rows). Read-only."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "res_v3")


def load(n):
    return json.load(open(os.path.join(R, n), encoding="utf-8"))


def show(x, depth=0, maxd=3):
    pad = "  " * depth
    if isinstance(x, dict):
        for k in list(x)[:12]:
            v = x[k]
            if isinstance(v, (dict, list)) and depth < maxd:
                print(f"{pad}{k}: {type(v).__name__}[{len(v)}]")
                show(v, depth + 1, maxd)
            else:
                s = json.dumps(v)
                print(f"{pad}{k}: {s[:150]}")
        if len(x) > 12:
            print(f"{pad}... ({len(x)} keys)")
    elif isinstance(x, list):
        if x:
            print(f"{pad}[0] ->")
            show(x[0], depth + 1, maxd) if isinstance(x[0], (dict, list)) else print(f"{pad}  {json.dumps(x[0])[:150]}")


scr = load("screen.json")
print("screen.json top keys:", list(scr), "accuracy", scr.get("accuracy"))
items = scr["items"]
print("n items", len(items))
k0 = list(items)[0]
print("first item", json.dumps(items[k0]))
G = load("gen.json")
print("\ngen.json top keys:", list(G))
for k in G:
    if k in ("tests", "train", "items"):
        continue
    print(k, json.dumps(G[k])[:400])
print("tests:", {k: len(v) for k, v in G["tests"].items()})
for k, v in G["tests"].items():
    print(" ", k, json.dumps(v[0])[:300])
print("train:", {k: len(v) for k, v in G["train"].items()})
for k, v in G["train"].items():
    print(" ", k, json.dumps(v[0])[:300], json.dumps(v[1])[:300])
print("items:", len(G["items"]), json.dumps(list(G["items"].items())[0]))
for f in sorted(os.listdir(R)):
    if f.startswith("eval_"):
        E = load(f)
        print("\n", f, {k: (len(v) if isinstance(v, list) else v) for k, v in E.items()})
        for k, v in E.items():
            if isinstance(v, list) and v:
                print("   ", k, json.dumps(v[0])[:400])
for f in sorted(os.listdir(R)):
    if f.startswith("trainlog_"):
        print(f, json.dumps(load(f))[:900])

"""Write (freeze) or check PREDICTIONS.sha256: SHA-256 of the preregistration, the docs it cites, every code and
config file of paper 9, and the six files it imports read-only from paper 8 (in-batch engine) and S2 (LoRA).
python freeze.py write | python freeze.py check

A file changed after the freeze must be logged in DEVIATIONS.md, and its new hash added to
PREDICTIONS.sha256.amend as "<sha256>  <path>  <deviation timestamp>". check accepts an amended hash for any
file except PREDICTIONS.md, which must never change."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WS = os.path.abspath(os.path.join(ROOT, "..", ".."))
OUT = os.path.join(ROOT, "PREDICTIONS.sha256")
AMEND = OUT + ".amend"
DOCS = ["PREDICTIONS.md", "NOVELTY.md", "README.md"]
DEPS = ["WS:paper8/selfreport/srlib.py", "WS:paper8/patch2/patchlib2.py", "WS:paper8/patch/patchlib.py",
        "WS:paper8/code/p8lib.py", "WS:paper8/code/p6lib.py", "WS:research_next/s2_monitors/code/s2lib.py"]
CODE_EXT = (".py", ".sh", ".json")


def files():
    fs = list(DOCS)
    for f in sorted(os.listdir(HERE)):
        if f.endswith(CODE_EXT) and f != "freeze.py":
            fs.append("code/" + f)
    return fs + ["code/freeze.py"] + DEPS


def resolve(p):
    return os.path.join(WS, p[3:]) if p.startswith("WS:") else os.path.join(ROOT, p)


def sha(p):
    with open(resolve(p), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check(verbose=True):
    amended = {}
    if os.path.exists(AMEND):
        for line in open(AMEND, encoding="utf-8"):
            if line.strip():
                h, p = line.split()[:2]
                amended.setdefault(p, set()).add(h)
    bad = 0
    for line in open(OUT, encoding="utf-8"):
        h, p = line.strip().split("  ", 1)
        if not os.path.exists(resolve(p)):
            ok, tag = False, "MISSING"
        else:
            cur = sha(p)
            ok = cur == h or (p != "PREDICTIONS.md" and cur in amended.get(p, ()))
            tag = "OK  " if cur == h else ("AMND" if ok else "BAD ")
        bad += not ok
        if verbose:
            print(tag, p)
    if verbose:
        print("all match" if not bad else f"{bad} mismatch(es)")
    return bad == 0


def main():
    if sys.argv[1] == "write":
        assert not os.path.exists(OUT), "already frozen"
        with open(OUT, "w", encoding="utf-8", newline="\n") as f:
            for p in files():
                f.write(f"{sha(p)}  {p}\n")
        print(open(OUT, encoding="utf-8").read())
    else:
        sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()

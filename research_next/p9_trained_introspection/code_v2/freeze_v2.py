"""Write or check PREDICTIONS_V2.sha256 (v2 preregistration, v2 code, and the frozen v1 code it imports).
python freeze_v2.py write | check"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WS = os.path.abspath(os.path.join(ROOT, "..", ".."))
OUT = os.path.join(ROOT, "PREDICTIONS_V2.sha256")
FILES = ["PREDICTIONS_V2.md", "code_v2/v2_premise.py", "code_v2/analyze_v2.py", "code_v2/job_v2.py",
         "code_v2/run_v2.sh", "code_v2/freeze_v2.py", "code/p9lib.py", "code/build.py", "code/cities.py",
         "WS:paper8/selfreport/srlib.py", "WS:paper8/patch2/patchlib2.py", "WS:paper8/patch/patchlib.py",
         "WS:paper8/code/p8lib.py", "WS:paper8/code/p6lib.py", "WS:research_next/s2_monitors/code/s2lib.py"]


def resolve(p):
    return os.path.join(WS, p[3:]) if p.startswith("WS:") else os.path.join(ROOT, p)


def sha(p):
    return hashlib.sha256(open(resolve(p), "rb").read()).hexdigest()


def check(verbose=True):
    if not os.path.exists(OUT):
        return False
    bad = 0
    for line in open(OUT, encoding="utf-8"):
        h, p = line.strip().split("  ", 1)
        ok = os.path.exists(resolve(p)) and sha(p) == h
        bad += not ok
        if verbose:
            print("OK  " if ok else "BAD ", p)
    if verbose:
        print("all match" if not bad else f"{bad} mismatch(es)")
    return bad == 0


if __name__ == "__main__":
    if sys.argv[1] == "write":
        assert not os.path.exists(OUT), "already frozen"
        with open(OUT, "w", encoding="utf-8", newline="\n") as f:
            for p in FILES:
                f.write(f"{sha(p)}  {p}\n")
        print(open(OUT, encoding="utf-8").read())
    else:
        sys.exit(0 if check() else 1)

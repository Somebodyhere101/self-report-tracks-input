"""Write or check PREDICTIONS_V3.sha256 (v3 preregistration, v3 code, and the frozen v1 files it imports)."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WS = os.path.abspath(os.path.join(ROOT, "..", ".."))
OUT = os.path.join(ROOT, "PREDICTIONS_V3.sha256")
FILES = ["PREDICTIONS_V3.md", "code_v3/v3lib.py", "code_v3/v3_run.py", "code_v3/trainlib_v3.py", "code_v3/items_v3.py",
         "code_v3/v3_analyze.py", "code_v3/job_v3.py", "code_v3/make_config_v3.py", "code_v3/run_v3.sh",
         "code_v3/freeze_v3.py", "code/p9lib.py", "code/analyze.py", "code/mcq.py", "code/cities.py",
         "WS:paper8/selfreport/srlib.py", "WS:paper8/patch2/patchlib2.py", "WS:paper8/patch/patchlib.py",
         "WS:paper8/code/p8lib.py", "WS:paper8/code/p6lib.py", "WS:research_next/s2_monitors/code/s2lib.py"]


def resolve(p):
    return os.path.join(WS, p[3:]) if p.startswith("WS:") else os.path.join(ROOT, p)


def sha(p):
    return hashlib.sha256(open(resolve(p), "rb").read()).hexdigest()


AMEND = OUT + ".amend"  # "<sha256>  <path>  <DEVIATIONS.md timestamp>" for files changed after the freeze


def check(verbose=True):
    if not os.path.exists(OUT):
        return False
    amended = {}
    if os.path.exists(AMEND):
        for line in open(AMEND, encoding="utf-8"):
            if line.strip():
                h, p = line.split()[:2]
                amended.setdefault(p, set()).add(h)
    bad = 0
    for line in open(OUT, encoding="utf-8"):
        h, p = line.strip().split("  ", 1)
        ok = os.path.exists(resolve(p)) and (sha(p) == h or (p != "PREDICTIONS_V3.md" and sha(p) in amended.get(p, ())))
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

"""Auditor's freeze/amendment check for v3 (independent of code_v3/freeze_v3.py).
- hashes every file listed in PREDICTIONS_V3.sha256 and says whether it matches the frozen hash or an amended hash
- reads the pre-change bytecode caches (code_v3/__pycache__, written 23:34:53-54, before the 23:36:19 edits)
  to recover the original source size/mtime and disassemble the original gen() and freeze module
- tries to reconstruct the frozen v3_run.py by reverting the logged change, and hashes it
Read-only on the study files; prints a report."""
import dis
import hashlib
import io
import marshal
import os
import struct
import sys
import time
import types
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.abspath(os.path.join(ROOT, "..", ".."))


def resolve(p):
    return os.path.join(WS, p[3:]) if p.startswith("WS:") else os.path.join(ROOT, p)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    frozen = [l.strip().split("  ", 1) for l in open(os.path.join(ROOT, "PREDICTIONS_V3.sha256"), encoding="utf-8") if l.strip()]
    amend = [l.split() for l in open(os.path.join(ROOT, "PREDICTIONS_V3.sha256.amend"), encoding="utf-8") if l.strip()]
    print("amend file lines:")
    for a in amend:
        print("  ", a)
    amended = {}
    for a in amend:
        amended.setdefault(a[1], set()).add(a[0])
    print("\nper-file status:")
    for h, p in frozen:
        b = open(resolve(p), "rb").read()
        s = sha_bytes(b)
        st = "FROZEN" if s == h else ("AMENDED" if s in amended.get(p, ()) else "MISMATCH")
        mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(resolve(p))))
        print(f"  {st:8s} {p}  (size {len(b)}, mtime {mt})")
    print("\nfiles named in the amend file but not in the freeze list:",
          [p for p in amended if p not in {p for _, p in frozen}])

    # ---- bytecode caches written before the edits
    for name in ("v3_run", "freeze_v3"):
        pyc = os.path.join(ROOT, "code_v3", "__pycache__", f"{name}.cpython-312.pyc")
        raw = open(pyc, "rb").read()
        flags = struct.unpack("<I", raw[4:8])[0]
        mtime, size = struct.unpack("<II", raw[8:16])
        cur = os.path.join(ROOT, "code_v3", f"{name}.py")
        print(f"\n{name}.pyc: flags {flags}, source mtime {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}, "
              f"source size {size}; current source size {os.path.getsize(cur)}")
        code = marshal.loads(raw[16:])
        names = set()

        def walk(c):
            names.update(c.co_names)
            for k in c.co_consts:
                if isinstance(k, types.CodeType):
                    walk(k)
        walk(code)
        if name == "freeze_v3":
            print("  original module names include AMEND:", "AMEND" in names, "; 'amended' in any code:",
                  any("amended" in (getattr(k, "co_varnames", ()) or ()) for k in [code] + [c for c in code.co_consts if isinstance(c, types.CodeType)]))
            consts = [c for c in code.co_consts if isinstance(c, types.CodeType)]
            for c in consts:
                if c.co_name == "check":
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        dis.dis(c)
                    print("  original check() varnames:", c.co_varnames)
        else:
            gen = [c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == "gen"][0]
            # find the dict comprehension building G["items"]; print the names it loads
            buf = io.StringIO()
            with redirect_stdout(buf):
                dis.dis(gen)
            txt = buf.getvalue()
            i = txt.find("ent_bin")
            seg = txt[i - 200: i + 1600] if i >= 0 else ""
            uses_smoke_near_items = "SMOKE" in txt[txt.find("'para_bin'"):txt.find("'para_bin'") + 2500] if "'para_bin'" in txt else None
            print("  original gen(): SMOKE referenced in the items-table region:", uses_smoke_near_items)
            # show the compare ops of the region
            region = txt[txt.find("'para_bin'"):txt.find("'para_bin'") + 2500]
            lines = [l for l in region.splitlines() if any(t in l for t in ("LOAD_GLOBAL", "LOAD_CONST", "CONTAINS_OP", "POP_JUMP", "COMPARE_OP", "BINARY_SUBSCR", "LOAD_FAST", "LOAD_DEREF", "STORE_SUBSCR", "LOAD_ATTR"))]
            print("  original items-table bytecode (abridged):")
            for l in lines[:60]:
                print("    ", l.strip()[:110])

    # ---- reconstruct the frozen v3_run.py by reverting the logged change
    src = open(os.path.join(ROOT, "code_v3", "v3_run.py"), "rb").read()
    new = b'if v["correct"] and (v["split"] in ("train", "test") or SMOKE)}'
    assert src.count(new) == 1
    for cand in (b'if v["correct"] and v["split"] in ("train", "test")}',):
        rec = src.replace(new, cand)
        print("\nreconstructed v3_run.py (revert '(... or SMOKE)'):", len(rec), "bytes, sha256", sha_bytes(rec),
              "matches frozen:", sha_bytes(rec) == dict((p, h) for h, p in frozen)["code_v3/v3_run.py"])


if __name__ == "__main__":
    sys.exit(main())

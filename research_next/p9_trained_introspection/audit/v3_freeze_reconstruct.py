import hashlib, os, sys
R = sys.argv[1]
raw = open(os.path.join(R, "code_v3", "freeze_v3.py"), "rb").read()
src = raw.decode("utf-8")
target = "b79cb7cb158c11fe5c533ef268d1674e522fefbd971b38125fcefae3992a79ac"
nl = "\r\n"
L = lambda *lines: "".join(l + nl for l in lines)
amend_def = L('AMEND = OUT + ".amend"  # "<sha256>  <path>  <DEVIATIONS.md timestamp>" for files changed after the freeze', "", "")
amended_blk = L('    amended = {}', '    if os.path.exists(AMEND):', '        for line in open(AMEND, encoding="utf-8"):',
                '            if line.strip():', '                h, p = line.split()[:2]', '                amended.setdefault(p, set()).add(h)')
ok_new = L('        ok = os.path.exists(resolve(p)) and (sha(p) == h or (p != "PREDICTIONS_V3.md" and sha(p) in amended.get(p, ())))')
ok_old = L('        ok = os.path.exists(resolve(p)) and sha(p) == h')
rec = src.replace(amend_def, "").replace(amended_blk, "").replace(ok_new, ok_old)
for tag, text in (("CRLF", rec), ("LF", rec.replace("\r\n", "\n"))):
    b = text.encode("utf-8")
    h = hashlib.sha256(b).hexdigest()
    print(tag, "size", len(b), "sha", h[:16], "matches frozen hash:", h == target)
# other code_v3 files: line endings now
for f in sorted(os.listdir(os.path.join(R, "code_v3"))):
    p = os.path.join(R, "code_v3", f)
    if os.path.isfile(p):
        b = open(p, "rb").read()
        print(f"{f:20s} CRLF {b.count(b'\r\n'):4d} LF {b.count(b'\n'):4d}")

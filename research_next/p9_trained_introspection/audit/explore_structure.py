"""Auditor helper: print the key structure of the raw JSON files (no study modules imported)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def shape(x, depth=0, maxdepth=4, maxkeys=40):
    pad = "  " * depth
    if isinstance(x, dict):
        keys = list(x.keys())
        out = f"dict[{len(keys)}]"
        if depth >= maxdepth:
            return out + " keys=" + str(keys[:8])
        lines = [out]
        for k in keys[:maxkeys]:
            lines.append(f"{pad}  {k!r}: " + shape(x[k], depth + 1, maxdepth, maxkeys))
        if len(keys) > maxkeys:
            lines.append(f"{pad}  ... ({len(keys)-maxkeys} more keys, e.g. {keys[maxkeys:maxkeys+5]})")
        return "\n".join(lines)
    if isinstance(x, list):
        if not x:
            return "list[0]"
        return f"list[{len(x)}] of " + shape(x[0], depth + 1, maxdepth, maxkeys)
    s = repr(x)
    return (type(x).__name__ + " " + s[:80])

for rel in sys.argv[1:]:
    p = os.path.join(ROOT, rel)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    print("=" * 20, rel)
    print(shape(d, maxdepth=int(os.environ.get("MAXD", "3")), maxkeys=int(os.environ.get("MAXK", "25"))))

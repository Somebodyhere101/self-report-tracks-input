"""Check every line of the three freeze hash files against the current files (read-only)."""
import hashlib, os, re, sys
os.chdir(sys.argv[1])
bad = []; n = 0
for hf in ['PREDICTIONS.sha256', 'PREDICTIONS_V2.sha256', 'PREDICTIONS_V3.sha256', 'PREDICTIONS_V3.sha256.amend']:
    for line in open(hf, encoding='utf-8'):
        parts = line.split()
        if len(parts) < 2 or not re.fullmatch(r'[0-9a-f]{64}', parts[0]): continue
        h, p = parts[0], parts[1].lstrip('*'); n += 1
        if not os.path.exists(p): bad.append((hf, p, 'MISSING (outside study folder?)')); continue
        if hashlib.sha256(open(p, 'rb').read()).hexdigest() != h: bad.append((hf, p, 'MISMATCH'))
print('checked', n); [print(b) for b in bad]

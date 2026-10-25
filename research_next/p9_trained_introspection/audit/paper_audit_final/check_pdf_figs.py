"""Compare figure XObjects embedded in main.pdf with the current fig_*.pdf page streams (CPU, pypdf only)."""
import sys, hashlib, glob, os
from pypdf import PdfReader
paper = sys.argv[1]
def h(b): return hashlib.sha256(b).hexdigest()[:16]
figs = {}
for f in sorted(glob.glob(os.path.join(paper, 'fig_*.pdf'))):
    r = PdfReader(f); p = r.pages[0]
    c = p.get_contents().get_data()
    figs[os.path.basename(f)] = h(c)
print('fig files:', figs)
r = PdfReader(os.path.join(paper, 'main.pdf'))
found = []
for i, p in enumerate(r.pages):
    res = p.get('/Resources') or {}
    xo = res.get('/XObject') if res else None
    if not xo: continue
    xo = xo.get_object()
    for k, v in xo.items():
        v = v.get_object()
        if v.get('/Subtype') == '/Form':
            d = v.get_data()
            found.append((i+1, k, h(d)))
for x in found: print('page', x[0], x[1], x[2], [n for n, hh in figs.items() if hh == x[2]])
print('n pages', len(r.pages))

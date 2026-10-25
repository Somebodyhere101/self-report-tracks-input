"""Build the paper 9 draft: assemble references.bib (the entries main.tex cites, taken from paper 8's checked bibliography
and from refs_new.bib, which a citation-check session verified), then pdflatex, bibtex, pdflatex x2. Reports undefined
citations and references, and any \\todo left in the text."""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P8BIB = os.path.join(WS, "paper8_ms", "references.bib")


def entries(path):
    if not os.path.exists(path):
        return {}
    s = open(path, encoding="utf-8").read()
    out = {}
    for e in re.split(r"\n(?=@)", s):
        m = re.match(r"\s*@\w+\{([^,]+),", e)
        if m:
            out[m.group(1).strip()] = e.strip() + "\n"
    return out


def main():
    tex = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
    keys = []
    for grp in re.findall(r"\\cite[a-z]*\{([^}]+)\}", tex):
        for k in grp.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    pool = entries(P8BIB)
    pool.update(entries(os.path.join(HERE, "refs_new.bib")))
    missing = [k for k in keys if k not in pool]
    with open(os.path.join(HERE, "references.bib"), "w", encoding="utf-8") as f:
        for k in keys:
            if k in pool:
                f.write(pool[k] + "\n")
    print(len(keys), "keys cited;", len(missing), "missing from the bibliographies:", missing)
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, errors="replace")
        if r.returncode != 0 and cmd[0] == "pdflatex":
            print("ERROR in", cmd[0])
            print("\n".join(l for l in r.stdout.splitlines() if l.startswith("!") or "Error" in l)[:3000])
            sys.exit(1)
    log = open(os.path.join(HERE, "main.log"), encoding="utf-8", errors="replace").read()
    und = sorted(set(re.findall(r"(?:Citation|Reference) `([^']+)' on page \d+ undefined", log)))
    print("undefined after build:", und)
    print("overfull boxes:", len(re.findall(r"Overfull \\hbox", log)))
    print("todo markers left:", tex.count("\\todo{") - 1)


if __name__ == "__main__":
    main()

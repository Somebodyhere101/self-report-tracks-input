"""Paper 9 manuscript audit: re-run the Stage 0a / 0b / tail-position probes from the stored features (CPU, 4 threads).
(1) exact re-run with Stage 0a's own probe code (stage0/s0a_probe.py, imported read-only) and the lambda recorded in each
    probe.json, for B2 (all cells), A at the first digit, B1 and C at Stage 0a's report position, plus a few lambda-selection
    re-runs; (2) the tail-position B2 grid (lambda 0.5, as s0b_tailpos.py); (3) an independent implementation
    (sklearn LogisticRegression, own group folds) of B2 at K = 2 for every source, position and layer.
Features are read from outside P9 (Temp\\p6\\stage0, Temp\\p9\\stage0b). Writes audit/paper_audit/reprobe.json."""
import importlib.util
import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import numpy as np
import torch

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
P9 = os.path.dirname(os.path.dirname(HERE))
TMP = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp")
spec = importlib.util.spec_from_file_location("s0a_probe_ro", os.path.join(P9, "stage0", "s0a_probe.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)  # defines Probe, folds, boot, cv_pred (no side effects beyond env defaults)
from sklearn.linear_model import LogisticRegression  # noqa: E402

SRC = {"0a": (os.path.join(TMP, "p6", "stage0", "feats.pt"), os.path.join(P9, "stage0")),
       "imm": (os.path.join(TMP, "p9", "stage0b", "feats_imm.pt"), os.path.join(P9, "stage0b", "probe_imm", "stage0")),
       "del": (os.path.join(TMP, "p9", "stage0b", "feats_del.pt"), os.path.join(P9, "stage0b", "probe_del", "stage0"))}
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
OUTP = os.path.join(HERE, f"reprobe_{MODE}.json")
out = {}
t0 = time.time()


def setup(rows, K):
    sw = np.array([r["kind"] == "swap" for r in rows])
    prod = np.array([r["product"] for r in rows])
    own = np.array([r[f"bin{K}_x"] for r in rows])
    eff = np.array([r[f"bin{K}_y"] if r["kind"] == "swap" else r[f"bin{K}_x"] for r in rows])
    return sw, prod, own, eff, sw & (own != eff)


def b2_exact(X, sw, prod, own, eff, diff, lam):
    allg = np.unique(prod)
    dpos = np.where(diff)[0]
    pred = np.full(diff.sum(), -1)
    for tr, te in S.folds(allg):
        trp, tep = set(allg[tr]), set(allg[te])
        mtr = (~sw) & np.isin(prod, list(trp))
        clf = S.fit(X[mtr], own[mtr], lam)
        sel = np.isin(prod[dpos], list(tep))
        pred[sel] = clf.predict(X[dpos[sel]])
    return float((pred == eff[diff]).mean()), float((pred == own[diff]).mean())


def b2_indep(X, sw, prod, own, eff, diff, seed=0, C=0.01):
    """Independent: sklearn L2 logistic regression on standardised features, random 5-fold split of products."""
    rng = np.random.default_rng(seed)
    allg = np.unique(prod)
    perm = rng.permutation(allg)
    fold = {g: i % 5 for i, g in enumerate(perm)}
    fid = np.array([fold[g] for g in prod])
    pred = np.full(len(prod), -1)
    for f in range(5):
        mtr = (~sw) & (fid != f)
        mu, sd = X[mtr].mean(0), X[mtr].std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=2000).fit((X[mtr] - mu) / sd, own[mtr])
        mte = diff & (fid == f)
        pred[mte] = clf.predict((X[mte] - mu) / sd)
    return float((pred[diff] == eff[diff]).mean()), float((pred[diff] == own[diff]).mean())


if MODE in ("all", "exact", "indep"):
    for name, (fp, rdir) in SRC.items():
        rows = json.load(open(os.path.join(rdir, "rows.json"), encoding="utf-8"))["rows"]
        PJ = json.load(open(os.path.join(rdir, "probe.json"), encoding="utf-8"))["res"]
        F = torch.load(fp)["feats"]
        for pos in ("rep", "R0"):
            for l in (8, 12, 16, 20, 24, 27):
                X = F[pos][l].float().numpy()
                for K in (2, 4):
                    sw, prod, own, eff, diff = setup(rows, K)
                    tag = f"{name}_{pos}_L{l}_K{K}"
                    lam = PJ[f"{pos}_L{l}_K{K}"]["C"]
                    rec = {"lam_file": lam}
                    if MODE in ("all", "exact"):
                        d, r = b2_exact(X, sw, prod, own, eff, diff, lam)
                        rec.update({"B2_donor": d, "B2_recipient": r, "B2_donor_file": PJ[f"{pos}_L{l}_K{K}"]["B2"]["acc"]})
                        if pos == "R0" and l == 27:
                            pa = S.cv_pred(X, eff, prod, lam)
                            rec["A"] = float((pa == eff).mean())
                            rec["A_file"] = PJ[f"{pos}_L{l}_K{K}"]["A"]["acc"]
                        if name == "0a" and pos == "rep" and K == 2:
                            xnat = {r_["x"]: i for i, r_ in enumerate(rows) if r_["kind"] == "nat"}
                            xi = np.array([xnat[r_["x"]] for r_ in rows])
                            pb1 = S.cv_pred(X[diff], eff[diff], prod[diff], lam)
                            pc = S.cv_pred(X[xi[diff]], eff[diff], prod[diff], lam)
                            rec.update({"B1": float((pb1 == eff[diff]).mean()), "B1_file": PJ[f"{pos}_L{l}_K{K}"]["B1"]["acc"],
                                        "C": float((pc == eff[diff]).mean()), "C_file": PJ[f"{pos}_L{l}_K{K}"]["C_input_probe"]["acc"]})
                        if (name, pos, l, K) in (("imm", "rep", 20, 2), ("del", "rep", 20, 2), ("imm", "R0", 8, 2), ("0a", "rep", 16, 2)):
                            Xn, yn, gn = X[~sw], own[~sw], prod[~sw]
                            sc = {c: float((S.cv_pred(Xn, yn, gn, c) == yn).mean()) for c in S.CS}
                            rec["lam_reselected"] = max(S.CS, key=lambda c: sc[c])
                            rec["lam_scores"] = sc
                            rec["B2_donor_at_lam0.5"] = b2_exact(X, sw, prod, own, eff, diff, 0.5)[0]
                    if MODE in ("all", "indep") and K == 2:
                        rec["B2_donor_indep"], rec["B2_recipient_indep"] = b2_indep(X, sw, prod, own, eff, diff)
                    out[tag] = rec
                    print(f"{time.time() - t0:7.1f}s {tag} {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k != 'lam_scores'})}", flush=True)
                    json.dump(out, open(OUTP, "w", encoding="utf-8"), indent=1)
        del F

if MODE in ("all", "tail"):
    rows = json.load(open(os.path.join(P9, "stage0b", "probe_imm", "stage0", "rows.json"), encoding="utf-8"))["rows"]
    D = torch.load(os.path.join(TMP, "p9", "stage0b", "feats_tailpos.pt"))
    Ft, layers = D["feats"], D["layers"]
    sw, prod, own, eff, diff = setup(rows, 2)
    tj = json.load(open(os.path.join(P9, "stage0b", "tailpos.json"), encoding="utf-8"))
    tail = {"tokens": D["tail_tokens"], "layers": layers, "B2_donor": {}, "maxabs_diff_vs_file": 0.0}
    for j, l in enumerate(layers):
        for p in range(Ft.shape[1]):
            X = Ft[:, p, j].float().numpy()
            d, r = b2_exact(X, sw, prod, own, eff, diff, 0.5)
            tail["B2_donor"][f"L{l}_p{p}"] = d
            tail["maxabs_diff_vs_file"] = max(tail["maxabs_diff_vs_file"], abs(d - tj["B2_donor"][f"L{l}_p{p}"]))
        print(f"{time.time() - t0:7.1f}s tail L{l} done; maxabs vs file {tail['maxabs_diff_vs_file']}", flush=True)
    # independent check on a subset of positions (d4, first three tail tokens, last token), K = 2
    tail["indep"] = {}
    for j, l in enumerate(layers):
        for p in (0, 1, 2, 3, Ft.shape[1] - 1):
            X = Ft[:, p, j].float().numpy()
            tail["indep"][f"L{l}_p{p}"] = b2_indep(X, sw, prod, own, eff, diff)[0]
    out["tailpos"] = tail
    json.dump(out, open(OUTP, "w", encoding="utf-8"), indent=1)
print("done", round((time.time() - t0) / 60, 1), "min", flush=True)

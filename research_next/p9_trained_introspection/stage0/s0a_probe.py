"""Stage 0a, CPU part: linear probes on the features from s0a_extract.py. Writes stage0/probe.json.

Settings, per position (rep = last turn-2 prompt token; R0 = first answer-digit position), layer, K in {2, 4}:
  A    all rows (natural + swapped), label = effective bin (own bin; donor bin for swapped rows), grouped 5-fold CV
  B1   swapped-only rows whose donor bin != recipient bin (at this K), label = donor bin, grouped 5-fold CV
  B2   transfer: probe trained on NATURAL rows of the training-fold products, tested on B1 rows of the held-out
       products; reports agreement with the donor bin (label) and with the recipient's own bin
  C    input-only probe control for B1: the same CV and labels, but the features are the RECIPIENT's unswapped
       (natural-row) features; for K = 2 the label is 1 - recipient bin, so this is the inversion confound
Groups = product (5 group folds, greedy size-balanced). L2 multinomial logistic regression on standardised
features (torch, full-batch LBFGS; sklearn's compiled extensions are blocked by Application Control on this
machine); loss = mean CE + lam * ||W||^2, lam picked once per (position, layer, K) by grouped CV on natural
rows only (grid 0.5, 0.05, 0.005, roughly sklearn C = 0.001 / 0.01 / 0.1 at n ~ 1000). CI = 95% cluster bootstrap
over products (1000 reps) of the out-of-fold predictions."""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import torch

torch.set_num_threads(4)
P9DIR = os.environ.get("P9DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(P9DIR, "stage0")
FEAT = os.environ.get("S0_FEAT", r"C:\Users\<user>\AppData\Local\Temp\p6\stage0\feats.pt")
CS = [0.5, 0.05, 0.005]  # lam


class Probe:
    def __init__(self, lam):
        self.lam = lam

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        k = int(y.max()) + 1
        self.k = max(k, 2)
        yt = torch.tensor(y, dtype=torch.long)
        W = torch.zeros(Z.shape[1], self.k, requires_grad=True)
        b = torch.zeros(self.k, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Z @ W + b, yt) + self.lam * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.W, self.b = W.detach(), b.detach()
        return self

    def predict(self, X):
        Z = (torch.tensor(X, dtype=torch.float32) - self.mu) / self.sd
        return (Z @ self.W + self.b).argmax(-1).numpy()


def fit(X, y, C):
    return Probe(C).fit(X, y)


def folds(groups, n=5):
    """Group k-fold: groups sorted by size (desc, then id), each to the currently smallest fold."""
    ug, cnt = np.unique(groups, return_counts=True)
    order = sorted(range(len(ug)), key=lambda i: (-cnt[i], ug[i]))
    load, fold_of = [0] * n, {}
    for i in order:
        f = int(np.argmin(load))
        fold_of[ug[i]] = f
        load[f] += cnt[i]
    fid = np.array([fold_of[g] for g in groups])
    return [(np.where(fid != f)[0], np.where(fid == f)[0]) for f in range(n)]


def boot(correct, groups, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(reps):
        s = rng.choice(ug, len(ug))
        ii = np.concatenate([idx[g] for g in s])
        vals.append(correct[ii].mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def cv_pred(X, y, g, C):
    pred = np.full(len(y), -1)
    for tr, te in folds(g):
        if len(np.unique(y[tr])) < 2:
            pred[te] = np.bincount(y[tr]).argmax()
            continue
        pred[te] = fit(X[tr], y[tr], C).predict(X[te])
    return pred


def summary(pred, y, g, k, extra=None):
    c = (pred == y).astype(float)
    d = {"n": int(len(y)), "acc": float(c.mean()), "ci": boot(c, g), "chance": 1.0 / k,
         "majority": float(np.bincount(y, minlength=k).max() / len(y))}
    if extra:
        d.update(extra)
    return d


def main():
    R = json.load(open(os.path.join(OUT, "rows.json"), encoding="utf-8"))
    rows = R["rows"]
    F = torch.load(FEAT)["feats"]
    layers = R["layers"]
    kind = np.array([r["kind"] for r in rows])
    prod = np.array([r["product"] for r in rows])
    nat_idx = {r["x"]: i for i, r in enumerate(rows) if r["kind"] == "nat"}
    xnat = np.array([nat_idx[r["x"]] for r in rows])
    out = {"checks": {}, "res": {}}
    sw = kind == "swap"
    out["checks"]["n_nat"] = int((~sw).sum())
    out["checks"]["n_swap"] = int(sw.sum())
    out["checks"]["swap_lp2_equals_donor_maxabs"] = float(max(abs(r["lp2_meas"] - r["lp2_y"]) for r in rows if r["kind"] == "swap"))
    out["checks"]["nat_lp2_equals_screen_maxabs"] = float(max(abs(r["lp2_meas"] - r["lp2_x"]) for r in rows if r["kind"] == "nat"))
    out["checks"]["swap_digits_kept"] = float(np.mean([r["correct_meas"] for r in rows if r["kind"] == "swap"]))
    for K in (2, 4):
        own = np.array([r[f"bin{K}_x"] for r in rows])
        eff = np.array([r[f"bin{K}_y"] if r["kind"] == "swap" else r[f"bin{K}_x"] for r in rows])
        meas = np.array([r[f"bin{K}_meas"] for r in rows])
        diff = sw & (own != eff)
        out["checks"][f"K{K}_measured_bin_equals_effective"] = float((meas == eff).mean())
        out["checks"][f"K{K}_n_swap_diff"] = int(diff.sum())
        out["checks"][f"K{K}_input_only_own_eq_label_on_B"] = float((own[diff] == eff[diff]).mean())
        out["checks"][f"K{K}_label_counts_nat"] = np.bincount(own[~sw], minlength=K).tolist()
        out["checks"][f"K{K}_label_counts_B"] = np.bincount(eff[diff], minlength=K).tolist()
        # base-model report (no adapter), K = 2 only: P(A = "Very sure") > 0.5 -> bin 0
        if K == 2:
            rep = np.array([0 if r["rep_lpA"] > np.log(0.5) else 1 for r in rows])
            out["checks"]["base_report_acc_nat"] = float((rep[~sw] == eff[~sw]).mean())
            out["checks"]["base_report_acc_B"] = float((rep[diff] == eff[diff]).mean())
            out["checks"]["base_report_frac_A"] = float((rep == 0).mean())
        for pos in ("rep", "R0"):
            for l in layers:
                X = F[pos][l].float().numpy()
                tag = f"{pos}_L{l}_K{K}"
                print(tag, flush=True)
                # C on natural rows only
                Xn, yn, gn = X[~sw], own[~sw], prod[~sw]
                best = max(CS, key=lambda C: (cv_pred(Xn, yn, gn, C) == yn).mean())
                res = {"C": best}
                # A
                res["A"] = summary(cv_pred(X, eff, prod, best), eff, prod, K)
                # B1
                Xb, yb, gb = X[diff], eff[diff], prod[diff]
                res["B1"] = summary(cv_pred(Xb, yb, gb, best), yb, gb, K)
                # C: recipient's natural features, same rows/labels
                Xc = X[xnat[diff]]
                res["C_input_probe"] = summary(cv_pred(Xc, yb, gb, best), yb, gb, K)
                # B2: natural-trained probe applied to held-out swapped rows
                allg = np.unique(prod)
                pred = np.full(diff.sum(), -1)
                dpos = np.where(diff)[0]
                for tr, te in folds(allg):
                    trp, tep = set(allg[tr]), set(allg[te])
                    mtr = (~sw) & np.isin(prod, list(trp))
                    clf = fit(X[mtr], own[mtr], best)
                    sel = np.isin(prod[dpos], list(tep))
                    pred[sel] = clf.predict(X[dpos[sel]])
                rec = own[diff]
                res["B2"] = summary(pred, yb, gb, K, {"agree_recipient": float((pred == rec).mean()),
                                                      "agree_recipient_ci": boot((pred == rec).astype(float), gb)})
                out["res"][tag] = res
                print(tag, {k: (v["acc"] if isinstance(v, dict) else v) for k, v in res.items()}, flush=True)
    with open(os.path.join(OUT, "probe.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["checks"], indent=1))


if __name__ == "__main__":
    main()

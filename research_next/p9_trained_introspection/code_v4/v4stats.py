"""Paper 9 v4: two-bin label and the input-only predictors (numpy only; shared by design_v4.py, the pilot ladder in
v4_run.py and v4_analyze.py, so that the bar is computed the same way everywhere).

Items are the rows of res_v3/screen.json (base model only): lp2 = log p of the emitted second digit at the first
answer position; para_lp2 = the same after the paraphrase prompt; ent2 = entropy at that position.
  med       median of lp2 over correct TRAIN items; bin 0 ("Very sure") if lp2 >= med, else bin 1 ("Less sure")
  med_para  median of para_lp2 over correct TRAIN items (for the recalibrated paraphrase predictor)
  med_ent   median of ent2 over correct TRAIN items (G3: entropy bin 0 if ent2 <= med_ent)
Input-only predictors of an item's bin:
  para        bin of para_lp2 at med (the v3 paraphrase predictor, two bins)
  recal_para  bin of para_lp2 at med_para (new in v4; the v3 audit showed it is stronger)
  majority    the majority bin of the fitting items
  learned     conditional logit on [is the para bin, is bin 1, is bin 1 x u], u = standardised log(-para_lp2),
              fitted on the fitting items (v1's cond_logit_fit, L-BFGS to |grad| < 1e-5)
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
from analyze import cond_logit_fit  # noqa: E402  (v1, frozen)

K = 2


def median_upper(v):
    v = sorted(v)
    return v[len(v) // 2]


def medians(scr):
    tr = [i for i in scr.values() if i["split"] == "train" and i["correct"]]
    return {"med": median_upper([i["lp2"] for i in tr]), "med_para": median_upper([i["para_lp2"] for i in tr]),
            "med_ent": median_upper([i["ent2"] for i in tr]), "n_train_items": len(tr)}


def bin2(lp2, med):
    return 0 if lp2 >= med else 1


def ent_bin2(e, med_ent):
    return 0 if e <= med_ent else 1


def _u(i):
    return math.log(max(-i["para_lp2"], 1e-12))


def _feats(i, M, mu, sd):
    pb = bin2(i["para_lp2"], M["med"])
    z = (_u(i) - mu) / sd
    return np.array([[float(b == pb), float(b == 1), float(b == 1) * z] for b in range(K)])


def predictors(fit_items, eval_items, M):
    """Predicted bins of eval_items by each input-only predictor; 'learned' and 'majority' fitted on fit_items."""
    y = np.array([bin2(i["lp2"], M["med"]) for i in fit_items])
    maj = int(np.bincount(y, minlength=K).argmax())
    us = np.array([_u(i) for i in fit_items])
    mu, sd = float(us.mean()), float(us.std() + 1e-9)
    w, gn = cond_logit_fit(np.array([_feats(i, M, mu, sd) for i in fit_items]), y)
    P = {"para": np.array([bin2(i["para_lp2"], M["med"]) for i in eval_items]),
         "recal_para": np.array([bin2(i["para_lp2"], M["med_para"]) for i in eval_items]),
         "majority": np.full(len(eval_items), maj),
         "learned": np.array([int(np.argmax(_feats(i, M, mu, sd) @ w)) for i in eval_items])}
    info = {"majority_bin": maj, "learned_w": w.tolist(), "learned_max_abs_grad": gn, "n_fit": len(fit_items)}
    return P, info

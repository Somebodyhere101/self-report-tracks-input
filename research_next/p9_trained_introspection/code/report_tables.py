"""Paper 9: print markdown tables for RESULTS.md from res/analysis.json, res_pilot/pilot.json and config.json.
Presentation only; no analysis logic. python report_tables.py [--selftest]"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p9lib import load_json, outdir  # noqa: E402

ROOT = os.path.dirname(HERE)


def f(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(f(x, nd) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    out = outdir("res_selftest" if a.selftest else "res")
    A = load_json(os.path.join(out, "analysis.json"))
    print(f"**Verdict: {A['verdict']}**\n")
    for r in A["reasons"]:
        print(f"- {r}")
    print("\n| Gate | Pass | Values |\n|---|---|---|")
    for g in sorted(A["gates"]):
        print(f"| {g} | {A['gates'][g]} | " + "; ".join(f"{k} {f(v)}" for k, v in A["gate_values"][g].items()) + " |")
    print("\n**T3 (natural transfer)**\n\n| Reporter | Accuracy vs CB (per seed) | Delta vs best input-only | 95% CI | divergent: names CB | divergent: names baseline |")
    print("|---|---|---|---|---|---|")
    for arm in ("N", "O", "I", "J"):
        t = A["T3"][arm]
        print(f"| {arm} | {f(t['acc_per_seed'])} | {f(t['delta_vs_best_baseline'])} | {f(t['ci'])} | "
              f"{f(t['divergent_names_cb'])} | {f(t['divergent_names_baseline'])} |")
    print("\n| Input-only predictor | Accuracy vs CB |\n|---|---|")
    for k, v in A["T3_baselines"].items():
        print(f"| {k} | {f(v)} |")
    print("\n**T1 (training family, held-out cities)**\n\n| Reporter | different-country acc [CI] | same-country acc | names input country (diff) |")
    print("|---|---|---|---|")
    for arm in ("N", "O", "I", "J"):
        t = A["T1"][arm]
        print(f"| {arm} | {f(t['acc_diff'])} {f(t['ci_diff'])} | {f(t['acc_same'])} | {f(t['names_input_country_diff'])} |")
    print("\n**T2 (unseen families; changed rows; chance 1/3)**\n\n| Reporter | changed rows (F2a, F2c) | acc F2a | acc F2c | pooled [CI] |")
    print("|---|---|---|---|---|")
    for arm in ("N", "O", "I", "J"):
        t = A["T2"][arm]
        print(f"| {arm} | {t['n_changed']['T2a']}, {t['n_changed']['T2c']} | {f(t['acc']['T2a'])} | {f(t['acc']['T2c'])} | "
              f"{f(t['pooled_acc'])} {f(t['pooled_ci'])} |")
    print("\n**T4 (tamper control: increase in error vs CB under random perturbation)**\n\n| Reporter | KL-matched [CI] | norm-matched [CI] |")
    print("|---|---|---|")
    for arm in ("N", "O", "I", "J"):
        t = A["T4"][arm]
        print(f"| {arm} | {f(t['kl']['switch_increase'])} {f(t['kl']['ci'])} | {f(t['norm']['switch_increase'])} {f(t['norm']['ci'])} |")
    print("\n**T5 (capability drift)**\n\n| Reporter | general MC (turn 2) | 2-hop, adapter on everywhere | 1-hop, adapter on everywhere |")
    print("|---|---|---|---|")
    for arm in ("N", "O", "I", "J"):
        t = A["T5"][arm]
        print(f"| {arm} | {f(t['mc_acc_gated'])} | {f(t['acc2_ungated'])} | {f(t['acc1_ungated'])} |")
    pil = os.path.join(outdir("res_pilot"), "pilot.json")
    if os.path.exists(pil):
        P = load_json(pil)
        print("\n**Pilot-chosen parameters**:", {k: v for k, v in P["chosen"].items() if k != "rules"})


if __name__ == "__main__":
    main()

"""v3 bed checks (PREDICTIONS_V3.md section 5) and ../config_v3.json. Exit 1 stops the chain before main data."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    load = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
    pil = load(os.path.join(ROOT, "res_pilot_v3", "pilot.json"))
    scr = load(os.path.join(ROOT, "res_v3", "screen.json"))["items"]
    acc = sum(i["correct"] for i in scr.values()) / len(scr)
    n_test = sum(1 for i in scr.values() if i["split"] == "test" and i["correct"])
    ch = pil["chosen"]
    checks = {
        "PG1_task": {"accuracy_all": acc, "n_test_usable": n_test, "pass": acc >= 0.70 and n_test >= 150},
        "PG2_ground_truth": {"bin_agreement_prob_vs_entropy": pil["P1"]["bin_agreement_prob_vs_entropy"],
                             "bin_counts": pil["P1"]["bin_counts"],
                             "pass": pil["P1"]["bin_agreement_prob_vs_entropy"] >= 0.70},
        "PG3_swap": {"l1": ch["l1"], "transfer": pil["P2"][f"from{ch['l1']}"]["transfer_to_donor"],
                     "digits_kept": pil["P2"][f"from{ch['l1']}"]["digits_kept"],
                     "pass": pil["P2"][f"from{ch['l1']}"]["transfer_to_donor"] >= 0.90},
        "PG5_smoke": {"pass": os.path.exists(os.path.join(ROOT, "res_v3_smoke", "analysis.json"))},
        "PG4_training_info": pil["P5"],
    }
    ok = all(v["pass"] for v in checks.values() if "pass" in v)
    json.dump({"checks": checks, "all_pass": ok}, open(os.path.join(ROOT, "res_pilot_v3", "pilot_gates.json"), "w"), indent=1)
    print(json.dumps(checks, indent=1))
    if not ok:
        print("V3 BED CHECK FAILED: the chain stops before any main data")
        sys.exit(1)
    cfg = {"n_train": 2400, "steps": 300, "bs": 8, "seeds": [0, 1]}
    cfg.update({k: ch[k] for k in ("l1", "ls", "la", "s_add", "s_kl", "lr")})
    json.dump(cfg, open(os.path.join(ROOT, "config_v3.json"), "w"), indent=1)
    print("wrote config_v3.json", cfg)


if __name__ == "__main__":
    main()

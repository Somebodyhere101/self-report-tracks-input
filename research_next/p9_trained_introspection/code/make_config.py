"""Paper 9: after the pilot, apply the frozen bed checks (PREDICTIONS.md section 10) and write res/config.json
(frozen base settings + parameters the pilot chose by its frozen rules). Exit 1 if a bed check fails, which
stops the chain before any analysis data exist. CPU only."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p9lib import load_json, outdir, save_json  # noqa: E402

ROOT = os.path.dirname(HERE)


def main():
    base = load_json(os.path.join(HERE, "config_base.json"))
    pil = load_json(os.path.join(outdir("res_pilot"), "pilot.json"))
    chosen = pil["chosen"]
    scr = [i for i in load_json(os.path.join(outdir("res"), "screen.json"))["items"] if i.get("ok")]
    acc2 = sum(i["acc2"] for i in scr) / len(scr)
    n_test = sum(1 for i in scr if i["split"] == "test" and i["acc2"])
    p1 = pil["P1"]
    l1 = chosen["l1"]
    checks = {
        "PG1_task": {"acc2_all": acc2, "n_test_usable": n_test, "pass": acc2 >= 0.70 and n_test >= 150},
        "PG2_readout": {"calibration_known": p1["calibration_known_cb_eq_gaz"], "reliability": p1["reliability"],
                        "pass": p1["calibration_known_cb_eq_gaz"] >= 0.80 and p1["reliability"] >= 0.70},
        "PG3_manipulation": {"l1": l1, "transfer": pil["P2"][f"from{l1}"]["transfer_to_donor"],
                             "pass": pil["P2"][f"from{l1}"]["transfer_to_donor"] >= 0.90},
        "PG5_selftest": {"pass": os.path.exists(os.path.join(outdir("res_selftest"), "analysis.json"))},
        "PG4_training_info": {k: v for k, v in pil["P5"].items()},
    }
    ok = all(v["pass"] for k, v in checks.items() if "pass" in v)
    cfg = dict(base)
    cfg.update({k: chosen[k] for k in ("l1", "ls", "la", "s_add", "s_kl", "lr")})
    save_json({"checks": checks, "all_pass": ok}, os.path.join(outdir("res_pilot"), "pilot_gates.json"))
    print(json.dumps(checks, indent=1, default=str))
    if not ok:
        print("BED CHECK FAILED: the chain stops before any analysis data are generated")
        sys.exit(1)
    save_json(cfg, os.path.join(ROOT, "config.json") if not os.environ.get("P9_TINY") else
              os.path.join(outdir("res_pilot"), "config_main.json"))
    print("wrote config:", cfg)


if __name__ == "__main__":
    main()

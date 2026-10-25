"""v4 learnability gate (PREDICTIONS_V4.md section 5) -> ../config_v4.json. Reads ../res_pilot_v4/ladder.json.
mode 'primary': the cheapest rung that passed the gate. mode 'exploratory': no rung passed, but at some rung I passed
and O learned (>= 2/3 on natural items); the verdict stays UNINFORMATIVE and the main data are reported as exploratory.
Exit 2 = neither (or the two-bin bed check failed): writes ../res_v4/NO_RUNG, and the chain skips the main phase (the
analysis then reports UNINFORMATIVE). In smoke mode the first rung is used whatever its result (mode 'smoke')."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SMOKE = bool(os.environ.get("V4_SMOKE"))
SUF = "_smoke" if SMOKE else ""


def main():
    load = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
    Ld = load(os.path.join(ROOT, "res_pilot_v4" + SUF, "ladder.json"))
    v3 = load(os.path.join(ROOT, "config_v3.json"))
    assert Ld.get("done"), "ladder not finished"
    assert v3["l1"] == 16
    ch, mode = Ld.get("chosen"), "primary"
    summary = {n: {k: r.get(k) for k in ("gate_I", "gate_O", "O_learned", "pass")} for n, r in Ld["rungs"].items()}
    print(json.dumps({"bed": Ld["bed"], "Pstar_pilot": Ld["Pstar_pilot"], "rungs": summary, "chosen": ch,
                      "fallback": Ld.get("fallback")}, indent=1))
    if ch is None and Ld.get("fallback"):
        ch, mode = Ld["fallback"], "exploratory"
    if ch is None and SMOKE:
        ch, mode = Ld["rungs"][Ld["order"][0]]["rung"], "smoke"
    if ch is None:
        os.makedirs(os.path.join(ROOT, "res_v4" + SUF), exist_ok=True)
        with open(os.path.join(ROOT, "res_v4" + SUF, "NO_RUNG"), "w", encoding="utf-8") as f:
            f.write(Ld.get("stopped", "no budget rung passed the learnability gate"))
        print("LEARNABILITY GATE NOT PASSED: the main phase is skipped (UNINFORMATIVE)")
        sys.exit(2)
    nr = os.path.join(ROOT, "res_v4" + SUF, "NO_RUNG")
    if os.path.exists(nr):  # a stale marker from an earlier, reset pilot must not skip a main phase that passed
        os.remove(nr)
    cfg = {"n_train": ch["steps"] * 8, "steps": ch["steps"], "bs": 8, "rank": ch["rank"], "lr": ch["lr"], "rung": ch["name"],
           "mode": mode, "seeds": [0] if SMOKE else [0, 1], "fmt": "imm"}
    cfg.update({k: v3[k] for k in ("l1", "ls", "la", "s_add")})
    cfg["s_kl"] = Ld["s_kl"]["s_kl"]  # recalibrated for the four-digit R by v3's rule (ladder setup)
    json.dump(cfg, open(os.path.join(ROOT, "config_v4" + SUF + ".json"), "w", encoding="utf-8"), indent=1)
    print("wrote config_v4" + SUF + ".json", cfg)


if __name__ == "__main__":
    main()

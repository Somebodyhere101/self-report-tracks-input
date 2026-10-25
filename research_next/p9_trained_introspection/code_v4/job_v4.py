"""v4 GPU job driver, run through tools/gpu_lock.py with an explicit --max-min. 'pilot' and 'main' refuse to run
unless PREDICTIONS_V4.sha256 verifies.
  python job_v4.py pilot   the budget ladder on pilot products, then make_config_v4.py (the learnability gate)
  python job_v4.py main    gen, 10 training runs (O, I, J, X immediate; Id delayed; 2 seeds), 12 evaluations
  python job_v4.py smoke   the same code with tiny counts on pilot items only (V4_SMOKE=1; *_smoke outputs)
Exit codes: 0 complete; 3 deadline near (state saved: re-queue); 2 the learnability gate or the two-bin bed check
failed (the main phase is skipped); 1 failed. Each step runs in its own process. A step starts only with at least
MIN_LEFT minutes before GPU_LOCK_DEADLINE (set by gpu_lock.py); training steps also watch it and save and exit 3."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import freeze_v4  # noqa: E402

PY = sys.executable
MIN_LEFT = 8
S = {"V4_SMOKE": "1"}
CHUNKS = {
    "pilot": [(["v4_run.py", "ladder"], {}), (["make_config_v4.py"], {})],
    "main": [(["v4_run.py", "gen"], {})]
    + [(["v4_run.py", "train", a, str(s)], {}) for s in (0, 1) for a in ("O", "I", "J", "X", "Id")]
    + [(["v4_run.py", "eval", a, "0"], {}) for a in ("N", "Nd")]
    + [(["v4_run.py", "eval", a, str(s)], {}) for s in (0, 1) for a in ("O", "I", "J", "X", "Id")],
    "smoke": [(["v4_run.py", "ladder"], S), (["make_config_v4.py"], S), (["v4_run.py", "gen"], S)]
    + [(["v4_run.py", "train", a, "0"], S) for a in ("O", "I", "J", "X", "Id")]
    + [(["v4_run.py", "eval", a, "0"], S) for a in ("N", "Nd", "O", "I", "J", "X", "Id")]
    + [(["v4_analyze.py"], S)],
}
DONE = {"pilot": os.path.join(ROOT, "res_pilot_v4", "PILOT_DONE"), "main": os.path.join(ROOT, "res_v4", "MAIN_DONE"),
        "smoke": os.path.join(ROOT, "res_v4_smoke", "SMOKE_DONE")}


def main():
    name = sys.argv[1]
    if name != "smoke" and not freeze_v4.check():
        print("v4 not frozen, or a frozen file changed: refusing to run", flush=True)
        sys.exit(1)
    dl = float(os.environ["GPU_LOCK_DEADLINE"]) if os.environ.get("GPU_LOCK_DEADLINE") else None
    for st, env in CHUNKS[name]:
        if dl and dl - time.time() < MIN_LEFT * 60:
            print(time.strftime("%H:%M:%S"), "deadline near; yielding before", " ".join(st), flush=True)
            sys.exit(3)
        t0 = time.time()
        print(time.strftime("%H:%M:%S"), "RUN", " ".join(st), env, flush=True)
        rc = subprocess.call([PY, os.path.join(HERE, st[0])] + st[1:], cwd=HERE, env=dict(os.environ, **env))
        print(time.strftime("%H:%M:%S"), "END", " ".join(st), "rc", rc, f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if rc == 3:
            sys.exit(3)
        if rc == 2 and st[0] == "make_config_v4.py":
            os.makedirs(os.path.dirname(DONE[name]), exist_ok=True)
            open(DONE[name], "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + " gate not passed")
            print("learnability gate not passed: main phase skipped", flush=True)
            sys.exit(2)
        if rc != 0:
            print("chunk", name, "FAILED at", st, flush=True)
            sys.exit(1)
    os.makedirs(os.path.dirname(DONE[name]), exist_ok=True)
    open(DONE[name], "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(time.strftime("%H:%M:%S"), "chunk", name, "complete", flush=True)


if __name__ == "__main__":
    main()

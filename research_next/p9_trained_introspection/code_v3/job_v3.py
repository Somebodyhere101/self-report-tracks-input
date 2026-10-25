"""v3 GPU job driver (through tools/gpu_lock.py). Refuses to run unless PREDICTIONS_V3.sha256 verifies.
  python job_v3.py pilot   screen, pilot, smoke pass (V3_SMOKE=1, res_v3_smoke/), then make_config_v3.py
  python job_v3.py main    gen, 8 training runs, 9 evaluations
Exit codes: 0 complete, 3 time budget reached (re-queue), 1 failed. Each step runs in its own process."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import freeze_v3  # noqa: E402

PY = sys.executable
YIELD_AFTER_MIN = 45
S = {"V3_SMOKE": "1"}
CHUNKS = {
    "pilot": [(["v3_run.py", "screen"], {}), (["v3_run.py", "pilot"], {}), (["v3_run.py", "gen"], S)]
    + [(["v3_run.py", "train", a, "0"], S) for a in "OIJX"]
    + [(["v3_run.py", "eval", a, "0"], S) for a in "NOIJX"]
    + [(["v3_analyze.py"], S), (["make_config_v3.py"], {})],
    "main": [(["v3_run.py", "gen"], {})]
    + [(["v3_run.py", "train", a, str(s)], {}) for s in (0, 1) for a in "OIJX"]
    + [(["v3_run.py", "eval", "N", "0"], {})]
    + [(["v3_run.py", "eval", a, str(s)], {}) for s in (0, 1) for a in "OIJX"],
}
DONE = {"pilot": os.path.join(ROOT, "res_pilot_v3", "PILOT_DONE"), "main": os.path.join(ROOT, "res_v3", "MAIN_DONE")}


def main():
    name = sys.argv[1]
    if not freeze_v3.check():
        print("v3 not frozen, or a frozen file changed: refusing to run", flush=True)
        sys.exit(1)
    t00 = time.time()
    for st, env in CHUNKS[name]:
        if (time.time() - t00) / 60 > YIELD_AFTER_MIN:
            print(time.strftime("%H:%M:%S"), "time budget reached; yielding", flush=True)
            sys.exit(3)
        t0 = time.time()
        print(time.strftime("%H:%M:%S"), "RUN", " ".join(st), env, flush=True)
        rc = subprocess.call([PY, os.path.join(HERE, st[0])] + st[1:], cwd=HERE, env=dict(os.environ, **env))
        print(time.strftime("%H:%M:%S"), "END", " ".join(st), "rc", rc, f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            print("chunk", name, "FAILED at", st, flush=True)
            sys.exit(1)
    os.makedirs(os.path.dirname(DONE[name]), exist_ok=True)
    open(DONE[name], "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(time.strftime("%H:%M:%S"), "chunk", name, "complete", flush=True)


if __name__ == "__main__":
    main()

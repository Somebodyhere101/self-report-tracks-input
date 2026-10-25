"""Paper 9 GPU job driver, run through tools/gpu_lock.py. Each step runs in its own process, so a CUDA failure
cannot leave a hung process holding the lock. Finished steps are skipped by the stages themselves.
  python job.py pilot   Qwen3 engine self-test, item screen, pilot, end-to-end self-test on pilot cities,
                        then make_config.py (frozen bed checks; exit 1 stops the chain)
  python job.py main    gen, 8 training runs, 9 evaluations
Exit codes: 0 chunk complete, 3 time budget reached (re-queue to continue), 1 a step failed."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ST = ["--selftest"]
YIELD_AFTER_MIN = 45  # no new step starts after this, so a chunk stays under about 60 minutes
CHUNKS = {
    "pilot": [["selftest_engine.py"], ["stage_screen.py"], ["stage_pilot.py"], ["stage_gen.py"] + ST]
    + [["stage_train.py", "--arm", a, "--seed", "0"] + ST for a in "OIJX"]
    + [["stage_eval.py", "--arm", a, "--seed", "0"] + ST for a in "OIJX"]
    + [["stage_eval.py", "--arm", "N"] + ST, ["analyze.py"] + ST, ["make_config.py"]],
    "main": [["stage_gen.py"]]
    + [["stage_train.py", "--arm", a, "--seed", str(s)] for s in (0, 1) for a in "OIJX"]
    + [["stage_eval.py", "--arm", "N"]]
    + [["stage_eval.py", "--arm", a, "--seed", str(s)] for s in (0, 1) for a in "OIJX"],
}
DONE = {"pilot": os.path.join(os.path.dirname(HERE), "res_pilot", "PILOT_DONE"),
        "main": os.path.join(os.path.dirname(HERE), "res", "MAIN_DONE")}


def frozen():
    """True if PREDICTIONS.sha256 exists and every frozen file (code, docs and the paper 8 / S2 files imported
    read-only) matches its hash or a hash logged in PREDICTIONS.sha256.amend; PREDICTIONS.md must match."""
    sys.path.insert(0, HERE)
    import freeze
    if not os.path.exists(freeze.OUT):
        return False
    return freeze.check(verbose=True)


def main():
    name = sys.argv[1]
    if not frozen():
        print("not frozen, or a frozen file changed without a logged amendment: refusing to run", flush=True)
        sys.exit(1)
    t00 = time.time()
    for st in CHUNKS[name]:
        if (time.time() - t00) / 60 > YIELD_AFTER_MIN:
            print(time.strftime("%H:%M:%S"), "time budget reached; yielding the GPU", flush=True)
            sys.exit(3)
        t0 = time.time()
        print(time.strftime("%H:%M:%S"), "RUN", " ".join(st), flush=True)
        rc = subprocess.call([PY, os.path.join(HERE, st[0])] + st[1:], cwd=HERE)
        print(time.strftime("%H:%M:%S"), "END", " ".join(st), "rc", rc, f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            print("chunk", name, "FAILED at", st[0], flush=True)
            sys.exit(1)
    open(DONE[name], "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(time.strftime("%H:%M:%S"), "chunk", name, "complete", f"{(time.time() - t00) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

"""Stage 0b GPU job (run through tools/gpu_lock.py): the probe extraction, then the tiny training runs."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for script in ("s0b_extract.py", "s0b_train.py"):
    rc = subprocess.call([sys.executable, os.path.join(HERE, script)], cwd=HERE)
    print("END", script, "rc", rc, flush=True)
    if rc != 0:
        sys.exit(rc)

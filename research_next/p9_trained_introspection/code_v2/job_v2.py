"""v2 GPU job (run through tools/gpu_lock.py): refuses to run unless PREDICTIONS_V2.sha256 verifies, then runs
v2_premise.py in its own process."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import freeze_v2  # noqa: E402

if not freeze_v2.check():
    print("v2 not frozen, or a frozen file changed: refusing to run", flush=True)
    sys.exit(1)
sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "v2_premise.py")], cwd=HERE))

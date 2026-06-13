#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click consistent-model regeneration + obs_cache cleanup + RQS_NF.py run.

Run from the repository root:
    python shell_run.py
"""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODEL1 = ROOT / "model1"
OBS_CACHE = ROOT / "obs_cache"
RQS_SCRIPT = ROOT / "RQS_NF.py"

TASKS = [
    (
        ROOT / "initial1" / "background" / "background_pre_pml.py",
        ["vp_background_pml.npy", "vp_background_pml.json"],
    ),
    (
        ROOT / "initial1" / "true" / "true_pre_pml.py",
        ["vp_true_pml.npy", "vp_true_pml.json"],
    ),
    (
        ROOT / "initial1" / "initial" / "initial_pre_pml.py",
        ["vp_initial_pml.npy", "vp_initial_pml.json"],
    ),
]

def run_checked(cmd, cwd):
    print(f"\n[RUN] cwd={Path(cwd).relative_to(ROOT)}")
    print("      " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)

def main():
    if not RQS_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find {RQS_SCRIPT}")

    MODEL1.mkdir(exist_ok=True)

    print("=" * 80)
    print("[STEP 1] Regenerate consistent PML models")
    print("=" * 80)

    for script_path, pml_files in TASKS:
        if not script_path.exists():
            raise FileNotFoundError(f"Cannot find {script_path}")

        run_checked([sys.executable, str(script_path)], cwd=script_path.parent)

        for name in pml_files:
            src = script_path.parent / name
            dst = MODEL1 / name
            if not src.exists():
                raise FileNotFoundError(f"Expected output was not generated: {src}")
            shutil.copy2(src, dst)
            print(f"[COPY] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    print("\n" + "=" * 80)
    print("[STEP 2] Remove old observation cache")
    print("=" * 80)

    if OBS_CACHE.exists():
        shutil.rmtree(OBS_CACHE)
        print(f"[DELETE] {OBS_CACHE.relative_to(ROOT)}")
    else:
        print("[SKIP] obs_cache does not exist")

    print("\n" + "=" * 80)
    print("[STEP 3] Run RQS_NF.py")
    print("=" * 80)

    # FORCE_REGEN_OBS_CACHE=1 is also passed defensively.
    # Even if obs_cache was already deleted, this prevents loading stale cache if paths change.
    import os
    env = os.environ.copy()
    env["FORCE_REGEN_OBS_CACHE"] = "1"

    subprocess.run([sys.executable, str(RQS_SCRIPT)], cwd=str(ROOT), env=env, check=True)

    print("\n[DONE] Consistent PML models regenerated, obs_cache removed, and RQS_NF.py finished.")

if __name__ == "__main__":
    main()

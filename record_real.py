"""
record_real.py  -  RUN THIS ON YOUR OPENWHISK MACHINE (the one with GPUs).

It launches your REAL models for every (function, arm), a few times each,
and saves the timings to measurements.json. This is the slow step and it
only needs to run ONCE. After it finishes, run_real.py replays the saved
file to produce every experiment and figure, with no further inference.

Requirements:
  - This file must sit in the SAME folder as your original profiler.py
    (the one that defines measure_latency_and_cost and function_cfg).
  - profiler_core.py and measurement.py must also be in that folder.
  - Your model scripts must be reachable at the filepaths in function_cfg.

Usage:
  python record_real.py
  # optional: change repeats below (more repeats = better variance numbers)
"""

from profiler import measure_latency_and_cost, function_cfg
from measurement import RealRecorder

REPEATS = 8  # bump to 8 or 10 for tighter error bars (runs longer, once)

if __name__ == "__main__":
    print(f"Recording real measurements, {REPEATS} repeats per arm ...")
    print("This launches your real models. It will take a while.\n")

    records = RealRecorder(measure_latency_and_cost, function_cfg).record_all(
        repeats=REPEATS, out_path="measurements.json"
    )

    n_arms = sum(len(v) for v in records.values())
    print(f"\nDone. Wrote measurements.json")
    print(f"  functions: {len(records)}")
    print(f"  total arms recorded: {n_arms}")
    print(f"  repeats each: {REPEATS}")
    print("\nNext: run  python run_real.py")

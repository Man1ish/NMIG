import math
import random
import subprocess
import sys
import time
from collections import defaultdict, namedtuple
from pathlib import Path
import json
import os
import shlex


# define your function metadata here
function_cfg = {
    # "alexnet": {
    #     "memory_mb":    1024,  # CPU mem in MB
    #     "batch":           1,
    #     "batching":    False,
    #     "gpu1_mb":       [558],  # MB
    #     # if you have two GPUs:
    #     "2xgpu": {"gpu1_mb": [798], "gpu2_mb": [656]},
    #     "batch_sequence": [1],
    #     "filename": "alexnet.py",
    #     "filepath": "python_runtime/core/mlruntime/alexnet/alexnet.py",
    #     "devices": ["CPU", "GPU", "2xGPU"]
    # },
    "alexnet": {
        "memory_mb":    1024,  # CPU mem in MB
        "batch":           15,
        "batching":    True,
        "gpu1_mb":       [588, 608, 634, 668, 720],  # MB
        # if you have two GPUs:
        "2xgpu": {"gpu1_mb": [950, 968, 976, 986], "gpu2_mb": [670, 690, 706, 706]},
        "batch_sequence": [1, 2, 3, 4],
        "filename": "alexnet_video.py",
        "filepath": "python_runtime/core/mlruntime/alexnet/alexnet_video.py",
        "devices": ["CPU", "GPU", "2xGPU"]
    },
    "efficientnet": {
        "memory_mb":    1024,  # CPU mem in MB
        "batch":           15,
        "batching":    False,
        "gpu1_mb":       [1020, 1434, 1856, 2280],  # MB
        # if you have two GPUs:
        "2xgpu": {"gpu1_mb": [954, 1150, 1384, 1584], "gpu2_mb": [888, 1102, 1306, 1528]},
        "batch_sequence": [1, 2, 3, 4],
        "filename": "efficientnet_video.py",
        "filepath": "python_runtime/core/mlruntime/efficientnet/efficientnet_video.py",
        "devices": ["CPU", "GPU", "2xGPU"]
    },
    "resnet50": {
        "memory_mb":    1024,  # CPU mem in MB
        "batch":           1,
        "batching":    True,
        "gpu1_mb":       [528],  # MB
        # if you have two GPUs:
        "2xgpu": {"gpu1_mb": [598], "gpu2_mb": [600]},
        "batch_sequence": [1],
        "filename": "resnet50.py",
        "filepath": "python_runtime/core/mlruntime/resnet50/resnet50.py",
        "devices": ["CPU", "GPU", "2xGPU"]
    },
    "inception": {
        "memory_mb":    1024,  # CPU mem in MB
        "batch":           1,
        "batching":    False,
        "gpu1_mb":       [536],  # MB
        # if you have two GPUs:
        "2xgpu": {"gpu1_mb": [630], "gpu2_mb": [630]},
        "batch_sequence": [1],
        "filename": "inception.py",
        "filepath": "python_runtime/core/mlruntime/inception/inception.py",
        "devices": ["CPU", "GPU", "2xGPU"]
    },
    "googlenet": {
        "memory_mb":    1024,  # CPU mem in MB
        "batch":           1,
        "batching":    False,
        "gpu1_mb":       [390],  # MB
        # if you have two GPUs:
        "2xgpu": {"gpu1_mb": [490], "gpu2_mb": [490]},
        "batch_sequence": [1],
        "filename": "googlenet.py",
        "filepath": "python_runtime/core/mlruntime/googlenet/googlenet.py",
        "devices": ["CPU", "GPU", "2xGPU"]
    },
    "bert": {
        "memory_mb":    1024,  # CPU mem in MB
        "batch":           1,
        "batching":    False,
        "gpu1_mb":       [778],  # MB
        # if you have two GPUs:
        "2xgpu": {"gpu1_mb": [778], "gpu2_mb": [778]},
        "batch_sequence": [1],
        "filename": "bert.py",
        "filepath": "python_runtime/core/mlruntime/bert/bert.py",
        "devices": ["CPU", "GPU"]
    },
    "distilgpt2": {
        "memory_mb":    1024,  # CPU mem in MB
        "batch":           1,
        "batching":    False,
        "gpu1_mb":       [668],  # MB
        # if you have two GPUs:
        "2xgpu": {"gpu1_mb": [0], "gpu2_mb": [0]},
        "batch_sequence": [1],
        "filename": "distilgpt2.py",
        "filepath": "python_runtime/core/mlruntime/distilgpt2/distilgpt2.py",
        "devices": ["CPU", "GPU"]
    }

}

# the arm tuple now carries the function_id too
Arm = namedtuple("Arm", ["fid", "device", "batch"])

from enum import Enum

# Define experiment types
class ExperimentType(Enum):
    LATENCY_FIRST = "latency"
    COST_FIRST = "cost"
    BALANCED = "balanced"

# Updated reward weights for each experiment
def get_reward_weights(exp_type: ExperimentType):
    if exp_type == ExperimentType.LATENCY_FIRST:
        return dict(gamma_time=0.0, gamma_mem=0.0)
    elif exp_type == ExperimentType.COST_FIRST:
        return dict(gamma_time=0.1, gamma_mem=0.05)
    elif exp_type == ExperimentType.BALANCED:
        return dict(gamma_time=0.05, gamma_mem=0.025)
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")




class MABProfiler:
    """UCB-1 over (device, batch) for a single function."""
    def __init__(self, arms):
        self.arms = arms
        self.counts = defaultdict(int)
        self.values = defaultdict(float)
        self.total_rounds = 0

    def select_arm(self):
        self.total_rounds += 1
        # try each once
        for a in self.arms:
            if self.counts[a] == 0:
                return a
        # UCB1
        log_t = math.log(self.total_rounds)
        best, best_score = None, -1e9
        for a in self.arms:
            mean = self.values[a]
            bonus = math.sqrt(2 * log_t / self.counts[a])
            score = mean + bonus
            if score > best_score:
                best, best_score = a, score
        return best

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


def run_and_time_script(cmd, python_exe=None):
    # normalize so flags and values are separate elements
    if isinstance(cmd, (list, tuple)):
        argv = []
        for elt in cmd:
            argv.extend(str(elt).split())
    else:
        # treat cmd as a path to a script
        script = Path(cmd)
        if not script.exists():
            raise FileNotFoundError(script)
        python_exe = python_exe or sys.executable
        argv = [python_exe, str(script)]

    start = time.time()
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {
        "returncode": p.returncode,
        "stdout":     p.stdout,
        "stderr":     p.stderr,
        "latency_s":  time.time() - start  # wall-clock; kept for diagnostics only
    }


def modify_input_for_function(arm):
    """
    Convert chosen arm into the actual arguments your function
    entrypoint expects.
    """
    cmd = []
    # device
    if arm.device in ("GPU", "2xGPU"):
        cmd.append("--device gpu")
    else:
        cmd.append("--device cpu")

    # multi-GPU
    if arm.device == "2xGPU":
        cmd.append("--multi-gpu")

    # batch size
    if arm.batch:
        cmd.append("--batch {}".format(int(arm.batch)))

    # local-mode is always on
    cmd.append("--local")

    # join with spaces and return
    return cmd

def find_nearest_index(value, array, tol=1e-5):
    for i, v in enumerate(array):
        if abs(v - value) < tol:
            return i
    raise ValueError(f"{value} not found in list within tolerance {tol}")


def _parse_metrics(stdout):
    """Pull the 'METRICS {...}' line the model script prints.

    Returns the parsed dict, or None if no valid METRICS line is present.
    The model entrypoint must print exactly one line of the form:
        METRICS {"exec_s": <float>, "inference_s": <float>, "status": "success"}
    where exec_s is the device-attributable execution time
    (output_end - gpu_transfer_start), in seconds.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("METRICS "):
            try:
                return json.loads(line[len("METRICS "):])
            except json.JSONDecodeError:
                return None
    return None


def measure_latency_and_cost(arm):
    """
    1) launch the real inference via subprocess
    2) extract the device-attributable exec_s from the model's METRICS line
       (NOT subprocess wall-clock, which is dominated by interpreter startup
        and model-load-from-disk and would make all arms look near-identical)
    3) compute cpu_mem_s = cpu_mem_mb/1024 * latency_s
       compute gpu_mem_s = sum(gpu_i_mb/1024 * latency_s)
       compute gpu_time_s = latency_s if GPU used else 0
    """
    fid, dev, bmul = arm
    cfg = function_cfg[fid]
    filepath = cfg["filepath"]

    # First we execute the function with the selected arm
    cmd = [sys.executable, filepath]
    # We modify the cmd to include arms parameter
    input_cmd = cmd + modify_input_for_function(arm)
    # we execute the script
    res = run_and_time_script(input_cmd)

    # Parse the device-attributable timing from the model's METRICS line.
    # Fail LOUDLY if it is missing -- silently falling back to wall-clock is
    # exactly what poisons short-function measurements with import overhead.
    metrics = _parse_metrics(res["stdout"])
    if metrics is None or "exec_s" not in metrics:
        raise RuntimeError(
            f"No METRICS line from {fid}/{dev}/batch={arm.batch}. "
            f"The model script must print 'METRICS {{...}}' with exec_s.\n"
            f"returncode={res['returncode']}\n"
            f"--- stderr ---\n{res['stderr']}\n"
            f"--- stdout ---\n{res['stdout']}"
        )
    if metrics.get("status") not in (None, "success"):
        raise RuntimeError(
            f"Model {fid}/{dev}/batch={arm.batch} reported non-success: {metrics}"
        )

    latency = float(metrics["exec_s"])

    # now compute memory-time in GB*s
    cpu_mem_s = (cfg["memory_mb"] / 1024) * latency

    if dev == "CPU":
        gpu_time_s = 0.0
        gpu_mem_s = 0.0
    elif dev == "GPU":
        gpu_time_s = latency
        batch_multiplier = arm.batch / cfg["batch"]
        idx = find_nearest_index(batch_multiplier, cfg["batch_sequence"])
        g1 = cfg["gpu1_mb"][idx]
        gpu_mem_s = (g1 / 1024) * latency
    else:  # "2xGPU"
        gpu_time_s = latency
        batch_multiplier = arm.batch / cfg["batch"]
        idx = find_nearest_index(batch_multiplier, cfg["batch_sequence"])
        g1 = cfg["2xgpu"]["gpu1_mb"][idx]
        g2 = cfg["2xgpu"]["gpu2_mb"][idx]
        gpu_mem_s = ((g1 + g2) / 1024) * latency

    return latency, cpu_mem_s, gpu_time_s, gpu_mem_s


# reward weights
T_SLO = 1.0
alpha = 0.1
beta = 1.0

def compute_reward(lat, cpu_mem_s, gpu_time_s, gpu_mem_s, gamma_time=0.1, gamma_mem=0.05):
    """
    Reward =
      + large bonus for finishing well under the SLO
      - steep penalty for SLO misses
      - gamma_time * GPU-seconds used
      - gamma_mem  * GPU-GB.seconds used

    NOTE: this legacy reward is kept only for the standalone run_simulation
    below. The revision pipeline (profiler_core.compute_reward) uses the
    unified per-function SLO cost model instead.
    """
    if lat <= T_SLO:
        slo_bonus = (T_SLO - lat)
    else:
        slo_bonus = -10.0 * (lat - T_SLO)

    gpu_cost = gamma_time * gpu_time_s + gamma_mem * gpu_mem_s
    return slo_bonus - gpu_cost


# one profiler per function id
profilers = {}

# Modify handle_invocation to pass the experiment type
def handle_invocation(fid, exp_type: ExperimentType):
    cfg = function_cfg[fid]

    if fid not in profilers:
        devices = cfg["devices"]
        base = cfg["batch"]
        factors = cfg["batch_sequence"]
        multipliers = [math.floor(f * base) for f in factors]
        arms = [Arm(fid, d, b) for d in devices for b in multipliers]
        profilers[fid] = MABProfiler(arms)

    prof = profilers[fid]
    arm = prof.select_arm()
    lat, cpu_m_s, gpu_t_s, gpu_m_s = measure_latency_and_cost(arm)

    weights = get_reward_weights(exp_type)
    reward = compute_reward(lat, cpu_m_s, gpu_t_s, gpu_m_s, **weights)
    prof.update(arm, reward)

    return arm, lat, reward




# Modify run_simulation to take experiment type
def run_simulation(name=None, num_rounds=20, output_file="dataset/best_configs.json", exp_type=ExperimentType.BALANCED):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                best_configs = json.load(f)
            except json.JSONDecodeError:
                best_configs = {}
    else:
        best_configs = {}

    fids = [name] if name else list(function_cfg.keys())

    for fid in fids:
        print(f"\n=== Profiling function: {fid} ({exp_type.value}) ===")
        for i in range(num_rounds):
            arm, lat, rew = handle_invocation(fid, exp_type)
            print(f"Round {i}: arm={arm}, latency={lat:.3f}, reward={rew:.4f}")

        best = max(profilers[fid].arms, key=lambda a: profilers[fid].values[a])
        print(f">>> Best config for {fid} ({exp_type.value}): {best}")
        best_configs[fid] = {
            "device": best.device,
            "batch": best.batch
        }

    with open(output_file, "w") as f:
        json.dump(best_configs, f, indent=2)

    print(f"\nBest configurations updated in: {output_file}")
    return best_configs




if __name__ == "__main__":
    run_simulation(num_rounds=20, output_file="dataset/latency_first.json", exp_type=ExperimentType.COST_FIRST)
    run_simulation(num_rounds=20, output_file="dataset/latency_first.json", exp_type=ExperimentType.BALANCED)
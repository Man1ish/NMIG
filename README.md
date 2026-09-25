# NMIG: No More Idle GPUs

**Decoupling GPU Contexts from Container Lifetimes in Serverless ML Inference**

Manish Pandey and Young-woo Kwon, Kyungpook National University

---

## Abstract

Serverless machine learning (ML) inference enables on-demand scalability and pay-per-use efficiency but often suffers from idle GPU retention, prolonged cold starts, and excessive initialization overheads. Latency-oriented optimizations such as pre-warming, zygote containers, and model pre-loading reduce cold-start delays but inadvertently bind GPU contexts to inactive containers. Idle GPU retention is therefore a first-class inefficiency, distinct from cold start, that wastes over 90% of allocated GPU memory and triggers cascading failures.

This paper introduces **No More Idle GPUs (NMIG)**, a serverless runtime that addresses this inefficiency by decoupling GPU context lifetime from container lifetime. NMIG uses three mechanisms: process-level isolation releases the GPU immediately after each inference, a UCB-1 bandit profiler selects device and batch size per invocation, and container reuse amortizes initialization across compatible functions while preserving serverless statelessness.

We implement NMIG in Apache OpenWhisk and evaluate it on real Azure traces across normal, similar, and bursty workloads. NMIG reduces GPU memory consumption by over 98.5% across all workloads, while peak memory falls only by 50–75% of the baseline, and completes every invocation with zero failures, where baselines drop up to 25% of requests. The per-invocation overhead of releasing and reloading GPU state is a fixed cost of 1.5–2.2 seconds that is largely independent of model size. This overhead is offset by the reduction in GPU memory and the elimination of GPU-exhaustion failures.

---

## Requirements

The experiments in the paper were run on the following setup:

- Ubuntu 20.04.6 LTS
- 128 GB RAM, 16-core x86-64 CPU
- 2× NVIDIA GeForce RTX 3090 (24 GB each)
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- Python 3.9

---

## Setup

### 1. Clone the repository

The code is on the `nmig` branch:

```bash
git clone -b nmig https://github.com/Man1ish/openwhisk_nmig.git
cd openwhisk_nmig
```

### 2. Install the Python dependencies

We recommend using a virtual environment:

```bash
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the large folders

The `python_runtime` and `results` folders are not included in the repository because of their size. Download them from Google Drive and place them in the root of the repository.

| Folder | Link |
|---|---|
| `python_runtime` | [Download](https://drive.google.com/file/d/1AYIyRhu40tshdXSJCn2T-fC0KEZmiWki/view?usp=sharing) |
| `results` | [Download](https://drive.google.com/file/d/1F300rLoJduhppV99t2yAn5GcrYVNFsVv/view?usp=sharing) |

You can also download them from the command line with `gdown`:

```bash
pip install gdown
gdown 1AYIyRhu40tshdXSJCn2T-fC0KEZmiWki   # python_runtime
gdown 1F300rLoJduhppV99t2yAn5GcrYVNFsVv   # results
```

Extract the downloaded archives so that the repository contains:

```
openwhisk_nmig/
├── python_runtime/
├── results/
├── resourceserver/
├── run_script.sh
├── app.py
└── ...
```

---

## Running NMIG

The project uses three terminals.

**Terminal 1: start OpenWhisk**

```bash
# start OpenWhisk
<command to start OpenWhisk>
```

**Terminal 2: start the resource monitor**

```bash
cd resourceserver/
python monitor_server.py
```

**Terminal 3: deploy the functions and run the experiment**

```bash
bash run_script.sh update_all
python app.py
```

### Configurations

`app.py` contains several configurations. Select the one you want to run before starting it:

<!-- List the available configurations here, for example: -->
<!-- - `openwhisk`: unmodified OpenWhisk baseline -->
<!-- - `histogram`: Histogram keep-alive policy -->
<!-- - `pagurus`: Pagurus container reuse -->
<!-- - `nmig`: full NMIG -->

---

## Results

The `results` folder (see [Download the large folders](#3-download-the-large-folders)) contains the measurements used for the tables and figures in the paper.

---

## Contact

If you have any problems running the code, please contact **Manish Pandey** at [manishpandeyabc@gmail.com](mailto:manishpandeyabc@gmail.com).

---

## Citation

If you use NMIG in your research, please cite:

```bibtex
@inproceedings{pandey2026nmig,
  author    = {Pandey, Manish and Kwon, Young-woo},
  title     = {No More Idle GPUs: Decoupling GPU Contexts from Container Lifetimes in Serverless ML Inference},
  booktitle = {<Conference name>},
  year      = {2026}
}
```

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2021-NR060080).
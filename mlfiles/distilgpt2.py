import sys
import datetime
import time
import warnings

# Mark library load time
lib_start = int(time.time() * 1000)

import torch
import json
import psutil
import os
import transformers
from transformers import GPT2Config, AutoTokenizer, AutoModelForCausalLM
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
import argparse

# Mark library load time
lib_end = int(time.time() * 1000)


def trace(msg):
    # print(json.dumps({"trace": msg}))
    # sys.stdout.flush()
    pass

def get_system_metrics():
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()

    gpu_stats = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            stats = {
                "gpu_index": i,
                "device": torch.cuda.get_device_name(i),
                "memory_allocated_mb": round(torch.cuda.memory_allocated(i) / (1024 * 1024), 2),
                "memory_reserved_mb": round(torch.cuda.memory_reserved(i) / (1024 * 1024), 2),
                "total_memory_mb": round(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024), 2)
            }
            gpu_stats.append(stats)

    return {
        "cpu_usage_percent": cpu_percent,
        "memory_used_mb": round(mem.used / (1024 * 1024), 2),
        "memory_total_mb": round(mem.total / (1024 * 1024), 2),
        "gpus": gpu_stats
    }

def main(args):
    try:
        system_stats_before = get_system_metrics()
        input_start = int(time.time() * 1000)
        # === Device selection ===
        trace("Starting GPT2 inference main()")
        use_gpu = args.get("device", "auto")  # "cpu", "gpu", or "auto"
        # device = torch.device("cuda" if torch.cuda.is_available() and args.get("device", "auto") != "cpu" else "cpu")
        multi_gpu = args.get("multi_gpu", False)
        gpu_first = args.get("gpu", "first")
        local = args.get("local", False)
        # Resolve paths
        SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
        if local:
            include_path = ""
        else:
            include_path = "/"
        
        input_end = int(time.time() * 1000)
        # === Load model from disk ===
        model_load_start = int(time.time() * 1000)

        model_path = os.path.join(SCRIPT_DIR, include_path, "distilgpt2_local")
        model_weights = os.path.join(SCRIPT_DIR, include_path, "distilgpt2.pth")

       
        trace("Loading tokenizer and config")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = GPT2Config.from_pretrained(model_path)

        model = AutoModelForCausalLM.from_config(config)
        model_load_end = int(time.time() * 1000)

         # === Transfer model to device ===
        gpu_transfer_start = int(time.time() * 1000)

        if gpu_first == "first":
            gpu_first = "cuda"
        elif gpu_first == "second":
            gpu_first = "cuda:1"


        # Set device
        if use_gpu == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
            model.to(device)
            # print("🖥️ Using CPU")
        else:
            device = torch.device(gpu_first)
            # print("🚀 Using GPU")

            # Use multiple GPUs if available
            if multi_gpu and torch.cuda.device_count() > 1:
                # print(f"🚀 Using DataParallel with {torch.cuda.device_count()} GPUs")
                # model = torch.nn.DataParallel(model)
                pass


        model.load_state_dict(torch.load(model_weights, map_location=device))


        model.to(device)
        model.eval()
        
        gpu_transfer_end = int(time.time() * 1000)

        preprocess_start = int(time.time() * 1000)

        trace("Running inference")

        prompt = "'''Python function to add two numbers'''\ndef add_numbers(a, b):"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        preprocess_end = int(time.time() * 1000)
        inference_start = int(time.time() * 1000)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        inference_end = int(time.time() * 1000)

        output_start = int(time.time() * 1000)
        trace("Prediction and label fetched")

        output_end = int(time.time() * 1000)
        system_stats_after = get_system_metrics()
        return {
            "status": "success",
            "type": "distilgpt2",
            "output": generated_text,
            "system_stats_before": system_stats_before,
            "system_stats_after": system_stats_after,
            "measurement": {
                "lib_load_start": lib_start,
                "lib_load_end": lib_end,
                "model_load_start": model_load_start,
                "model_load_end": model_load_end,
                "image_load_start": input_start,
                "image_load_end": input_end,
                "preprocess_start": preprocess_start,
                "preprocess_end": preprocess_end,
                "gpu_transfer_start": gpu_transfer_start,
                "gpu_transfer_end": gpu_transfer_end,
                "inference_start": inference_start,
                "inference_end": inference_end,
                "output_start": output_start,
                "output_end": output_end,
                "is_cold": 0,
                "is_loaded": 1
            }
        }

    except Exception as e:
        trace(f"Exception: {str(e)}")
        return {
            "status": "error",
            "trace": f"Exception: {str(e)}",
            "message": str(e)
        }

def parse_args():
    p = argparse.ArgumentParser(description="Run inference profiler")
    p.add_argument(
        "--device", choices=["cpu","gpu"], default="gpu",
        help="Which device to run on"
    )
    p.add_argument(
        "--multi-gpu", action="store_true", default=False,
        help="Whether to use multiple GPUs"
    )
    p.add_argument(
        "--local", action="store_true", default=True,
        help="Run in local (development) mode"
    )
    p.add_argument("--batch", type=int, default=1, help="Batch size")

    return vars(p.parse_args())

# === Entrypoint ===
if __name__ == "__main__":
    args = parse_args()
    try:
        result = main(args)
        if result.get("status") == "success":
            m = result["measurement"]
            exec_ms = m["output_end"] - m["gpu_transfer_start"]   # device-attributable
            print("METRICS " + json.dumps({
                "exec_s": exec_ms / 1000.0,
                "inference_s": (m["inference_end"] - m["inference_start"]) / 1000.0,
                "status": "success",
            }))
        else:
            print("METRICS " + json.dumps(result))
        sys.stdout.flush()
    except Exception as e:
        print("METRICS " + json.dumps({"status": "fatal_error", "message": str(e)}))
        sys.exit(1)
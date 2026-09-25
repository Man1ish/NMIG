import time

# === Record library load time ===
lib_load_start = int(time.time() * 1000)
import sys

import torch
import json
import psutil
import os
import warnings
import transformers
from transformers import BertTokenizer, BertForMaskedLM
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
lib_load_end = int(time.time() * 1000)
import argparse
# === Trace helper ===
def trace(msg):
    # print(json.dumps({"trace": msg}))
    # sys.stdout.flush()
    pass

# === System metrics ===
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

# === Main inference logic ===
def main(args):
    try:
        system_stats_before = get_system_metrics()
        # === Input preparation ===
        input_load_start = int(time.time() * 1000)
        use_gpu = args.get("device", "auto")
        multi_gpu = args.get("multi_gpu", False)
        gpu_first = args.get("gpu", "first")
        local = args.get("local", False)


        sentence = "The capital of Nepal is [MASK]."
        input_load_end = int(time.time() * 1000)

        # === Path and device setup ===
        SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
        include_path = "" if local else "/"

        # === Load tokenizer and model ===
        model_load_start = int(time.time() * 1000)
        tokenizer = BertTokenizer.from_pretrained(include_path + "bert-base-uncased")
        model = BertForMaskedLM.from_pretrained(include_path + "bert-base-uncased")
        model_load_end = int(time.time() * 1000)

        # === Transfer model to device ===
        gpu_transfer_start = int(time.time() * 1000)

        if gpu_first == "first":
            gpu_first = "cuda"
        elif gpu_first == "second":
            gpu_first = "cuda:1"
            
        if use_gpu == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(gpu_first)

        model.to(device)
        if use_gpu != "cpu" and multi_gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.eval()
        gpu_transfer_end = int(time.time() * 1000)

        # === Preprocessing ===
        preprocess_start = int(time.time() * 1000)
        inputs = tokenizer(sentence, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        preprocess_end = int(time.time() * 1000)

        # === Inference ===
        inference_start = int(time.time() * 1000)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        predicted_token_id = logits[0, mask_token_index].argmax(dim=-1)
        predicted_token = tokenizer.decode(predicted_token_id)
        inference_end = int(time.time() * 1000)

        # === Output ===
        output_start = int(time.time() * 1000)
        result_sentence = sentence.replace("[MASK]", predicted_token)
        output_end = int(time.time() * 1000)
        system_stats_after = get_system_metrics()

        return {
            "status": "success",
            "type": "bert",
            "output": result_sentence,
            "system_stats_before": system_stats_before,
            "system_stats_after": system_stats_after,
            "measurement": {
                "lib_load_start": lib_load_start,
                "lib_load_end": lib_load_end,
                "model_load_start": model_load_start,
                "model_load_end": model_load_end,
                "gpu_transfer_start": gpu_transfer_start,
                "gpu_transfer_end": gpu_transfer_end,
                "image_load_start": input_load_start,
                "image_load_end": input_load_end,
                "preprocess_start": preprocess_start,
                "preprocess_end": preprocess_end,
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
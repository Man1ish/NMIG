import time
# === Record library load time ===
lib_load_start = int(time.time() * 1000)

import sys
import warnings
import psutil
import json
import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b7
import argparse
warnings.filterwarnings("ignore")

lib_load_end = int(time.time() * 1000)

# === Utilities ===

def squirrel_load_model(model_path):
    return torch.load(model_path, map_location="cpu")

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

# === Main function ===
def main(args):
    try:
        system_stats_before = get_system_metrics()
        image_load_start = int(time.time() * 1000)

        use_gpu = args.get("device", "auto")
        multi_gpu = args.get("multi_gpu", False)
        local = args.get("local", False)
        gpu_first = args.get("gpu", "first")
        batch_size = args.get("batch", 1)
        no_of_gpu = 1

        script_dir = os.path.abspath(os.path.dirname(__file__))
        include_path = "../assets/" if local else "/"

        class_index_path = os.path.join(script_dir, include_path + "imagenet_classes.txt")
        image_path = os.path.join(script_dir, include_path + "image.jpg")
        if local:
            include_path = ""
        model_path = os.path.join(script_dir, include_path + "efficientnet_b7.pth")

        with open(class_index_path) as f:
            labels = [line.strip() for line in f.readlines()]

        input_image = Image.open(image_path).convert("RGB")
        image_load_end = int(time.time() * 1000)

        model_load_start = int(time.time() * 1000)
        model = efficientnet_b7(pretrained=False)
        model.load_state_dict(squirrel_load_model(model_path))
        model_load_end = int(time.time() * 1000)

        gpu_transfer_start = int(time.time() * 1000)
        if gpu_first == "first":
            gpu_first = "cuda"
        elif gpu_first == "second":
            gpu_first = "cuda:1"
        if use_gpu == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(gpu_first)

        if use_gpu != "cpu" and multi_gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            no_of_gpu = torch.cuda.device_count()

        model.to(device)
        model.eval()
        gpu_transfer_end = int(time.time() * 1000)

        preprocess_start = int(time.time() * 1000)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_batch = preprocess(input_image).unsqueeze(0).to(device)
        if no_of_gpu > 1:
            batch_size = no_of_gpu * batch_size
            input_batch = input_batch.repeat(batch_size, 1, 1, 1)

        preprocess_end = int(time.time() * 1000)

        inference_start = int(time.time() * 1000)
        output = model(input_batch)
        inference_end = int(time.time() * 1000)

        output_start = int(time.time() * 1000)
        _, indices = torch.max(output, 1)
        probs = torch.nn.functional.softmax(output, dim=1)
        result_val = [
            {
                "index": idx.item(),
                "class": labels[idx.item()],
                "confidence": round(probs[i][idx].item(), 4)
            }
            for i, idx in enumerate(indices)
        ]
        output_end = int(time.time() * 1000)
        system_stats_after = get_system_metrics()

        return {
            "status": "success",
            "result": result_val,
            "system_stats_before": system_stats_before,
            "system_stats_after": system_stats_after,
            "measurement": {
                "lib_load_start": lib_load_start,
                "lib_load_end": lib_load_end,
                "model_load_start": model_load_start,
                "model_load_end": model_load_end,
                "image_load_start": image_load_start,
                "image_load_end": image_load_end,
                "gpu_transfer_start": gpu_transfer_start,
                "gpu_transfer_end": gpu_transfer_end,
                "preprocess_start": preprocess_start,
                "preprocess_end": preprocess_end,
                "inference_start": inference_start,
                "inference_end": inference_end,
                "output_start": output_start,
                "output_end": output_end
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

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
    p.add_argument(
    "--batch", type=int, default=1,
    help="Batch size to use"
    )

    return vars(p.parse_args())

# === Entrypoint ===
if __name__ == "__main__":
    args = parse_args()
    
    try:
        # print(json.dumps(main(args)))
        # args = {
        #     "device": "gpu",
        #     "multi_gpu": True,
        #     "local": True
        # }
        t1 = time.time()
        result = main(args)

        # print(json.dumps(result))
        t2 = time.time()
        print(t2 - t1)
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({"status": "fatal_error", "message": str(e)}))
        sys.exit(1)

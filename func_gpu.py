import time
import psutil
import os
ts_imported = time.time_ns()

import torch
from torchvision import models, transforms
from PIL import Image
ts_libraries_loaded = time.time_ns()

# Memory tracking helper
def memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return round(mem_bytes / (1024 * 1024), 2)  # in MB

def main(args):
    ts_start = time.time_ns()
    mem_log = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_device_bound = time.time_ns()
    mem_log["after_device_binding"] = memory_usage_mb()

    model = models.alexnet(pretrained=True)
    ts_model_loaded = time.time_ns()
    mem_log["after_model_loading"] = memory_usage_mb()

    model = model.to(device)
    model.eval()
    ts_model_to_gpu = time.time_ns()
    mem_log["after_model_to_gpu"] = memory_usage_mb()

    image_path = args.get("image_path", "image.jpg")
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    ts_image_loaded = time.time_ns()
    mem_log["after_image_loaded"] = memory_usage_mb()

    input_tensor = input_tensor.to(device)
    ts_input_to_gpu = time.time_ns()
    mem_log["after_input_to_gpu"] = memory_usage_mb()

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    ts_inference_done = time.time_ns()
    mem_log["after_inference"] = memory_usage_mb()

    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]
    prediction = labels[pred.item()]
    ts_output_done = time.time_ns()
    mem_log["after_output"] = memory_usage_mb()

    def duration(ns_start, ns_end):
        ms = (ns_end - ns_start) / 1e6
        sec = ms / 1000
        return {"milliseconds": round(ms, 3), "seconds": round(sec, 6)}

    return {
        "prediction": prediction,
        "memory_mb": mem_log,
        "timing": {
            "library_loading": duration(ts_imported, ts_libraries_loaded),
            "device_binding": duration(ts_start, ts_device_bound),
            "model_loading": duration(ts_device_bound, ts_model_loaded),
            "model_to_gpu": duration(ts_model_loaded, ts_model_to_gpu),
            "image_loading": duration(ts_model_to_gpu, ts_image_loaded),
            "input_to_gpu": duration(ts_image_loaded, ts_input_to_gpu),
            "inference": duration(ts_input_to_gpu, ts_inference_done),
            "output_handling": duration(ts_inference_done, ts_output_done),
            "main_total": duration(ts_start, ts_output_done),
            "full_total": duration(ts_imported, ts_output_done)
        }
    }


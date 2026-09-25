import pynvml
import docker
import os
import time
import csv
from datetime import datetime

# Initialize NVML and Docker
pynvml.nvmlInit()
client = docker.from_env()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Create output directory and CSV file
log_dir = "results"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"gpu_usage_by_container_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "ContainerID", "ImageName", "GPU_Memory_MB"])
    try:
        while True:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)
            for p in processes:
                pid = p.pid
                try:
                    # Locate Docker container from GPU PID via /proc/<pid>/cgroup
                    with open(f"/proc/{pid}/cgroup", 'r') as cg:
                        lines = cg.readlines()
                        docker_lines = [line for line in lines if 'docker' in line or 'kubepods' in line]
                        if docker_lines:
                            cid = docker_lines[0].strip().split('/')[-1][:12]

                            # Get container info
                            try:
                                container = client.containers.get(cid)
                                image = container.image.tags[0] if container.image.tags else container.image.short_id
                            except docker.errors.NotFound:
                                image = "unknown"

                            mem_mb = round(p.usedGpuMemory / 1024 / 1024, 2)
                            timestamp = int(time.time())  # Unix timestamp
                            writer.writerow([timestamp, cid, image, mem_mb])
                except Exception:
                    continue

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Monitoring stopped.")
    finally:
        pynvml.nvmlShutdown()

import pynvml
import docker
import os
import time
import csv
from datetime import datetime

class GPUUsageMonitor:
    def __init__(self, log_dir="results", idle_grace_period=30, polling_interval=0.5):
        self.log_dir = log_dir
        self.idle_grace_period = idle_grace_period
        self.polling_interval = polling_interval
        self.simulation_done = False
        self._running = False

        # Setup NVML and Docker
        pynvml.nvmlInit()
        self.client = docker.from_env()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Log file setup
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"gpu_usage.csv")

    def notify_simulation_done(self):
        self.simulation_done = True
        print("[GPU MONITOR] Simulation complete. Monitoring will stop after grace period of GPU idleness.")

    def start(self):
        self._running = True
        idle_start_time = None

        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "ContainerID", "ImageName", "GPU_Memory_MB"])

            print("[GPU MONITOR] Started monitoring GPU usage...")

            while self._running:
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self.gpu_handle)
                except pynvml.NVMLError:
                    print("[GPU MONITOR] Error accessing NVML processes.")
                    break

                docker_gpu_process_found = False

                for p in processes:
                    pid = p.pid
                    try:
                        with open(f"/proc/{pid}/cgroup", 'r') as cg:
                            lines = cg.readlines()
                            docker_lines = [line for line in lines if 'docker' in line or 'kubepods' in line]
                            if docker_lines:
                                docker_gpu_process_found = True
                                cid = docker_lines[0].strip().split('/')[-1][:12]
                                try:
                                    container = self.client.containers.get(cid)
                                    image = container.image.tags[0] if container.image.tags else container.image.short_id
                                except docker.errors.NotFound:
                                    image = "unknown"

                                mem_mb = round(p.usedGpuMemory / 1024 / 1024, 2)
                                timestamp_sec = int(time.time())
                                timestamp_ms = int(timestamp_sec * 1000)
                                writer.writerow([timestamp_ms, cid, image, mem_mb])
                    except Exception:
                        continue

                if docker_gpu_process_found:
                    idle_start_time = None  # Reset timer
                elif self.simulation_done:
                    if idle_start_time is None:
                        idle_start_time = time.time()
                        print("[GPU MONITOR] Docker GPU idle detected. Grace timer started...")
                    elif time.time() - idle_start_time >= self.idle_grace_period:
                        print(f"[GPU MONITOR] No Docker GPU usage for {self.idle_grace_period}s. Stopping monitor.")
                        break

                time.sleep(self.polling_interval)

        pynvml.nvmlShutdown()
        print("[GPU MONITOR] Monitoring stopped.")

    def stop(self):
        self._running = False

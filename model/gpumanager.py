import pynvml

class GPUManager:
    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
        self.gpu_info = self._collect_gpu_metadata()

    def _collect_gpu_metadata(self):
        info = {}
        for i, handle in enumerate(self.handles):
            name = pynvml.nvmlDeviceGetName(handle)
            total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            info[f"GPU_{i}"] = {
                "name": name,
                "total_memory_mib": round(total / 1024**2, 2)
            }
        return info

    def get_metadata(self):
        return {
            "device_count": self.device_count,
            "gpus": self.gpu_info
        }

    def get_usage(self):
        usage = {}

        for i, handle in enumerate(self.handles):
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            usage[f"GPU_{i}"] = {
                # "memory_used_mib": round(mem.used / 1024**2, 2),
                # "gpu_util_percent": util.gpu,
                # "mem_util_percent": util.memory,
                # "total_memory_mib": round(total / 1024**2, 2)
                "memory_used_percent": round((mem.used / total) * 100, 2),
            }
        return usage

    def shutdown(self):
        pynvml.nvmlShutdown()

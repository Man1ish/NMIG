import subprocess
from datetime import datetime
import time


class FunctionExecutor:
    def __init__(self, **kwargs):
        self.functions = kwargs.get('functions')
        self.device = kwargs.get('device')
        self.multi_gpu = kwargs.get('multi_gpu')
        self.gpu_manager = kwargs.get('gpu_manager')
        self.function_policy = kwargs.get('function_policy')
        self.results = []
        self.event_type =  kwargs.get('event_type')

    def decide_device_and_batch(self,function_name: str, 
                            gpu_memory: dict, 
                            function_run_gpu_first, 
                            function_run_gpu_second, 
                            ) -> tuple[str, int]:
        """
        Decides device (cpu / first / second / multi-gpu) and batch size based on profiling config and available GPU memory.
        """
        # Default batch size fallback
        batch = 1
        config = self.function_policy

        # removing _p from the end of function name
        if function_name.endswith("_p"):
            function_name = function_name[:-2]
        
        if function_name in config:
            device_type = config[function_name]["device"]
            batch = config[function_name]["batch"]

            if device_type == "CPU":
                return "cpu", batch
            elif device_type == "GPU":
                gpu = self.assign_gpu(function_name, gpu_memory, function_run_gpu_first, function_run_gpu_second)
                # print(gpu)
                return gpu, batch
            elif device_type == "2xGPU":
                return "multi-gpu", batch
            else:
                raise ValueError(f"Unknown device type '{device_type}' for function '{function_name}'")
        else:
            # Fallback if no config found: use assign_gpu and default batch
            gpu = self.assign_gpu(function_name, gpu_memory, function_run_gpu_first, function_run_gpu_second)
            return gpu, batch



    def assign_gpu(self, function_name: str, gpu_memory: dict,function_run_gpu_first,function_run_gpu_second) -> str:
        # Map internal GPU IDs to final output
        label_map = {"GPU_0": "first", "GPU_1": "second"}
       
        # Set preference
        if function_name in function_run_gpu_first:
            preferred = "GPU_0"
            fallback = "GPU_1"
        elif function_name in function_run_gpu_second:
            preferred = "GPU_1"
            fallback = "GPU_0"
        else:
            selected = min(gpu_memory, key=lambda k: gpu_memory[k]['memory_used_percent'])
            return label_map[selected]

        thresholds = [80, 90, 95, 100]

        for threshold in thresholds:
            if gpu_memory[preferred]['memory_used_percent'] < threshold:
                return label_map[preferred]
            elif gpu_memory[fallback]['memory_used_percent'] < threshold:
                return label_map[fallback]

        # If nothing satisfies thresholds
        selected = min(gpu_memory, key=lambda k: gpu_memory[k]['memory_used_percent'])
        return label_map[selected]


    def handle_event(self, row):
        func_name = row['func']
        print(f"[EXECUTE] Function: {func_name} - Arrival Time: {row['arrival_time']} seconds")
        function_run_gpu_first = ["resnet50","inception","bert","efficientnet","resnet50_p","inception_p","bert_p","efficientnet_p"]
        function_run_gpu_second = ["alexnet","googlenet","distilgpt2","alexnet_p","googlenet_p","distilgpt2_p"]

        if self.event_type == 'proposed_process':
            function_path = self.functions[func_name]['func']
            function_path = function_path+"_p"
            

            input_array = ["wsk", "action", "invoke", function_path]

            if function_path in function_run_gpu_first:
                input_array.extend(["--param", "device", "gpu"]) 
                input_array.extend(["--param", "gpu", "first"])
            elif function_path in function_run_gpu_second:
                input_array.extend(["--param", "device", "gpu"]) 
                input_array.extend(["--param", "gpu", "second"])

        elif self.event_type == 'proposed':
            pass

        elif self.event_type == 'normal':
            if func_name.endswith("_p"):
                func_name = func_name[:-2]

            function_path = self.functions[func_name]['func']
            
            # gpu = self.assign_gpu(function_path, self.gpu_manager.get_usage(), function_run_gpu_first, function_run_gpu_second)
            input_array = ["wsk", "action", "invoke", function_path]

            if function_path in function_run_gpu_first:
                input_array.extend(["--param", "device", "gpu"]) 
                input_array.extend(["--param", "gpu", "first"])
            elif function_path in function_run_gpu_second:
                input_array.extend(["--param", "device", "gpu"]) 
                input_array.extend(["--param", "gpu", "second"])
        # The `else` block in the `handle_event` method is used to handle the case where the `if`
        # condition before it evaluates to `False`. In this specific context, the `else` block is
        # executed when the `if` condition `if self.event_type == 'normal':` is not met.
        elif self.event_type == 'proposed_process_profiler':
            
            function_path = self.functions[func_name]['func']
            function_path = function_path+"_p"
            # function_can_run_cpu = ["resnet50","inception","googlenet"]

            input_array = ["wsk", "action", "invoke", function_path]
                
            device, batch = self.decide_device_and_batch(function_path, self.gpu_manager.get_usage(),function_run_gpu_first,function_run_gpu_second)
            
            if device == "cpu":
                input_array.extend(["--param", "device", "cpu"]) 
            elif device == "multi-gpu":
                input_array.extend(["--param", "device", "gpu"])
                input_array.extend(["--param", "multi_gpu","True"])
            elif device == "first":
                input_array.extend(["--param", "device", "gpu"]) 
                input_array.extend(["--param", "gpu", "first"])
            elif device == "second":
                input_array.extend(["--param", "device", "gpu"]) 
                input_array.extend(["--param", "gpu", "second"])

            input_array.extend(["--param", "batch", str(batch)])

       
        try:
            

            result = subprocess.run(
                input_array,
                capture_output=True, text=True
            )

            output = result.stdout.strip()

            # Extract ID using known pattern
            if "with id" in output:
                activation_id = output.split("with id")[-1].strip()
                timestamp_sec = int(time.time())
                timestamp_ms = int(timestamp_sec * 1000)

                self.results.append({
                    'timestamp': timestamp_ms,
                    'func': func_name,
                    'activation_id': activation_id
                })
            else:
                print(f"[ERROR] Activation ID not found of func {function_path}.")
                print(output)
        except Exception as e:
            print(f"[ERROR] Subprocess failed for {func_name}: {e}")

    def get_results(self):
        return self.results
import pandas as pd
import sys
import time
import argparse
from model.dataset import DatasetLoader
from model.simulate import Simulator, TimeBasedExecution,OneTimeExecution,BrustyExecution,CustomExecution
from model.executor import FunctionExecutor
from model.gpu_monitor import GPUUsageMonitor
from model.gpumanager import GPUManager
from model.experiment_utils import create_experiment_folder
from model.database import SQLiteDatabase
import ast
from utils.activation_helpers import ActivationHelper
import threading
import requests


if __name__ == "__main__":
    try:
        # parse arguments
        parser = argparse.ArgumentParser()
        #import data
        parser.add_argument('--data', type=str, default='dataset/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt')
        parser.add_argument('--function_path', type=str, default='dataset/functions.csv')
        parser.add_argument('--threshold', type=int, default=300)
        parser.add_argument('--top_functions_no', type=int, default=7)
        parser.add_argument('--device', type=str, default='gpu')
        # parser.add_argument('--multi_gpu', type=bool, default=True)
        parser.add_argument('--threshold_stop', type=int, default=14400)
        parser.add_argument('--idle_grace_period', type=int, default=30) # For GPU monitor
        parser.add_argument('--polling_interval', type=int, default=0.5)
        parser.add_argument('--monitor_gpu', type=bool, default=True)
        parser.add_argument('--database', type=str, default='database.db')
        parser.add_argument('--function_pattern', type=str, default='normal') # normal, bursty
        parser.add_argument('--hour_trace', type=int, default=4)

        parser.add_argument('--dataset', type=str, default="normal_gen") # Type of dataset bursty_gen, similar_gen, normal_gen, hybrid_gen,control_trace
        parser.add_argument('--method', type=str, default='TimeBasedExecution') #OneTimeExecution,TimeBasedExecution, CustomExecution
        parser.add_argument('--event_type', type=str, default='proposed_process_profiler') # proposed, normal, proposed_process, proposed_process_profiler
        parser.add_argument('--method_type', type=str, default='proposed') # proposed, hist, openwhisk
        parser.add_argument('--activation', type=str, default='normal') # normal, bursty

        args = parser.parse_args()

        # Convert args to dictionary
        args_dict = vars(args)



    # Convert string to boolean
        experiment_path = create_experiment_folder(params=args_dict)
        
        
        # Step 1: Conditionally start GPU monitor thread
        if args_dict['monitor_gpu']:
            gpu_monitor = GPUUsageMonitor(
                idle_grace_period=args_dict['idle_grace_period'],
                log_dir=experiment_path,
                polling_interval=args_dict['polling_interval']
            )
            gpu_thread = threading.Thread(target=gpu_monitor.start)
            gpu_thread.start()

            url = "http://localhost:8001/start-recording"
            payload = {"folder": experiment_path}
            headers = {"Content-Type": "application/json"}

            try:
                response = requests.post(url, json=payload, headers=headers, timeout=2)
                print("Start recording response:", response.json())
            except requests.RequestException as e:
                print("Failed to start recording:", e)
        else:
            gpu_monitor = None
            gpu_thread = None
 

        # Step 2: Load dataset and functions
        dataset = DatasetLoader(**args_dict)
        df = dataset.filtered_df

        

        
        functions = dataset.functions
       
        function_policy = dataset.function_policy
    

        if args_dict['method'] == 'TimeBasedExecution':
            strategy = TimeBasedExecution()
        elif args_dict['method'] == 'OneTimeExecution':
            strategy = OneTimeExecution()
        elif args_dict['method'] == 'BrustyExecution':
            strategy = BrustyExecution()
        elif args_dict['method'] == 'NormalExecution':
            strategy = BrustyExecution()
        elif args_dict['method'] == 'CustomExecution':
            strategy = CustomExecution()

        gpu_manager = GPUManager()
        executor = FunctionExecutor(**args_dict,functions=functions,gpu_manager=gpu_manager,function_policy=function_policy)
        
        simulator = Simulator(df, executor,strategy=strategy,threshold_stop=args_dict['threshold_stop'])
        simulator.simulate()

        # save results
        results = executor.get_results()

        # Save the results in the experiment folder, results is a list of dictionaries
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{experiment_path}/results.csv",
                        index=False)

        # Step 3: Notify and join GPU monitor if it was used
        if gpu_monitor:
            gpu_monitor.notify_simulation_done()
            gpu_thread.join()

        time.sleep(10)

        url = "http://localhost:8001/stop-recording"
        payload = {"folder": experiment_path}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=2)
            print("Start recording response:", response.json())
        except requests.RequestException as e:
            print("Failed to start recording:", e)

        # Step 4: Save results to database
        db = SQLiteDatabase(args_dict['database'])
        ah = ActivationHelper()
        # first the array 
        for result in results:
            temp = {}
            temp.update(result)
            output = ah.get_activation(result['activation_id'])
            temp['json_obj'] = output
            temp['experiment_id'] = experiment_path
            db.save_data(**temp)
    except KeyboardInterrupt:
        print("Caught Ctrl+C. Sending stop-recording request...")
        try:
            requests.post("http://localhost:8001/stop-recording")
            print("Stop request sent.")
        except Exception as e:
            print("Failed to send stop request:", e)
        
        sys.exit(0)  # Clean exit
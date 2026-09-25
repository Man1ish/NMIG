import os
import json
from datetime import datetime

def create_experiment_folder(base_dir="results", params=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"exp_{timestamp}"
    experiment_path = os.path.join(base_dir, folder_name)
    os.makedirs(experiment_path, exist_ok=True)

    # Save parameters
    if params:
        with open(os.path.join(experiment_path, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

    return experiment_path

import pandas as pd
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser(description="Demo script")
parser.add_argument("--experiment_id", help="Experiment Id")

if len(sys.argv) < 2:
    print("You need to provide args")
    sys.exit(1)
args = parser.parse_args()

# Experiment ID and path
experiment_id = args.experiment_id
# experiment_id = "exp_20250520_213401"
path = f"../results/{experiment_id}/results.csv"

# Load CSV file into DataFrame
df = pd.read_csv(path)

# Initialize a list to store results
results = []

# Process each activation ID
for idx, row in df.iterrows():
    activation_id = str(row.get("activation_id", "")).strip()
    
    if not activation_id:
        results.append(None)
        continue

    try:
        # Run the wsk command and capture output
        result = subprocess.run(
            ["wsk", "activation", "get", activation_id],
            capture_output=True,
            text=True
        )

        # Prefer stdout, fallback to stderr
        raw_output = result.stdout.strip() or result.stderr.strip()

        # Extract from first '{' if it exists
        if "{" in raw_output:
            json_part = raw_output[raw_output.index("{"):]
            results.append(json_part)
        else:
            results.append(None)

    except Exception as e:
        results.append(None)

# Assign results and save
df["result"] = results
df.to_csv(f"../results/{experiment_id}/results_updated.csv", index=False)

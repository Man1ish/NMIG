import os
import sqlite3
import sys
import argparse

import pandas as pd

parser = argparse.ArgumentParser(description="Build results_updated.csv from database.db instead of `wsk activation get`")
parser.add_argument("--experiment_id", help="Experiment Id (e.g. exp_20250520_213401)")
parser.add_argument("--output", help="Output CSV path (default: results/<experiment_id>/results_updated_db.csv)")
parser.add_argument("--force", action="store_true", help="Overwrite output file if it already exists")

if len(sys.argv) < 2:
    print("You need to provide args")
    sys.exit(1)
args = parser.parse_args()

experiment_id = args.experiment_id
results_path = f"results/{experiment_id}/results.csv"
out_path = args.output or f"results/{experiment_id}/results_updated.csv"

if os.path.exists(out_path) and not args.force:
    print(f"{out_path} already exists. Use --force to overwrite.")
    sys.exit(1)

df = pd.read_csv(results_path)

with sqlite3.connect("database.db") as conn:
    cur = conn.execute("SELECT activation_id, json_data FROM activations")
    json_by_activation_id = dict(cur.fetchall())

missing = 0
results = []
for activation_id in df["activation_id"].astype(str).str.strip():
    result = json_by_activation_id.get(activation_id)
    if result is None:
        missing += 1
    results.append(result)

df["result"] = results
df.to_csv(out_path, index=False)

if missing:
    print(f"Warning: {missing}/{len(df)} activation_id(s) not found in database.db")
print(f"Wrote {len(df)} rows to {out_path}")

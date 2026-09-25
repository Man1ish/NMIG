import os
import pandas as pd
import numpy as np

# ============================================================
# 1. Create dataset folder
# ============================================================

os.makedirs("dataset", exist_ok=True)


# ============================================================
# 2. Function metadata, this will be saved as functions CSV
# ============================================================

functions_data = [
    {
        "func": "alexnet",
        "trace_id": 59,
        "category_id": 1,
        "id": "1f33077bb2db73fd76a51e40e43f20173e3f038dd3f3705551751cf7724c3c40",
        "memory": 1024,
        "file_path": "alexnet.py"
    },
    {
        "func": "efficientnet",
        "trace_id": 59,
        "category_id": 1,
        "id": "cd9f7f333d59aede8088772edac3bff78c45e2dad8eb498b450b9af511452a36",
        "memory": 1024,
        "file_path": "efficientnet.py"
    },
    {
        "func": "bert",
        "trace_id": 52,
        "category_id": 2,
        "id": "9b61fd55aa093a2d172db1a68a60af5cf6cbfa7f5ea1fbc71846027a5954616d",
        "memory": 1024,
        "file_path": "bert.py"
    },
    {
        "func": "distilgpt2",
        "trace_id": 52,
        "category_id": 2,
        "id": "762835950e81a11cd04cedcb05275dc111c651625d575077fce49f82170e0986",
        "memory": 1024,
        "file_path": "distilgpt2.py"
    },
    {
        "func": "resnet50",
        "trace_id": 52,
        "category_id": 3,
        "id": "c9f8e30e36d1aef62c10b3cfca6e289a93848a148d876dd514753040314f4817",
        "memory": 1024,
        "file_path": "resnet50.py"
    },
    {
        "func": "googlenet",
        "trace_id": 3,
        "category_id": 3,
        "id": "313c03f53a0d31f70aec25f62efb33e7dd779725ca4af579018452d1204beaad",
        "memory": 1024,
        "file_path": "googlenet.py"
    },
    {
        "func": "inception",
        "trace_id": 3,
        "category_id": 3,
        "id": "556ccf8758c8c2a20082c161e955405e950439f0503522fe129e709a5dc0e58f",
        "memory": 1024,
        "file_path": "inception.py"
    },
]

functions_df = pd.DataFrame(functions_data)

functions_csv_path = "dataset/controlled_1hour_500_requests_functions.csv"
functions_df.to_csv(functions_csv_path, index=False)


# ============================================================
# 3. Build helper dictionaries from function metadata
# ============================================================

FUNCTION_META = {
    row["func"]: {
        "trace_id": row["trace_id"],
        "category_id": row["category_id"],
        "id": row["id"],
        "memory": row["memory"],
        "file_path": row["file_path"]
    }
    for _, row in functions_df.iterrows()
}

CATEGORY_TO_FUNCTIONS = {}

for _, row in functions_df.iterrows():
    category_id = int(row["category_id"])
    func_name = row["func"]

    if category_id not in CATEGORY_TO_FUNCTIONS:
        CATEGORY_TO_FUNCTIONS[category_id] = []

    CATEGORY_TO_FUNCTIONS[category_id].append(func_name)


# ============================================================
# 4. Generate one phase of arrivals
# ============================================================

def generate_phase(start_time, end_time, num_requests, category_probs, phase_name, seed=42):
    """
    Generate exactly num_requests between start_time and end_time.

    Important:
    In controlled_1hour_500_requests.csv, the 'func' column stores
    the function ID, not the readable function name.
    """

    rng = np.random.default_rng(seed)

    categories = list(category_probs.keys())
    probs = np.array([category_probs[c] for c in categories], dtype=float)
    probs = probs / probs.sum()

    time_range = np.arange(start_time, end_time)

    arrival_times = rng.choice(
        time_range,
        size=num_requests,
        replace=False if len(time_range) >= num_requests else True
    )

    arrival_times = np.sort(arrival_times)

    rows = []

    for t in arrival_times:
        category_id = int(rng.choice(categories, p=probs))

        selected_func_name = str(rng.choice(CATEGORY_TO_FUNCTIONS[category_id]))

        meta = FUNCTION_META[selected_func_name]

        rows.append({
            "arrival_time": int(t),

            # func is the function ID
            "func": meta["id"],

            "trace_id": int(meta["trace_id"]),
            "category_id": int(meta["category_id"]),
            "memory": int(meta["memory"]),
            "file_path": meta["file_path"],
            "phase": phase_name
        })

    return rows


# ============================================================
# 5. Generate complete 1-hour, 500-request trace
# ============================================================

def generate_1hour_500_trace(seed=42):
    rows = []

    phases = [
    # start, end, requests, category probabilities, phase name

    # 0 to 10 min, category 1 appears slowly as background
    (0, 600, 40, {1: 0.30, 2: 0.50, 3: 0.20}, "category_1_slow_background"),

    # 10 to 20 min, category 2 dominant
    (600, 1200, 60, {1: 0.05, 2: 0.85, 3: 0.10}, "category_2_dominant"),

    # 20 to 30 min, category 3 dominant
    (1200, 1800, 60, {1: 0.05, 2: 0.10, 3: 0.85}, "category_3_dominant"),

    # 30 to 45 min, mixed traffic, category 2 and 3 stronger
    (1800, 2700, 70, {1: 0.10, 2: 0.45, 3: 0.45}, "mixed_category_2_3"),

    # 45 to 60 min, category 2 and 3 burst together
    (2700, 3600, 70, {1: 0.05, 2: 0.50, 3: 0.45}, "category_2_3_burst"),
] 

    for i, phase in enumerate(phases):
        start, end, num_requests, category_probs, phase_name = phase

        rows.extend(
            generate_phase(
                start_time=start,
                end_time=end,
                num_requests=num_requests,
                category_probs=category_probs,
                phase_name=phase_name,
                seed=seed + i
            )
        )

    trace_df = pd.DataFrame(rows)
    trace_df = trace_df.sort_values("arrival_time").reset_index(drop=True)
    trace_df.insert(0, "request_id", range(1, len(trace_df) + 1))

    return trace_df


# ============================================================
# 6. Save invocation trace CSV
# ============================================================

trace_df = generate_1hour_500_trace(seed=42)

trace_csv_path = "dataset/controlled_1hour_500_requests.csv"
trace_df.to_csv(trace_csv_path, index=False)


# ============================================================
# 7. Print summary
# ============================================================

print("Saved function metadata CSV:")
print(functions_csv_path)

print("\nSaved invocation trace CSV:")
print(trace_csv_path)

print("\nTotal requests:")
print(len(trace_df))

print("\nFunction metadata CSV preview:")
print(functions_df.head())

print("\nInvocation trace CSV preview:")
print(trace_df.head())

print("\nCategory distribution in invocation trace:")
print(trace_df["category_id"].value_counts().sort_index())

print("\nFunction ID distribution in invocation trace:")
print(trace_df["func"].value_counts())
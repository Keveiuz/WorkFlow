import json
import pandas as pd
from pathlib import Path

def load_jsonl(file_path):
    """Load JSONL file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_parquet(data, file_path):
    """Save data to Parquet file"""
    df = pd.DataFrame(data)
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)

def format_data(input_file, output_file, mode):
    """
    Format data based on mode and save as Parquet
    mode ∈ {"dpo", "csft"}
    """
    data = load_jsonl(input_file)

    print(f"[INFO] Mode: {mode}")
    if mode == "dpo":
        # Save all fields to parquet
        save_parquet(data, output_file)
        print(f"[INFO] {len(data)} items saved to {output_file}")

    elif mode == "csft":
        # Keep only 'chosen' and rename to 'conversations'
        csft_data = []
        for item in data:
            if "chosen" in item:
                csft_data.append({
                    "question": item["chosen"][0]["content"],
                    "response": item["chosen"][1]["content"],
                })
        save_parquet(csft_data, output_file)
        print(f"[INFO] {len(csft_data)} items saved to {output_file}")

    else:
        raise ValueError("Mode must be 'dpo' or 'csft'")

if __name__ == "__main__":
    # ===== Manual input parameters =====
    experiment_epoch = "4"
    dataset = "guardreasoner"
    sample_mode = "boundary"     # ["in-distribution", "boundary", "outlier"]
    train_mode = "dpo"        # ["dpo", "csft"]
    sft_n = 100
    po_n = 4500

    if type(sample_mode) is str:
        input_file = f"experiment-{experiment_epoch}/temp/{dataset}-{sample_mode}-{po_n}.jsonl"
        output_file = f"experiment-{experiment_epoch}/train/{dataset}-{sft_n}K-{sample_mode}-{train_mode}-{po_n}.parquet"
    elif type(sample_mode) is list:
        t = "_".join(sample_mode)
        input_file = f"experiment-{experiment_epoch}/temp/{dataset}-{t}-mix-{po_n}.jsonl"
        output_file = f"experiment-{experiment_epoch}/train/{dataset}-{sft_n}K-{t}-mix_{train_mode}.parquet"
    else:
        raise ValueError("Invalid sample_mode")

    format_data(input_file, output_file, train_mode)

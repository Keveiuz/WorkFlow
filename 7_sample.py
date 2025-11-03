import json
import random
from pathlib import Path

def load_jsonl(file_path):
    """Load JSONL file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, file_path):
    """Save data to JSONL file"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def random_sample(data, n, seed=42):
    """Randomly sample n items from a list"""
    random.seed(seed)
    n = min(n, len(data))
    return random.sample(data, n)


def sample_data(file_map, output_file, n, mode, seed=42):
    """
    Sample data based on the mode
    mode ∈ {"boundary", "outlier", "mix"}
    """
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Number of samples: {n}")

    if type(mode) is str:
        data = load_jsonl(file_map[mode])
        sampled = random_sample(data, n, seed)

        

    elif type(mode) is list:
        data = []
        for sample_type in mode:
            type_data = load_jsonl(file_map[sample_type])
            data.extend(type_data)
        sampled = random_sample(data, n, seed)

    else:
        raise TypeError("Mode must be str or list, str for single type sample, list for mix sample")

    save_jsonl(sampled, output_file)
    print(f"[INFO] Sampling completed: {len(sampled)} items saved to {output_file}")


if __name__ == "__main__":
    # ===== Manual input parameters =====
    experiment_epoch = "4"
    dataset = "guardreasoner"
    sample_mode = "outlier"     # ["in_distribution", "boundary", "outlier"]
    sft_n = 100                         # sft sample size (K)
    po_n = 4500                         # sample use for optimization = min(boundary, outlier, in_distribution)

    outlier_file = f"experiment-{experiment_epoch}/temp/{dataset}-outlier.jsonl"
    boundary_file = f"experiment-{experiment_epoch}/temp/{dataset}-boundary.jsonl"
    in_distribution_file = f"experiment-{experiment_epoch}/temp/{dataset}-in_distribution.jsonl"

    if type(sample_mode) is str:
        output_file = f"experiment-{experiment_epoch}/temp/{dataset}-{sample_mode}-{po_n}.jsonl"
    elif type(sample_mode) is list:
        t = "_".join(sample_mode)
        output_file = f"experiment-{experiment_epoch}/temp/{dataset}-{t}-mix-{po_n}.jsonl"

    file_map = {
        "outlier": outlier_file,
        "boundary": boundary_file,
        "in_distribution": in_distribution_file,
    }
    
    sample_data(file_map, output_file, po_n, sample_mode)

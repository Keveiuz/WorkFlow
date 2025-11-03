import json
import random
from tqdm import tqdm

# 可选：设定随机种子以便可复现（取消注释并设置seed以固定shuffle）
random.seed(42)

# 文件路径
file_path = "experiment-4/temp/guardreasoner-filtered.jsonl"
output_path = "experiment-4/temp/guardreasoner-sampled.jsonl"

# 配置每个类别要采样的数量
sample_config = {
    "user_harmful": 25000,
    "user_unharmful": 25000,
    "assistant_harmful": 24000,
    "assistant_unharmful": 26000,
}

# 读取 JSONL 文件
def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 写入 JSONL 文件
def write_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"[INFO] Loading jsonl file {file_path} ...")
data = read_jsonl(file_path)

# 构建 id -> 数据 的字典（避免使用内建名 dict）
data_dict = {item["id"]: item for item in data}

# 找到 id 
ids = list(data_dict.keys())

# 分类存放 id
categories = {
    "user_harmful": [],
    "user_unharmful": [],
    "assistant_harmful": [],
    "assistant_unharmful": [],
}

for i in tqdm(ids, desc="Categorizing"):
    d1 = data_dict[i]
    response_harm = d1.get("response_harm_label")
    prompt_harm = d1.get("prompt_harm_label")

    if response_harm is None:  # 属于 user 端
        if prompt_harm == "harmful":
            categories["user_harmful"].append(i)
        else:
            categories["user_unharmful"].append(i)
    else:  # 属于 assistant 端
        if response_harm == "harmful":
            categories["assistant_harmful"].append(i)
        else:
            categories["assistant_unharmful"].append(i)

# 按照采样配置随机抽取
selected_ids = []
print("[INFO] Distribution before sampling: ")
for cat, id_list in categories.items():
    print(f"  {cat}: {len(id_list)}")
    n = sample_config.get(cat, 0)
    if n > 0:
        if len(id_list) <= n:
            chosen = id_list.copy()
        else:
            chosen = random.sample(id_list, n)
        selected_ids.extend(chosen)

# 如果希望先按 id 排序再打乱也可以，但这里直接乱序 final 结果
# selected_ids = sorted(selected_ids)

# ========== 这里执行乱序 shuffle ==============
random.shuffle(selected_ids)

# ========= 采样后分布统计 =========
sampled_categories = {k: 0 for k in categories.keys()}
for i in selected_ids:
    d1 = data_dict[i]
    response_harm = d1.get("response_harm_label")
    label = d1.get("label")
    prompt_harm = d1.get("prompt_harm_label")

    if response_harm is None:
        if label == "harmful" or prompt_harm == "harmful":
            sampled_categories["user_harmful"] += 1
        else:
            sampled_categories["user_unharmful"] += 1
    else:
        if response_harm == "harmful":
            sampled_categories["assistant_harmful"] += 1
        else:
            sampled_categories["assistant_unharmful"] += 1

print("\n[INFO] Distribution after sampling: ")
for cat, count in sampled_categories.items():
    print(f"  {cat}: {count}")

# 根据选择的 id 筛选数据（此处保持与 selected_ids 的顺序一致，即已被 shuffle）
filtered_data = [data_dict[i] for i in selected_ids]

print(f"\n[INFO] Sampling finished, total selected {len(selected_ids)} datapieces.")

# 写入新 JSONL 文件
print(f"[INFO] Saving data to {output_path} ...")
write_jsonl(filtered_data, output_path)
print("[INFO] Done.")

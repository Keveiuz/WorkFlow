import ray
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from typing import List, Dict
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import resource
import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

@dataclass
class Config:
    input_file: str = "experiment-4/temp/guardreasoner-sampled.jsonl"
    output_file: str = "experiment-4/temp/guardreasoner-inference-50K.jsonl"
    model_path: str = "experiment-4/model/guardreasoner-sft-50K"

    max_model_len: int = 4096
    max_tokens: int = 2048
    dtype: str = "float16"

    temperature: float = 0.7
    top_p: float = 0.9
    n: int = 8

    node_number: int = 4
    num_gpus: int = 1
    tensor_parallel_size: int = 1

    gpu_memory_utilization: float = 0.85
    num_models: int = 4
    total_gpus: int = 4


config = Config()


def load_dataset(file_path: str, num_queries: int = None) -> List[Dict]:
    """从文件加载数据集"""
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            if num_queries is not None:
                df = df.head(num_queries)
            return df.to_dict('records')
        elif file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if num_queries is not None and i >= num_queries:
                        break
                    data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"不支持的输入文件格式: {file_path}")
    except Exception as e:
        raise ValueError(f"无法加载文件: {file_path}, 错误: {str(e)}")


def save_data(data: List[Dict], file_path: str):
    """保存数据"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif path.suffix == '.jsonl':
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    elif path.suffix == '.parquet':
        pd.DataFrame(data).to_parquet(path, index=False)
    else:
        raise ValueError(f"不支持的格式: {path.suffix}")


def set_resource_limits():
    """设置系统资源限制"""
    try:
        # 设置进程数限制
        resource.setrlimit(resource.RLIMIT_NPROC, (65536, 65536))
    
        print("Resource limits set successfully")
        
    except Exception as e:
        print(f"Warning: Could not set resource limits: {e}")


# 使用Actor模式预加载模型
@ray.remote(num_gpus=config.num_gpus, num_cpus=4)
class Qwen3Model:
    def __init__(self, model_id: int, config: Config):
        self.model_id = model_id
        self.config = config
        
        print(f"Loading model {model_id}...")
        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
            trust_remote_code=True,
            enforce_eager=True,
        )
        
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            n=config.n
        )
        print(f"Model {model_id} loaded successfully!")
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """批量生成文本"""
        try:
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [ [resp.text for resp in output.outputs] for output in outputs ]
        except Exception as e:
            error_msg = f"Error in model {self.model_id}: {str(e)}"
            return [error_msg] * len(prompts)


def distribute_prompts(prompts: List[str], num_models: int) -> List[List[str]]:
    """将prompts平均分配给模型"""
    distributed = [[] for _ in range(num_models)]
    for i, prompt in enumerate(prompts):
        model_index = i % num_models
        distributed[model_index].append(prompt)
    return distributed


def main():
    # 设置资源限制
    set_resource_limits()
    
    # 初始化Ray
    ray.init(num_gpus=config.total_gpus)
    
    try:
        # 预加载所有模型
        print("Pre-loading models...")
        models = []
        
        # 创建模型实例
        for i in range(config.num_models):
            model = Qwen3Model.options(name=f"qwen3_model_{i+1}").remote(i+1, config)
            models.append(model)
        
        print(f"Waiting for all {config.num_models} models to load...")
        # 等待所有模型加载完成
        test_results = ray.get([model.generate_batch.remote(["Hello"]) for model in models])
        print("All models loaded!")
        
        # 加载tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)

        # 加载数据集
        print(f"Loading data from: {config.input_file}")
        data = load_dataset(config.input_file)

        # 处理每个对话
        print("Filtering prompts...")
        prompts = []
        new_data = []

        for item in tqdm(data, desc="Filter Prompt"):
            prompt = tokenizer.apply_chat_template(
                [item["conversations"][0]],
                tokenize=False,
                add_generation_prompt=True,
            )

            # 检查长度
            tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            num_prompt_tokens = tokenized.input_ids.shape[1]
            if num_prompt_tokens > config.max_model_len:
                print(f"[跳过] prompt 超过最大长度: {num_prompt_tokens} > {config.max_model_len}")
                continue
            new_data.append(item)
            prompts.append(prompt)

        print(f"✅ Filtered {len(prompts)} prompts for generation")

        # 将prompts平均分配给模型
        distributed_prompts = distribute_prompts(prompts, config.num_models)
        
        # 为每个模型创建批量推理任务
        print("Creating batch inference tasks...")
        tasks = []
        for i, (model, batch_prompts) in enumerate(zip(models, distributed_prompts)):
            print(f"Model {i+1} will process {len(batch_prompts)} prompts")
            if batch_prompts:  # 只有当有prompts时才创建任务
                task = model.generate_batch.remote(batch_prompts)
                tasks.append(task)
            else:
                tasks.append(None)  # 没有prompts的模型
        
        # 批量获取所有结果
        print("Running batch inference...")
        all_results = []
        for i, task in enumerate(tasks):
            if task is not None:
                batch_results = ray.get(task)
                all_results.extend(batch_results)
                print(f"Model {i+1} completed {len(batch_results)} prompts")
            else:
                print(f"Model {i+1} had no prompts to process")
        
        # 将结果添加到数据中（保持原始顺序）
        prompt_to_result = {}
        result_index = 0
        
        # 重新分配结果到对应的prompt
        for i, batch_prompts in enumerate(distributed_prompts):
            for j in range(len(batch_prompts)):
                if result_index < len(all_results):
                    # 找到这个prompt在原始prompts中的位置
                    original_index = i + j * config.num_models
                    if original_index < len(prompts):
                        prompt_to_result[original_index] = all_results[result_index]
                        result_index += 1
        
        # 按照原始顺序整理结果
        final_results = []
        for i in range(len(prompts)):
            final_results.append(prompt_to_result.get(i, "Error: Result not found"))
        
        # 将结果添加到数据中
        for i, (item, result) in enumerate(zip(new_data, final_results)):
            item['candidates'] = result

        # 保存结果
        print(f"Saving results to: {config.output_file}")
        save_data(new_data, config.output_file)
        
        print("✅ Inference completed successfully!")
        print(f"Processed {len(final_results)} prompts")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

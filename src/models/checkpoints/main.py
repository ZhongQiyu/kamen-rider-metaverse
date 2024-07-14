# main.py

import os
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def save_model_and_tokenizer(model_name, save_directory):
    # 创建保存目录
    os.makedirs(save_directory, exist_ok=True)

    # 加载模型和分词器
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 保存模型权重
    model.save_pretrained(save_directory)

    # 保存分词器
    tokenizer.save_pretrained(save_directory)

    # 创建配置文件
    config = model.config.to_dict()
    config_path = os.path.join(save_directory, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 保存词汇表文件
    vocab_path = os.path.join(save_directory, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=4)

    # 保存其他辅助文件
    tokenizer.save_pretrained(save_directory)

    # 如果有 added_tokens.json 文件，也可以一并保存
    added_tokens_path = os.path.join(save_directory, 'added_tokens.json')
    if os.path.exists(added_tokens_path):
        with open(added_tokens_path, 'w') as f:
            json.dump(tokenizer.special_tokens_map, f, ensure_ascii=False, indent=4)

    print(f"Model and tokenizer files have been saved to {save_directory}")

if __name__ == "__main__":
    model_name = "gpt2"  # 你可以替换为任何其他模型名称，例如 "gpt-3", "t5-large" 等
    save_directory = "models/checkpoints/gpt2"
    save_model_and_tokenizer(model_name, save_directory)

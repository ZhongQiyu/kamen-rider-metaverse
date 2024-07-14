# rlhf.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AdamW, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import random
import argparse
import os
from utils.data_loader import load_data, load_config, parse_train_data, parse_asr_data
from nltk.translate.bleu_score import sentence_bleu

class StreamingInference:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    def stream_infer(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').cuda()
        outputs = self.model.generate(inputs['input_ids'], max_length=50, stream=True)
        for output in outputs:
            yield self.tokenizer.decode(output, skip_special_tokens=True)

class DialogueModelTrainer:
    def __init__(self, model_name, train_data, lr=5e-5, num_epochs=3):
        self.model_name = model_name
        self.train_data = train_data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs

    def reward_function(self, generated_response, ideal_response):
        # 使用BLEU分数作为奖励
        reference = [ideal_response.split()]
        candidate = generated_response.split()
        reward = sentence_bleu(reference, candidate)
        return reward

    def train_step(self, input_text, ideal_response):
        self.model.train()

        inputs = self.tokenizer(input_text, return_tensors='pt').input_ids.cuda()
        ideal_outputs = self.tokenizer(ideal_response, return_tensors='pt').input_ids.cuda()

        outputs = self.model.generate(inputs, max_length=50)
        generated_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 计算奖励
        reward = self.reward_function(generated_response, ideal_response)

        # 计算损失并进行反向传播
        model_output = self.model(inputs, labels=ideal_outputs)
        loss = model_output.loss
        reward_loss = loss * reward
        reward_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item(), reward

    def train(self):
        for epoch in range(self.num_epochs):
            random.shuffle(self.train_data)
            total_loss = 0
            total_reward = 0
            for data in self.train_data:
                input_text = data['input']
                ideal_response = data['ideal_response']
                loss, reward = self.train_step(input_text, ideal_response)
                total_loss += loss
                total_reward += reward
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.train_data)}, Reward: {total_reward/len(self.train_data)}")

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

class BERTFineTuner:
    def __init__(self, model_name, data, num_labels=2, num_epochs=3):
        self.model_name = model_name
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        if torch.cuda.is_available():
            self.model.cuda()
        self.num_epochs = num_epochs

    def preprocess_function(self, examples):
        return self.tokenizer(examples['texts'], truncation=True, padding=True)

    def train(self):
        dataset = Dataset.from_dict(self.data)
        encoded_dataset = dataset.map(self.preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        metric = load_metric("accuracy")

        def compute_metrics(p):
            preds = p.predictions.argmax(-1)
            return {'accuracy': (preds == p.label_ids).astype(float).mean().item()}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_dataset,
            eval_dataset=encoded_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        self.save_model('./finetuned_model')

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

class ASRInference:
    def __init__(self, model_names):
        self.models = {name: AutoModelForSeq2SeqLM.from_pretrained(name) for name in model_names}
        self.tokenizers = {name: AutoTokenizer.from_pretrained(name) for name in model_names}
        if torch.cuda.is_available():
            for model in self.models.values():
                model.cuda()

    def generate_response(self, model, tokenizer, text):
        inputs = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = model.generate(inputs['input_ids'], max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def run_inference(self, text):
        responses = {}
        for name, model in self.models.items():
            tokenizer = self.tokenizers[name]
            responses[name] = self.generate_response(model, tokenizer, text)
        return responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Inference Script")
    parser.add_argument('--train_data_file', type=str, required=True, help="Path to the training data file")
    parser.add_argument('--asr_data_file', type=str, required=True, help="Path to the ASR data file")
    parser.add_argument('--config_file', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.config_file)

    # 加载并解析训练数据
    train_data = load_data(args.train_data_file, config)
    parsed_train_data = parse_train_data(train_data, config)

    # 初始化和训练对话模型
    dialogue_trainer = DialogueModelTrainer(model_name='gemma-2-9b', train_data=parsed_train_data)
    dialogue_trainer.train()
    dialogue_trainer.save_model('./finetuned_dialogue_model')

    # 示例日语对话文本数据集
    data = {
        "texts": [
            "これは情報検索タスクのための例文です。",
            "次の文を探しています。",
            "前の文と次の文の両方が必要です。",
            "これが私のリクエストです。",
            "情報検索は面白い分野です。"
        ],
        "labels": [0, 1, 1, 0, 0]
    }

    # 初始化和训练BERT分类模型
    bert_trainer = BERTFineTuner(model_name='cl-tohoku/bert-base-japanese', data=data)
    bert_trainer.train()

    # 初始化ASR推理模块
    model_names = ['dpo-model', 'rlhf-model', 'ragflow-model', 'gemma-2-9b']
    asr_inference = ASRInference(model_names=model_names)

    # 加载并解析ASR数据
    asr_data = load_data(args.asr_data_file, config)
    parsed_asr_data = parse_asr_data(asr_data, config)

    # 拆分ASR文本并进行推理
    results = []

    for entry in parsed_asr_data:
        text = entry['text']
        if text.strip():  # 确保非空
            responses = asr_inference.run_inference(text)
            results.append(responses)

    # 打印或处理推理结果
    for i, response_set in enumerate(results):
        print(f"Segment {i+1}:")
        for model_name, response in response_set.items():
            print(f"{model_name}: {response}")
        print()

    # Streaming Inference 示例
    streaming_inference = StreamingInference(model_name='gemma-2-9b')
    streaming_text = "请用中文回答我的问题：今天的天气怎么样？"
    for response in streaming_inference.stream_infer(streaming_text):
        print(response)

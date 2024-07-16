# main.py

import os
import re
import json
import csv
import torch
import pickle
import argparse
import random
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AdamW, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric

class ConfigLoader:
    @staticmethod
    def load_config(config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config

class FileManager:
    @staticmethod
    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return lines

    @staticmethod
    def export_to_txt(data, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            for content in data:
                file.write(json.dumps(content, ensure_ascii=False) + '\n')

    @staticmethod
    def save_as_json(data, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    @staticmethod
    def save_as_csv(data, output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['episode', 'time', 'speaker', 'text']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for episode in data:
                if 'dialogs' not in episode:
                    continue
                for dialog in episode['dialogs']:
                    writer.writerow({
                        'episode': episode['episode'],
                        'time': dialog['time'],
                        'speaker': dialog['speaker'],
                        'text': dialog['text']
                    })

class DataProcessor:
    def __init__(self, config):
        self.data = []
        self.dialog = []
        self.current_time = None
        self.current_episode = {'episode': 'Unknown', 'dialogs': []}
        self.current_speaker = None
        self.config = config

    @staticmethod
    def sort_files(filename):
        part = filename.split('.')[0]
        try:
            return int(part)
        except ValueError:
            return float('inf')

    def finalize_episode(self):
        if self.current_episode:
            if self.dialog:
                self.current_episode['dialogs'].append({
                    'speaker': self.current_speaker,
                    'time': self.current_time,
                    'text': ' '.join(self.dialog).strip()
                })
                self.dialog = []
            self.data.append(self.current_episode)
            print(f"Finalized episode: {self.current_episode}")
            self.current_episode = {'episode': 'Unknown', 'dialogs': []}

    def process_line(self, line):
        speaker_match = re.match(r'^話者(\d+)\s+(\d{2}:\d{2})\s+(.*)$', line)
        if speaker_match:
            if self.dialog:
                self.current_episode['dialogs'].append({
                    'speaker': self.current_speaker,
                    'time': self.current_time,
                    'text': ' '.join(self.dialog).strip()
                })
                self.dialog = []
            self.current_speaker, self.current_time, text = speaker_match.groups()
            self.dialog = [text]
        else:
            self.dialog.append(line)

    def process_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    self.process_line(line)
        self.finalize_episode()
        print(f"Processed file: {file_path} with data: {self.data[-1] if self.data else 'No Data'}")

    def process_all_files(self, directory_path):
        files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        files = sorted(files, key=self.sort_files)
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            self.process_file(file_path)

class DataLoader:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def parse_train_data(self, data):
        input_key = self.config['train_data']['input_key']
        output_key = self.config['train_data']['output_key']
        parsed_data = [{'input': entry[input_key], 'ideal_response': entry[output_key]} for entry in data]
        return parsed_data

    def parse_asr_data(self, data):
        text_key = self.config['asr_data']['text_key']
        parsed_data = [{'text': entry[text_key]} for entry in data]
        return parsed_data

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
        similarity = F.cosine_similarity(generated_response.unsqueeze(0), ideal_response.unsqueeze(0), dim=1)
        return similarity.item()

    def train_step(self, input_text, ideal_response):
        self.model.train()

        inputs = self.tokenizer(input_text, return_tensors='pt').input_ids.cuda()
        ideal_outputs = self.tokenizer(ideal_response, return_tensors='pt').input_ids.cuda()

        outputs = self.model.generate(inputs, max_length=50)
        generated_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated_response_embedding = self.model(input_ids=self.tokenizer(generated_response, return_tensors='pt').input_ids.cuda()).last_hidden_state.mean(dim=1)
        ideal_response_embedding = self.model(input_ids=ideal_outputs).last_hidden_state.mean(dim=1)
        reward = self.reward_function(generated_response_embedding, ideal_response_embedding)

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

class StreamingInference:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    def stream_infer(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').cuda()
        outputs = self.model.generate(inputs['input_ids'], max_length=50, stream=True)
        for output in outputs:
            yield self.tokenizer.decode(output, skip_special_tokens=True)

def save_model_and_tokenizer(model_name, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    config = model.config.to_dict()
    config_path = os.path.join(save_directory, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    vocab_path = os.path.join(save_directory, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=4)

    tokenizer.save_pretrained(save_directory)

    added_tokens_path = os.path.join(save_directory, 'added_tokens.json')
    if os.path.exists(added_tokens_path):
        with open(added_tokens_path, 'w') as f:
            json.dump(tokenizer.special_tokens_map, f, ensure_ascii=False, indent=4)

    print(f"Model and tokenizer files have been saved to {save_directory}")

def create_directory_structure():
    os.makedirs("models/fine_tuned/gpt2_fine_tuned", exist_ok=True)

    config = {
        "architectures": ["BertForMaskedLM"],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "model_type": "bert"
    }

    with open("models/fine_tuned/gpt2_fine_tuned/config.json", "w") as f:
        json.dump(config, f, indent=4)

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", ".", ",", "and", "of", "to"]
    with open("models/fine_tuned/gpt2_fine_tuned/vocab.txt", "w") as f:
        for token in vocab:
            f.write(token + "\n")

    merges = ["#version: 0.2", "e s", "t h", "th e"]
    with open("models/fine_tuned/gpt2_fine_tuned/merges.txt", "w") as f:
        for merge in merges:
            f.write(merge + "\n")

    training_args = {
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "save_steps": 10_000,
        "save_total_limit": 2,
        "evaluation_strategy": "epoch",
        "logging_dir": "./logs",
    }

    with open("models/fine_tuned/gpt2_fine_tuned/training_args.bin", "wb") as f:
        pickle.dump(training_args, f)

    with open("models/fine_tuned/gpt2_fine_tuned/pytorch_model.bin", "wb") as f:
        f.write(b"")

class ModelA:
    def __init__(self):
        print("ModelA initialized")

    def predict(self, input_data):
        return "ModelA prediction for {}".format(input_data)

class ModelB:
    def __init__(self):
        print("ModelB initialized")

    def predict(self, input_data):
        return "ModelB prediction for {}".format(input_data)

def train_model(args):
    data_loader = DataLoader(args.config_file)
    train_data = data_loader.load_data(args.train_data_file)
    parsed_train_data = data_loader.parse_train_data(train_data)

    dialogue_trainer = DialogueModelTrainer(model_name=args.model_name, train_data=parsed_train_data)
    dialogue_trainer.train()
    dialogue_trainer.save_model(args.output_dir)

def process_data(args):
    data_loader = DataLoader(args.config_file)
    asr_data = data_loader.load_data(args.asr_data_file)
    parsed_asr_data = data_loader.parse_asr_data(asr_data)
    print("数据处理完成。")

    bert_trainer = BERTFineTuner(model_name='bert-base-uncased', data={
        'texts': [entry['text'] for entry in parsed_asr_data],
        'labels': [0] * len(parsed_asr_data)
    })
    bert_trainer.train()
    bert_trainer.save_model('./bert_finetuned_model')

def main():
    parser = argparse.ArgumentParser(description="Model Training and Data Processing Script")

    parser.add_argument('--config_file', type=str, required=True, help="Path to the config file")
    parser.add_argument('--train_data_file', type=str, help="Path to the training data file")
    parser.add_argument('--model_name', type=str, help="Model name")
    parser.add_argument('--output_dir', type=str, help="Output directory")
    parser.add_argument('--asr_data_file', type=str, help="Path to the ASR data file")

    args = parser.parse_args()

    if args.model_name and args.train_data_file:
        train_model(args)
    elif args.asr_data_file:
        process_data(args)
    else:
        print("Please provide the necessary arguments for the chosen operation.")

if __name__ == "__main__":
    main()
    
    a = ModelA()
    print(a.predict("test data A"))

    b = ModelB()
    print(b.predict("test data B"))

    create_directory_structure()
    print("Directory structure and files created successfully.")

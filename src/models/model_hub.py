# model_hub.py

import logging
import torch
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed,
                          AutoModelForSeq2SeqLM, AutoTokenizer)

class ModelHub:
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.initialize_models()
        set_seed(42)  # 设置随机种子，以确保文本生成的可重复性
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def initialize_models(self):
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.config.get('tokenizer_model', 'bert-base-multilingual-cased'))
            self.classification_model = BertForSequenceClassification.from_pretrained(
                self.config.get('classification_model', 'bert-base-multilingual-cased'), 
                num_labels=self.config.get('num_labels', 2)
            ).to(self.device)
            
            self.generator = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self.generator_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
            # 加载其他需要的模型
            self.multi_model_manager = MultiModelManager()
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        outputs = self.classification_model(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()

    def generate_text(self, prompt, temperature=0.7):
        inputs = self.generator_tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.generator.generate(inputs, max_length=100, num_return_sequences=1, temperature=temperature)
        return self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_pipeline_text(self, prompt, temperature=0.7):
        generator = pipeline('text-generation', model=self.generator, tokenizer=self.generator_tokenizer)
        outputs = generator(prompt, max_length=100, num_return_sequences=1, temperature=temperature)
        return outputs[0]['generated_text']

class MultiModelManager:
    def __init__(self):
        # 加载所有需要的模型
        self.gpt_model, self.gpt_tokenizer = self.load_model('gpt2', GPT2LMHeadModel)
        self.bert_model, self.bert_tokenizer = self.load_model('bert-base-multilingual-cased', BertForSequenceClassification)

    def load_model(self, model_name, model_class=AutoModelForSeq2SeqLM):
        model = model_class.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def generate_text(self, text, model_type='gpt'):
        if model_type == 'gpt':
            inputs = self.gpt_tokenizer(text, return_tensors='pt')
            outputs = self.gpt_model.generate(inputs['input_ids'], max_length=100)
            return self.gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # 处理其他类型模型的生成
            pass

# Example usage
if __name__ == "__main__":
    config = {
        'tokenizer_model': 'bert-base-multilingual-cased',
        'classification_model': 'bert-base-multilingual-cased',
        'num_labels': 2
    }

    model_hub = ModelHub(config, device='cuda' if torch.cuda.is_available() else 'cpu')
    text_to_classify = "This is a test sentence."
    classification_result = model_hub.classify(text_to_classify)
    print(f"Classification result: {classification_result}")

    text_to_generate = "Once upon a time"
    generated_text = model_hub.generate_text(text_to_generate)
    print(f"Generated text: {generated_text}")

    multi_model_manager = MultiModelManager()
    response = multi_model_manager.generate_text("Hello, how are you?", model_type='gpt')
    print(response)

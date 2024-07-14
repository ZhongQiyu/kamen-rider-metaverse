# bias_handler.py

import torch
import re
from transformers import BertJapaneseTokenizer, BertForMaskedLM
from janome.tokenizer import Tokenizer as JanomeTokenizer

class BiasHandler:
    def __init__(self):
        self.tokenizer = JanomeTokenizer()
        self.bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        self.bert_model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese')
        self.bias_words = {'悪い': '良くない', 'ダメ': '良くない', '嫌い': '好きではない'}

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号和特殊字符
        tokens = self.tokenizer.tokenize(text, wakati=True)  # 分词
        return tokens

    def detect_bias(self, tokens):
        # 检测文本中是否存在偏见词
        return [word for word in tokens if word in self.bias_words]

    def correct_bias(self, tokens):
        # 纠正偏见词
        return [self.bias_words.get(word, word) for word in tokens]

    def contextual_correction(self, text):
        # 将偏见词替换为[MASK]
        masked_text = re.sub('|'.join(map(re.escape, self.bias_words.keys())), '[MASK]', text)
        input_ids = self.bert_tokenizer.encode(masked_text, return_tensors='pt')

        # 使用BERT模型预测掩码位置的单词
        with torch.no_grad():
            outputs = self.bert_model(input_ids)
        logits = outputs.logits
        mask_token_index = torch.where(input_ids == self.bert_tokenizer.mask_token_id)[1]
        top_k_words = torch.topk(logits[0, mask_token_index, :], 1, dim=1).indices[0].tolist()

        # 将预测的单词替换回文本
        for word_id in top_k_words:
            predicted_word = self.bert_tokenizer.decode([word_id])
            text = text.replace('[MASK]', predicted_word, 1)
        return text

    def process_text(self, text):
        # 处理文本
        tokens = self.preprocess_text(text)
        if self.detect_bias(tokens):
            tokens = self.correct_bias(tokens)
            corrected_text = ''.join(tokens)
            return self.contextual_correction(corrected_text)
        return text

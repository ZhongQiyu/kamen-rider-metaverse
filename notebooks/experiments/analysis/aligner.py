# aligner.py

import re
import json
import torch
from transformers import BertJapaneseTokenizer, BertForTokenClassification, RagTokenizer, RagRetriever, RagSequenceForGeneration
from sentence_transformers import SentenceTransformer
from typing import List

class JapaneseGrammarAligner:
    def __init__(self):
        self.bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        self.bert_model = BertForTokenClassification.from_pretrained('cl-tohoku/bert-base-japanese')

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    def bert_tokenize(self, text):
        inputs = self.bert_tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.bert_model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)
        tokens = self.bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return tokens, predictions[0].tolist()

    def align_grammar(self, text):
        processed_text = self.preprocess_text(text)
        bert_tokens, bert_predictions = self.bert_tokenize(processed_text)
        
        alignment = {
            "original_text": text,
            "processed_text": processed_text,
            "bert_tokens": bert_tokens,
            "bert_predictions": bert_predictions
        }
        return alignment

class RAGDialogueGenerator:
    def __init__(self, retriever_model_name="facebook/dpr-ctx_encoder-multiset-base", rag_model_name="facebook/rag-sequence-nq"):
        self.retriever = SentenceTransformer(retriever_model_name)
        self.rag_tokenizer = RagTokenizer.from_pretrained(rag_model_name)
        self.rag_retriever = RagRetriever.from_pretrained(rag_model_name, index_name="exact", use_dummy_dataset=True)
        self.rag_model = RagSequenceForGeneration.from_pretrained(rag_model_name)

    def generate_response(self, question, context_documents):
        inputs = self.rag_tokenizer(question, return_tensors="pt")
        question_embeddings = self.retriever.encode([question], convert_to_tensor=True)
        
        # Perform retrieval using the context documents
        docs = self.rag_retriever(question_inputs=inputs['input_ids'], prefix_allowed_tokens_fn=None)
        
        # Generate response using RAG
        outputs = self.rag_model.generate(input_ids=inputs['input_ids'], context_input_ids=docs['context_input_ids'])
        response = self.rag_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response

    def retrieve_and_generate(self, query, context):
        context_embeddings = self.retriever.encode(context, convert_to_tensor=True)
        response = self.generate_response(query, context_embeddings)
        return response

if __name__ == "__main__":
    # Example conversation from "Kamen Rider Blade"
    conversation = [
        "剣崎、一緒に戦おう！",
        "俺の運命は俺が決める！",
        "ああ、分かった。共に行こう！"
    ]

    context_documents = [
        "剣崎は決意を新たにした。",
        "彼は運命を自ら切り開くと誓った。",
        "仲間たちとの絆が深まった。"
    ]

    aligner = JapaneseGrammarAligner()
    generator = RAGDialogueGenerator()

    aligned_conversation = []
    for line in conversation:
        alignment = aligner.align_grammar(line)
        response = generator.retrieve_and_generate(line, context_documents)
        alignment["response"] = response
        aligned_conversation.append(alignment)

    # Save the results to a JSON file
    with open('rag_aligned_conversation.json', 'w', encoding='utf-8') as f:
        json.dump(aligned_conversation, f, ensure_ascii=False, indent=4)

    # Print the aligned conversation
    for alignment in aligned_conversation:
        print(json.dumps(alignment, ensure_ascii=False, indent=4))

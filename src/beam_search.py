# beam_search.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer

class BeamSearch:
    def __init__(self, model, tokenizer, beam_size=3):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size

    def generate(self, input_text, max_length=50, temperature=1.0, top_k=50, repetition_penalty=2.0):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        beam_outputs = self.model.generate(
            input_ids, 
            max_length=max_length, 
            num_beams=self.beam_size, 
            early_stopping=True,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in beam_outputs]

def generate_and_save(input_text, output_file, beam_size=5, max_length=50, temperature=1.0, top_k=50, repetition_penalty=2.0):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    beam_search = BeamSearch(model, tokenizer, beam_size)

    generated_texts = beam_search.generate(input_text, max_length, temperature, top_k, repetition_penalty)

    with open(output_file, 'w', encoding='utf-8') as f:
        for text in generated_texts:
            f.write(text + '\n')

if __name__ == "__main__":
    input_text = "The quick brown fox"
    output_file = "generated_texts.txt"

    generate_and_save(input_text, output_file)

    print(f"Generated texts saved to {output_file}")

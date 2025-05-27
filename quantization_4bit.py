import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

class PLaMoTranslate4bit:
    def __init__(self):
        self.model_name = "pfnet/plamo-2-translate"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        print("Loading model with 4-bit quantization...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config
        )
        print("Model loaded successfully!")
    
    def create_translation_prompt(self, text: str) -> str:
        return f'''<|plamo:op|>dataset
translation
<|plamo:op|>input lang=Japanese
{text}
<|plamo:op|>output lang=English
'''
    
    def translate(self, japanese_text: str) -> str:
        prompt = self.create_translation_prompt(japanese_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                stop_strings=["<|plamo:op|>"],
                tokenizer=self.tokenizer
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        if "<|plamo:op|>" in generated_text:
            generated_text = generated_text.split("<|plamo:op|>")[0]
        
        return generated_text.strip()

def main():
    translator = PLaMoTranslate4bit()
    
    print("\nPLaMo 2-translate (4-bit quantized)")
    print("Enter Japanese text to translate to English (Ctrl+C to exit):")
    
    try:
        while True:
            japanese_input = input("\n日本語: ")
            if japanese_input.strip():
                english_output = translator.translate(japanese_input)
                print(f"English: {english_output}")
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
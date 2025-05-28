import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import warnings
import uvicorn
from typing import Optional, Literal
import logging
import os

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PLaMo Translation API", version="1.0.0")

@app.get("/")
async def serve_html():
    html_file = "index.html"
    if os.path.exists(html_file):
        return FileResponse(html_file, media_type="text/html")
    else:
        return {
            "message": "PLaMo Translation API",
            "model": translator.model_name,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_loaded": False,
            "quantization": None,
            "note": "Place index.html in the same directory to serve the web UI"
        }

class InitRequest(BaseModel):
    quantization: Literal["4bit", "8bit", "none"] = "4bit"

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "Japanese"
    target_lang: str = "English"

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str

class PLaMoTranslator:
    def __init__(self):
        self.model_name = "pfnet/plamo-2-translate"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.quantization_type = None
        
    def initialize_model(self, quantization: str = "4bit"):
        if self.model is not None:
            logger.info("Model already loaded. Skipping initialization.")
            return
            
        self.quantization_type = quantization
        logger.info(f"Loading model with {quantization} quantization...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        quantization_config = None
        torch_dtype = torch.bfloat16
        
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
            quantization_config=quantization_config
        )
        
        logger.info(f"Model loaded successfully with {quantization} quantization!")
    
    def create_translation_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        return f'''<|plamo:op|>dataset
translation
<|plamo:op|>input lang={source_lang}
{text}
<|plamo:op|>output lang={target_lang}
'''
    
    def translate(self, text: str, source_lang: str = "Japanese", target_lang: str = "English") -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Please call initialize_model first.")
        
        prompt = self.create_translation_prompt(text, source_lang, target_lang)
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
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        if "<|plamo:op|>" in generated_text:
            generated_text = generated_text.split("<|plamo:op|>")[0]
        
        return generated_text.strip()

translator = PLaMoTranslator()

@app.get("/api")
async def root():
    return {
        "message": "PLaMo Translation API",
        "model": translator.model_name,
        "device": translator.device,
        "model_loaded": translator.model is not None,
        "quantization": translator.quantization_type
    }

@app.post("/api/initialize")
async def initialize_model(request: InitRequest):
    try:
        translator.initialize_model(request.quantization)
        return {
            "message": f"Model initialized successfully with {request.quantization} quantization",
            "quantization": request.quantization,
            "device": translator.device
        }
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

@app.post("/api/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    if translator.model is None:
        raise HTTPException(
            status_code=400, 
            detail="Model not initialized. Please call /initialize first."
        )
    
    try:
        translated_text = translator.translate(
            request.text, 
            request.source_lang, 
            request.target_lang
        )
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": translator.model is not None,
        "device": translator.device,
        "quantization": translator.quantization_type
    }

@app.delete("/api/model")
async def unload_model():
    try:
        if translator.model is not None:
            del translator.model
            del translator.tokenizer
            translator.model = None
            translator.tokenizer = None
            translator.quantization_type = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {"message": "Model unloaded successfully"}
        else:
            return {"message": "No model was loaded"}
    except Exception as e:
        logger.error(f"Failed to unload model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting PLaMo Translation API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
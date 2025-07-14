import requests
import json
import logging
import torch
import platform
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, MODEL_TEMPERATURE, MODEL_MAX_TOKENS

logger = logging.getLogger(__name__)

BASE_MODELS = {
    "codellama-7b-libsmart": "codellama/CodeLlama-7b-hf",
    "gemma-2b-libsmart": "google/gemma-2b", 
    "mistral-7b-libsmart": "mistralai/Mistral-7B-Instruct-v0.2"
}

class ModelService:

    def __init__(self, model_name: str = None):

        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.use_ollama = model_name is None
        
        if not self.use_ollama:
            self._load_finetuned_model()
        else:
            self.base_url = OLLAMA_BASE_URL
            self.api_url = f"{self.base_url}/api/generate"

    def _load_finetuned_model(self):

        try:

            base_model_id = BASE_MODELS.get(self.model_name)

            if not base_model_id:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            adapter_path = Path(__file__).parent.parent / "fine-tuning" / "models" / self.model_name
            
            logger.info(f"Loading base model: {base_model_id}")
            
            if platform.system() == "Darwin":
                logger.info("Using PyTorch dynamic quantization for macOS")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype = torch.float32,
                    low_cpu_mem_usage = True
                )
                
                logger.info(f"Loading adapter from: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
                
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype = torch.qint8
                )

                logger.info("Applied int8 dynamic quantization")
                
            else:

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit = True,
                    bnb_4bit_quant_type = "nf4",
                    bnb_4bit_compute_dtype = torch.float16,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    quantization_config = bnb_config,
                    torch_dtype = torch.float16,
                    device_map = "auto"
                )
                
                logger.info(f"Loading adapter from: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _generate_finetuned(self, prompt: str) -> str:
        
        instruction = """Modernize deprecated NumPy code. Output in two parts separated by "---EXPLANATION---":
1. First part: Only the modernized code
2. Second part: Explanation of what changes were made and why. Keep this very concise, don't be too verbose. No more than 1-2 sentences per change.
If no deprecated functionality is found, output only 'No deprecated functionality found'"""
        
        if "mistral" in self.model_name:
            full_prompt = f"<s>[INST] {instruction}\n\nCode:\n{prompt} [/INST]"
        elif "gemma" in self.model_name:
            full_prompt = f"<start_of_turn>user\n{instruction}\n\nCode:\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            full_prompt = f"{instruction}\n\nCode:\n{prompt}\n\nOutput:\n"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
        
        if torch.backends.mps.is_available():
            inputs = inputs.to("mps")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = 256,
                temperature = MODEL_TEMPERATURE,
                do_sample = True,
                eos_token_id = self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens = True)
        return response

    def create_prompt(self, code: str, version: str, funcs: List[str], ctx: Dict[str, List[Dict[str, Any]]]) -> str:

        if not self.use_ollama:
            return code
        
        context_parts = []

        for fn, chunks in ctx.items():
            if chunks and chunks[0]['similarity_score'] > 0.4:
                content = chunks[0]['content']
                if 'deprecated' in content.lower() or 'replacement' in content.lower():
                    context_parts.append(f"- {fn}: {content[:200]}...")

        context = "\n".join(context_parts) if context_parts else ""

        prompt = f"""Modernize this NumPy {version} code by replacing deprecated functions, parameters, or keywords.

{f"Context from documentation:\n{context}\n" if context else ""}Code:
```python
{code}
```

Instructions:
- Check if any NumPy functionality is deprecated (functions, parameters, keywords)
- Ignore context that doesn't indicate deprecation
- Output in two parts separated by "---EXPLANATION---"
- First part: Only the modernized code in a ```python block
- Second part: Explanation of what changes were made and why. Keep this very concise, don't be too verbose. No more than 1-2 sentences per change.
- If no deprecated functionality found, output only "No deprecated functionality found"

Output:"""
        
        return prompt

    def call_model(self, code: str, version: str, funcs: List[str], ctx: Dict[str, List[Dict[str, Any]]]) -> str:

        logger.info(f"Calling model for {len(funcs)} functions")
        
        try:

            if self.use_ollama:

                prompt = self.create_prompt(code, version, funcs, ctx)
                
                resp = requests.post(
                    self.api_url,
                    json = {
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": MODEL_TEMPERATURE,
                            "num_predict": MODEL_MAX_TOKENS
                        }
                    },
                    timeout = 30
                )
                
                if resp.status_code == 200:
                    result = resp.json()
                    return result.get("response", "")
                else:
                    logger.error(f"Ollama call failed: {resp.status_code}")
                    return "Model call failed"
                
            else:

                return self._generate_finetuned(code)
                
        except Exception as e:

            logger.exception("Model call exception")
            raise

    def is_available(self) -> bool:

        if self.use_ollama:

            try:
                resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
                return resp.status_code == 200
            except:
                return False
            
        else:
            return self.model is not None and self.tokenizer is not None
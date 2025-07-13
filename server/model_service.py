import requests
import json
import re
import logging
import torch
import platform
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import  BitsAndBytesConfig
from peft import PeftModel

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, MODEL_TEMPERATURE, MODEL_MAX_TOKENS

logger = logging.getLogger(__name__)

# Base model mapping
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

    # DOESN'T WORK AT THE MOMENT, QUANTIZATION WITH PICKLE FILE SOMEHOW NOT SUPPORTED!!!
    
    # Load fine-tuned model
    def _load_finetuned_model(self):

        try:
            base_model_id = BASE_MODELS.get(self.model_name)
            if not base_model_id:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            adapter_path = Path(__file__).parent.parent / "fine-tuning" / "models" / self.model_name
            
            logger.info(f"Loading base model: {base_model_id}")
            
            # macOS: Use PyTorch quantization
            if platform.system() == "Darwin":
                logger.info("Using PyTorch dynamic quantization for macOS")
                
                # Load in float32 first
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype = torch.float32,
                    low_cpu_mem_usage = True
                )
                
                # Apply PEFT adapter before quantization
                logger.info(f"Loading adapter from: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
                
                # Apply dynamic quantization
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype = torch.qint8
                )

                logger.info("Applied int8 dynamic quantization")
                
            else:
                # Linux/Windows: Use bitsandbytes
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

    # Generate response from fine-tuned model
    def _generate_finetuned(self, prompt: str) -> str:
        instruction = "Analyze the following Python code for a deprecated NumPy function and generate a JSON object with the suggested fix and context."
        
        if "mistral" in self.model_name:
            full_prompt = f"<s>[INST] {instruction}\n\n### INPUT CODE:\n```python\n{prompt}\n``` [/INST]"
        elif "gemma" in self.model_name:
            full_prompt = f"<start_of_turn>user\n{instruction}\n\n### INPUT CODE:\n```python\n{prompt}\n```<end_of_turn>\n<start_of_turn>model\n"
        else:
            full_prompt = f"{instruction}\n\n### INPUT CODE:\n```python\n{prompt}\n```\n\n### OUTPUT:\n"
        
        inputs = self.tokenizer(full_prompt, return_tensors = "pt", add_special_tokens = False)
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens = 256,
                temperature = MODEL_TEMPERATURE,
                do_sample = True,
                eos_token_id = self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response

    def create_prompt(self, code: str, version: str, funcs: List[str], ctx: Dict[str, List[Dict[str, Any]]]) -> str:
        
        if not self.use_ollama:
            return code
        
        context_parts = []

        for fn, chunks in ctx.items():
            if chunks:
                for chunk in chunks[:3]:
                    if chunk['similarity_score'] > 0.4:
                        content = chunk['content'].replace('\n', ' ')[:600]
                        context_parts.append(f"- {content}")

        context_section = "\n".join(context_parts) if context_parts else "No deprecation information found."

        prompt = f"""You are a NumPy expert. Modernize NumPy {version} code by replacing deprecated functions.

Deprecation Information:
{context_section}

Code to modernize:
```python
{code}
```

For each deprecated function call, provide the exact code snippet that needs to be replaced and its modernized version.

Respond in JSON format:
{{
  "code": "full modernized code here",
  "changes": [
    {{"input": "exact_deprecated_code_snippet", "modernized_code": "exact_replacement_snippet", "reason": "explanation"}}
  ]
}}

JSON Response:"""
        
        return prompt

    def parse_response(self, resp_text: str, orig_code: str) -> Dict[str, Any]:

        try:
            json_match = re.search(r'\{[\s\S]*\}', resp_text)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "modernized_code": data.get("code", orig_code),
                    "changes": data.get("changes", []),
                    "success": bool(data.get("code") and data.get("code") != orig_code)
                }
        except json.JSONDecodeError:
            logger.warning("JSON parse failed, using fallback")
        
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', resp_text, re.DOTALL)
        code = code_blocks[0].strip() if code_blocks else orig_code
        
        changes = []
        deprecated_patterns = [
            r'(np\.\w+_?\([^)]*\))',
            r'(numpy\.\w+_?\([^)]*\))',
            r'(\w+\s*=\s*np\.\w+)',
        ]
        
        for pattern in deprecated_patterns:
            for match in re.finditer(pattern, orig_code):
                snippet = match.group(1)
                if 'deprecated' in resp_text.lower() and snippet in resp_text:
                    changes.append({
                        "input": snippet,
                        "modernized_code": snippet,
                        "reason": "Deprecated function detected"
                    })
        
        return {
            "modernized_code": code,
            "changes": changes,
            "success": code != orig_code
        }

    # Main call method
    def call_model(self, code: str, version: str, funcs: List[str], ctx: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:

        logger.info(f"Calling model for {len(funcs)} functions")
        
        try:
            if self.use_ollama:
                prompt = self.create_prompt(code, version, funcs, ctx)
                
                resp = requests.post(
                    self.api_url,
                    json={
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
                    response_text = result.get("response", "")
                    return self.parse_response(response_text, code)
                else:
                    logger.error(f"Ollama call failed: {resp.status_code}")
                    return {
                        "modernized_code": code,
                        "changes": [],
                        "success": False
                    }
                
            else:

                response_text = self._generate_finetuned(code)
                return self.parse_response(response_text, code)
                
        except Exception as e:
            logger.exception("Model call exception")
            return {
                "modernized_code": code,
                "changes": [],
                "success": False
            }

    # Check availability
    def is_available(self) -> bool:
        
        if self.use_ollama:
            try:
                resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
                return resp.status_code == 200
            except:
                return False
        else:
            return self.model is not None and self.tokenizer is not None
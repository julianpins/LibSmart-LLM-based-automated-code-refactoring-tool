import logging
from pathlib import Path
from typing import Dict, List, Any
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class ModelService:

    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.model = None
        
        if model_name:
            self._load_gguf_model()
        else:
            raise ValueError("Model name is required")

    def _load_gguf_model(self):
        try:
            models_base_dir = Path(__file__).parent.parent / "fine-tuning" / "models"
            direct_gguf_path = models_base_dir / f"{self.model_name}.gguf"
            
            if not direct_gguf_path.exists():
                raise ValueError(f"GGUF file not found: {direct_gguf_path}")
            
            logger.info(f"Loading GGUF model: {direct_gguf_path}")
            
            self.model = Llama(
                model_path=str(direct_gguf_path),
                n_ctx=1024,
                n_gpu_layers=-1,
                verbose=False,
                n_threads=8
            )
            
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise
    

    def _create_system_prompt(self, context: Dict[str, List[Dict[str, Any]]] | None = None) -> str:
        base_prompt = (
            "You are a Python code refactoring tool for NumPy. Your task is to replace only the deprecated functions in the given code snippet with their modern equivalents.\n"
            "Your response must be structured with two markdown sections:\n"
            "1. A '### Refactored Code' section containing ONLY the updated Python code block. Do not change the code's logic. Do not add imports. Do not add comments.\n"
            "2. A '### Deprecation Context' section containing a brief explanation of the deprecation.\n"
            " If no functions are deprecated, return the original code and state that no changes were needed in the context section."
        )
        
        if context:
            context_parts = []
            for fn, chunks in context.items():
                if chunks and chunks[0]['similarity_score'] > 0.5:
                    content = chunks[0]['content']
                    if 'deprecated' in content.lower() or 'replacement' in content.lower():
                        context_parts.append(f"- {fn}: {content[:150]}...")
            
            if context_parts:
                base_prompt += f"\n\nRelevant context (use only if applicable):\n{chr(10).join(context_parts)}"
        
        return base_prompt

    def _generate_gguf(self, code: str, context: Dict[str, List[Dict[str, Any]]] | None = None) -> str:
        system_prompt = self._create_system_prompt(context)
        
        if "gemma-2-2b-it" in self.model_name:
            user_content = f"{system_prompt}\n\n### INPUT CODE:\n```python\n{code}\n```"
            full_prompt = f"<bos><start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
            stop_tokens = ["<end_of_turn>", "<start_of_turn>"]
        elif "tinyllama-1.1b-chat" in self.model_name:
            user_message = f"### INPUT CODE:\n```python\n{code}\n```"
            full_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_message}</s>\n<|assistant|>\n"
            stop_tokens = ["</s>", "<|user|>"]
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        try:
            output = self.model(
                full_prompt,
                max_tokens=256,
                temperature=0.0,
                stop=stop_tokens,
                echo=False
            )
            
            if output and 'choices' in output and output['choices']:
                result = output['choices'][0]['text'].strip()
            else:
                result = "Model returned empty response"
            logger.info(f"Model generated {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            return f"Model generation error: {str(e)}"


    def call_model(self, code: str, version: str, funcs: List[str], ctx: Dict[str, List[Dict[str, Any]]] | None = None) -> str:
        logger.info(f"Calling model for {len(funcs)} functions")
        try:
            return self._generate_gguf(code, ctx)
        except Exception as e:
            logger.exception("Model call exception")
            raise

    def is_available(self) -> bool:
        return self.model is not None
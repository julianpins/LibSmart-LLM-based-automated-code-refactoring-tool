import requests
import json
import re
import logging
from typing import Dict, List, Any

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, MODEL_TEMPERATURE, MODEL_MAX_TOKENS

logger = logging.getLogger(__name__)


class ModelService:

    def __init__(self):
        self.model = OLLAMA_MODEL
        self.base_url = OLLAMA_BASE_URL
        self.api_url = f"{self.base_url}/api/generate"

    def create_prompt(self, code: str, version: str, funcs: List[str], ctx: Dict[str, List[Dict[str, Any]]]) -> str:
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
            r'(np\.\w+_?\([^)]*\))',  # np.function() calls
            r'(numpy\.\w+_?\([^)]*\))',  # numpy.function() calls
            r'(\w+\s*=\s*np\.\w+)',  # assignments like dtype = np.int, probably need even more
        ]
        
        for pattern in deprecated_patterns:
            for match in re.finditer(pattern, orig_code):
                snippet = match.group(1)
                if 'deprecated' in resp_text.lower() and snippet in resp_text:
                    changes.append({
                        "input": snippet,
                        "modernized_code": snippet,  # Fallback: same as input
                        "reason": "Deprecated function detected"
                    })
        
        return {
            "modernized_code": code,
            "changes": changes,
            "success": code != orig_code
        }

    # Call model with prompt
    def call_model(self, code: str, version: str, funcs: List[str], ctx: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        prompt = self.create_prompt(code, version, funcs, ctx)
        
        logger.info(f"Calling model for {len(funcs)} functions")
        logger.debug(f"Prompt length: {len(prompt)} chars")
        
        try:
            resp = requests.post(
                self.api_url,
                json={
                    "model": self.model,
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
                logger.debug(f"Model response length: {len(response_text)} chars")
                return self.parse_response(response_text, code)
            else:
                logger.error(f"Model call failed: {resp.status_code}")
                return {
                    "modernized_code": code,
                    "changes": [],
                    "success": False
                }
                
        except Exception as e:
            logger.exception("Model call exception")
            return {
                "modernized_code": code,
                "changes": [],
                "success": False
            }

    # Check Ollama availability
    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except:
            return False
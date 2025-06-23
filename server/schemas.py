from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CodeAnalysisRequest(BaseModel):
    code: str = Field(..., description="Python code to analyze")
    numpy_version: str = Field(..., description="NumPy version (e.g., '1.24.0')")


class FunctionInfo(BaseModel):
    name: str
    line: int
    call: str


class CodeAnalysisResponse(BaseModel):
    modernized_code: str
    changes: List[Dict[str, str]] = Field(default_factory=list)
    retrieved_context: Dict[str, List[str]] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    chroma_connected: bool
    ollama_available: bool
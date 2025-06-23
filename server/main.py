from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import API_TITLE, API_VERSION, API_HOST, API_PORT
from schemas import CodeAnalysisRequest, CodeAnalysisResponse, HealthResponse
from rag_service import RAGService

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title = API_TITLE, version = API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGService()

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status = "healthy",
        chroma_connected = rag.is_connected(),
        ollama_available = rag.is_model_available()
    )


# Main endpoint
@app.post("/analyze", response_model = CodeAnalysisResponse)
async def analyze(req: CodeAnalysisRequest) -> CodeAnalysisResponse:
    logger.info(f"Analyzing code: {len(req.code)} chars, NumPy {req.numpy_version}")
    
    if not rag.is_connected():
        raise HTTPException(status_code = 503, detail = "Vector database unavailable")
    
    try:
        result = rag.analyze_code(req.code, req.numpy_version)
        
        if result.error:
            logger.error(f"Analysis error: {result.error}")
        else:
            logger.info(f"Analysis successful: {len(result.changes)} changes, {len(result.retrieved_context)} functions analyzed")
        
        return result
        
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code = 500, detail = str(e))


# Startup event handler, deprecated though – will fix this later
@app.on_event("startup")
async def startup() -> None:
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info(f"ChromaDB: {'connected' if rag.is_connected() else 'disconnected'}")
    logger.info(f"Ollama: {'available' if rag.is_model_available() else 'unavailable'}")


if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
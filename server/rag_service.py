import ast
import re
import logging
from typing import List, Dict, Set, Any
import chromadb
import torch
from chromadb.config import Settings

from config import CHROMA_DB_DIR, COLLECTION_NAME, NUMPY_ALIASES
from schemas import FunctionInfo, CodeAnalysisResponse
from model_service import ModelService


logger = logging.getLogger(__name__)


class NumpyFunctionExtractor(ast.NodeVisitor):

    def __init__(self, aliases: Set[str]):
        self.aliases = aliases
        self.funcs: List[FunctionInfo] = []
    
    # Extract NumPy function calls from AST
    def visit_call(self, node: ast.Call) -> None:
        name = self._get_name(node.func)
        if name and self._is_numpy(name):
            self.funcs.append(FunctionInfo(
                name = name.split('.')[-1],
                line = node.lineno,
                call = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
            ))
        self.generic_visit(node)
    
    def _get_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parts = []
            current: Any = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return ""
    
    def _is_numpy(self, name: str) -> bool:
        parts = name.split('.')
        return len(parts) > 1 and parts[0] in self.aliases

class RAGService:
    def __init__(self):
        self.collection: Any = None
        self._init_chroma()
        self.model = ModelService()
        
    # Initialize ChromaDB connection
    def _init_chroma(self) -> None:
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            
            embed_fn = SentenceTransformerEmbeddingFunction(
                model_name = "BAAI/bge-base-en-v1.5",
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            )
            
            client = chromadb.PersistentClient(
                path = CHROMA_DB_DIR,
                settings = Settings(anonymized_telemetry=False)
            )
            
            try:
                self.collection = client.get_collection(COLLECTION_NAME, embedding_function = embed_fn)
                logger.info(f"Connected to collection: {COLLECTION_NAME}")
            except:
                self.collection = client.create_collection(
                    name=COLLECTION_NAME,
                    embedding_function = embed_fn,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created collection: {COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
    
    # Extract NumPy functions from code
    def extract_funcs(self, code: str) -> List[FunctionInfo]:
        try:
            aliases = set(NUMPY_ALIASES)
            
            for match in re.findall(r'import\s+numpy\s+as\s+(\w+)', code):
                aliases.add(match)
            if 'import numpy' in code:
                aliases.add('numpy')
            
            tree = ast.parse(code)
            extractor = NumpyFunctionExtractor(aliases)
            extractor.visit(tree)
            
            return extractor.funcs
        except Exception as e:
            logger.error(f"Function extraction failed: {e}")
            return []
    
    # Query vector DB with multiple variations
    def query_db(self, func: str, version: str) -> List[Dict[str, Any]]:
        if not self.collection:
            return []
        
        try:
            variations = [func, f"numpy.{func}", f"np.{func}"]
            all_chunks: List[Dict[str, Any]] = []
            
            for variant in variations:
                query = f"{variant} numpy {version} deprecated"
                
                res = self.collection.query(
                    query_texts=[query],
                    n_results=3,
                    include=["documents", "metadatas", "distances"]
                )
                
                if res['documents'] and res['documents'][0]:
                    for i, doc in enumerate(res['documents'][0]):
                        chunk_data = {
                            'content': doc,
                            'metadata': res['metadatas'][0][i] if res['metadatas'] else {},
                            'similarity_score': 1 - res['distances'][0][i]
                        }
                        all_chunks.append(chunk_data)
            
            # Sort by similarity and return top 3
            all_chunks.sort(key = lambda x: x['similarity_score'], reverse=True)
            return all_chunks[:3]
            
        except Exception as e:
            logger.error(f"DB query failed for {func}: {e}")
            return []
    
    # Main analysis function
    def analyze_code(self, code: str, version: str) -> CodeAnalysisResponse:
        logger.info(f"Analyzing code with NumPy {version}")
        
        funcs = self.extract_funcs(code)
        unique_funcs = list({f.name for f in funcs})
        
        logger.info(f"Found {len(unique_funcs)} unique NumPy functions")
        
        ctx: Dict[str, List[Dict[str, Any]]] = {}
        for fn in unique_funcs:
            ctx[fn] = self.query_db(fn, version)
        
        if self.model.is_available():
            result = self.model.call_model(code, version, unique_funcs, ctx)
            
            retrieved_chunks: Dict[str, List[str]] = {}
            for fn, chunks in ctx.items():
                if chunks and len(chunks) > 0:
                    # Simply take the full content of the best chunk
                    best_chunk = chunks[0]
                    content = best_chunk.get('content', '')
                    if content:
                        retrieved_chunks[fn] = [content]
                    else:
                        retrieved_chunks[fn] = ["No content found"]
                else:
                    retrieved_chunks[fn] = ["No matching documentation found"]
            
            return CodeAnalysisResponse(
                modernized_code = result["modernized_code"],
                changes = result["changes"],
                retrieved_context = retrieved_chunks,
                success = result["success"]
            )
        else:
            logger.warning("Model not available")
            return CodeAnalysisResponse(
                modernized_code = code,
                changes = [],
                retrieved_context = {},
                success = False,
                error = "Model service unavailable"
            )
    
    def is_connected(self) -> bool:
        return self.collection is not None
    
    def is_model_available(self) -> bool:
        return self.model.is_available()
import ast
import re
import logging
from typing import List, Dict, Set, Any, Tuple
import chromadb
import torch
from chromadb.config import Settings

from config import CHROMA_DB_DIR, COLLECTION_NAME, NUMPY_ALIASES
from schemas import FunctionInfo, CodeAnalysisResponse
from model_service import ModelService

logger = logging.getLogger(__name__)

class NumpyFunctionExtractor(ast.NodeVisitor):

    def __init__(self):

        self.funcs: List[FunctionInfo] = []
        self.imports: Dict[str, str] = {}
        self.star_imports: Set[str] = set()
        
    def visit_Import(self, node):

        for alias in node.names:
            if 'numpy' in alias.name or alias.name == 'numpy_financial':
                self.imports[alias.asname or alias.name] = alias.name
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):

        if node.module and ('numpy' in node.module or node.module == 'numpy_financial'):

            if any(alias.name == '*' for alias in node.names):
                self.star_imports.add(node.module)

            else:

                for alias in node.names:
                    name = alias.asname or alias.name
                    self.imports[name] = f"{node.module}.{alias.name}"

        self.generic_visit(node)
    
    def visit_Call(self, node):

        func_str = self._extract_call(node)

        if func_str:
            self.funcs.append(FunctionInfo(
                name=func_str,
                line=node.lineno,
                call=ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
            ))

        self.generic_visit(node)
    
    def visit_Attribute(self, node):

        if isinstance(node.value, ast.Name) and node.value.id in ['np', 'numpy', 'npf', 'numpy_financial']:
            return
        
        chain = self._get_chain(node)

        if chain and any(part in ['mean', 'std', 'sum', 'min', 'max', 'var', 'cumprod', 'cumsum', 
                                  'argsort', 'sort', 'tostring', 'tofile', 'astype', 'reshape',
                                  'flatten', 'ravel', 'transpose', 'swapaxes', 'squeeze'] for part in chain):
            self.funcs.append(FunctionInfo(
                name = chain[-1],
                line = node.lineno,
                call = '.'.join(chain)
            ))
        self.generic_visit(node)
    
    def _extract_call(self, node):

        if isinstance(node.func, ast.Name):

            name = node.func.id
            if name in self.imports:
                return self.imports[name]
            if self.star_imports:
                return name
            
        elif isinstance(node.func, ast.Attribute):

            chain = self._get_chain(node.func)

            if chain:
                root = chain[0]
                if root in self.imports:
                    base = self.imports[root]
                    full_name = f"{base}.{'.'.join(chain[1:])}"
                    return full_name.replace('numpy.', 'np.')
                elif root in ['np', 'numpy', 'npf', 'numpy_financial'] + list(NUMPY_ALIASES):
                    full_name = '.'.join(chain)
                    return full_name.replace('numpy.', 'np.')
                elif self.star_imports and len(chain) > 1:
                    return '.'.join(chain)
                
        return None
    
    def _get_chain(self, node):

        parts = []

        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            parts.append(node.id)

        return list(reversed(parts))

class RAGService:

    def __init__(self, model_name: str = None):

        self.collection: Any = None
        self._init_chroma()
        self.model = ModelService(model_name)
        
    def _init_chroma(self) -> None:

        try:

            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            
            embed_fn = SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-base-en-v1.5",
                device="mps" if torch.backends.mps.is_available() else "cpu"
            )
            
            client = chromadb.PersistentClient(
                path=CHROMA_DB_DIR,
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:

                self.collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
                logger.info(f"Connected to collection: {COLLECTION_NAME}")

            except:

                self.collection = client.create_collection(
                    name = COLLECTION_NAME,
                    embedding_function = embed_fn,
                    metadata = {"hnsw:space": "cosine"}
                )

                logger.info(f"Created collection: {COLLECTION_NAME}")

        except Exception as e:

            logger.error(f"ChromaDB init failed: {e}")
    
    def extract_funcs(self, code: str) -> List[FunctionInfo]:

        try:

            tree = ast.parse(code)
            extractor = NumpyFunctionExtractor()
            extractor.visit(tree)

            return extractor.funcs
        
        except Exception as e:

            logger.error(f"Function extraction failed: {e}")

            return []
    
    def _func_matches_content(self, func: str, content: str) -> bool:

        func_parts = func.replace('np.', '').replace('numpy.', '').split('.')
        content_lower = content.lower()

        return any(part.lower() in content_lower for part in func_parts if len(part) > 2)
    
    def query_db(self, func: str, version: str) -> List[Dict[str, Any]]:

        if not self.collection:
            return []
        
        try:

            base_func = func.split('.')[-1]
            variations = [base_func, func, f"numpy.{base_func}", f"np.{base_func}"]
            seen_content = set()
            unique_chunks = []
            
            for variant in variations:
                query = f"{variant} numpy {version} deprecated"
                
                res = self.collection.query(
                    query_texts = [query],
                    n_results = 3,
                    include = ["documents", "metadatas", "distances"]
                )
                
                if res['documents'] and res['documents'][0]:

                    for i, doc in enumerate(res['documents'][0]):

                        score = 1 - res['distances'][0][i]

                        if score >= 0.4 and self._func_matches_content(func, doc):

                            content_hash = hash(doc[:200])

                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                chunk_data = {
                                    'content': doc,
                                    'metadata': res['metadatas'][0][i] if res['metadatas'] else {},
                                    'similarity_score': score
                                }

                                unique_chunks.append(chunk_data)
            
            unique_chunks.sort(key = lambda x: x['similarity_score'], reverse = True)
            return unique_chunks[:3]
            
        except Exception as e:

            logger.error(f"DB query failed for {func}: {e}")
            return []
    
    def extract_changes(self, output: str) -> tuple[str, List[str]]:

        if "---EXPLANATION---" not in output:
            return output.strip(), []
        
        parts = output.split("---EXPLANATION---")
        code = parts[0].strip()
        explanation = parts[1].strip()
        
        if not explanation or "no deprecated functionality" in explanation.lower():
            return code, []
            
        return code, [explanation]
    
    def analyze_code(self, code: str, version: str) -> CodeAnalysisResponse:

        logger.info(f"Analyzing code with NumPy {version}")
        
        funcs = self.extract_funcs(code)
        unique_funcs = list({f.name for f in funcs})
        
        logger.info(f"Found {len(unique_funcs)} unique NumPy functions: {unique_funcs}")
        
        ctx: Dict[str, List[Dict[str, Any]]] = {}

        for fn in unique_funcs:
            ctx[fn] = self.query_db(fn, version)
        
        retrieved_context = {}

        for fn, chunks in ctx.items():
            if chunks:
                retrieved_context[fn] = [chunks[0]['content']]
        
        if self.model.is_available():

            try:

                output = self.model.call_model(code, version, unique_funcs, ctx)
                modernized_code, changes = self.extract_changes(output)

                return CodeAnalysisResponse(
                    modernized_code = modernized_code,
                    retrieved_context = retrieved_context,
                    changes = changes
                )
            
            except Exception as e:

                logger.error(f"Model call failed: {e}")

                return CodeAnalysisResponse(
                    modernized_code = "",
                    retrieved_context = retrieved_context,
                    changes = [],
                    error = str(e)
                )
            
        else:
            
            return CodeAnalysisResponse(
                modernized_code = "",
                retrieved_context = retrieved_context,
                changes = [],
                error = "Model service unavailable"
            )
    
    def is_connected(self) -> bool:
        return self.collection is not None
    
    def is_model_available(self) -> bool:
        return self.model.is_available()
import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import torch


class NumpyDocProcessor:

    def __init__(self):
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.embeddings = HuggingFaceEmbeddings(
            model_name = "BAAI/bge-base-en-v1.5",
            model_kwargs = {'device': device},
            encode_kwargs = {'normalize_embeddings': True}
        )
        db_path = Path("data/docs/chroma_db")
        db_path.mkdir(parents = True, exist_ok = True)    
        self.chroma = Chroma(
            collection_name = "numpy_docs",
            persist_directory = str(db_path),
            embedding_function = self.embeddings
        )

    def parse_markdown(self, content):
        chunks = []
        current_version = None
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('## Version'):
                current_version = line.replace('## Version ', '').strip()
                i += 1
                continue
            
            if line.startswith('**') and line.endswith('**'):
                func_name = line[2:-2]
                i += 1
                replacement = None
                context = None
                while i < len(lines) and not (lines[i].strip().startswith('**') or lines[i].strip().startswith('## Version')):
                    if lines[i].strip().startswith('- **Replacement**:'):
                        replacement = lines[i].strip()[18:].strip()
                    elif lines[i].strip().startswith('- **Context**:'):
                        context = lines[i].strip()[14:].strip()
                    i += 1
                if current_version and (replacement or context):
                    chunk_text = f"NumPy {current_version} - {func_name}"
                    if replacement:
                        chunk_text += f"\nReplacement: {replacement}"
                    if context:
                        chunk_text += f"\nContext: {context}"
                    chunks.append({
                        'content': chunk_text,
                        'metadata': {
                            'version': current_version,
                            'function': func_name,
                            'has_replacement': bool(replacement)
                        }
                    })
                continue
            i += 1
        return chunks

    def create_documents(self, chunks):
        docs = []
        for chunk in chunks:
            doc = Document(
                page_content = chunk['content'],
                metadata = chunk['metadata']
            )
            docs.append(doc)
        return docs

    def process_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        chunks = self.parse_markdown(content)
        docs = self.create_documents(chunks)
        if docs:
            self.chroma.add_documents(docs)
            print(f"Added {len(docs)} chunks to ChromaDB")
        return len(docs)


if __name__ == "__main__":
    processor = NumpyDocProcessor()
    md_file = Path("data/docs/numpy_deprecations.md")
    if md_file.exists():
        count = processor.process_file(md_file)
        print(f"Processing complete. Total chunks: {count}")
    else:
        print(f"File not found: {md_file}")
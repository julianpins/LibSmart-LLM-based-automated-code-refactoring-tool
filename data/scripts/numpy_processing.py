import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import time
from typing import List, Optional, Dict
from pathlib import Path
import torch

class NumpyDocsProcessor:
    def __init__(self):
        self.base_url = "https://numpy.org/devdocs/release/"
        self.versions = [
            "2.4.0", "2.3.0", "2.2.6", "2.2.5", "2.2.4", "2.2.3", "2.2.2", "2.2.1", "2.2.0",
            "2.1.3", "2.1.2", "2.1.1", "2.1.0", "2.0.2", "2.0.1", "2.0.0",
            "1.26.4", "1.26.3", "1.26.2", "1.26.1", "1.26.0",
            "1.25.2", "1.25.1", "1.25.0",
            "1.24.4", "1.24.3", "1.24.2", "1.24.1", "1.24.0",
            "1.23.5", "1.23.4", "1.23.3", "1.23.2", "1.23.1", "1.23.0",
            "1.22.4", "1.22.3", "1.22.2", "1.22.1", "1.22.0",
            "1.21.6", "1.21.5", "1.21.4", "1.21.3", "1.21.2", "1.21.1", "1.21.0",
            "1.20.3", "1.20.2", "1.20.1", "1.20.0",
            "1.19.5", "1.19.4", "1.19.3", "1.19.2", "1.19.1", "1.19.0"
        ]
        
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {device}")
        print("Loading BGE embedding model...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )
        
        db_path = Path("data/docs/chroma_db")
        db_path.mkdir(parents=True, exist_ok=True)
        
        self.chroma = Chroma(
            collection_name="numpy_docs",
            persist_directory=str(db_path),
            embedding_function=self.embeddings
        )
    
    def fetch_release_notes(self, version: str) -> Optional[Dict]:
        url = f"{self.base_url}{version}-notes.html"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            main_content = soup.find('div', {'role': 'main'})
            if not main_content:
                main_content = soup.find('div', class_='document')
            if not main_content:
                main_content = soup.find('div', class_='body')
            if not main_content:
                main_content = soup.find('main')
            if not main_content:
                main_content = soup.find('article')
                
            if main_content:
                content = self._extract_structured_content(main_content)
                return {
                    "version": version,
                    "url": url,
                    "content": content,
                    "sections": self._extract_sections(main_content)
                }
            else:
                print(f"  Warning: Could not find main content for {version}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {version}: {e}")
            return None
    
    def _extract_structured_content(self, soup_element) -> str:
        lines = []
        
        for element in soup_element.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'pre', 'code']):
            if element.name.startswith('h'):
                level = int(element.name[1])
                prefix = "#" * level
                lines.append(f"\n{prefix} {element.get_text(strip=True)}\n")
            elif element.name in ['pre', 'code']:
                code_text = element.get_text()
                if element.name == 'pre' or len(code_text) > 50:
                    lines.append(f"\n```python\n{code_text}\n```\n")
                else:
                    lines.append(f"`{code_text}`")
            else:
                text = element.get_text(strip=True)
                if text:
                    lines.append(text)
        
        return "\n".join(lines)
    
    def _extract_sections(self, soup_element) -> Dict[str, str]:
        sections = {}
        
        for header in soup_element.find_all(['h2', 'h3']):
            header_text = header.get_text(strip=True).lower()
            
            if any(keyword in header_text for keyword in ['deprecat', 'expired', 'removed', 'breaking', 'compatibility']):
                section_content = []
                for sibling in header.find_next_siblings():
                    if sibling.name and sibling.name.startswith('h') and sibling.name <= header.name:
                        break
                    section_content.append(sibling.get_text(strip=True))
                
                sections[header_text] = "\n".join(section_content)
        
        return sections
    
    def create_chunks(self, content: dict, version: str) -> List[Document]:
        main_text = content['content']
        chunks = self.splitter.split_text(main_text)
        
        documents = []
        
        for i, chunk in enumerate(chunks):
            is_deprecation = any(word in chunk.lower() for word in ['deprecat', 'removed', 'expired', 'breaking'])
            
            doc = Document(
                page_content=chunk,
                metadata={
                    "package": "numpy",
                    "version": version,
                    "url": content['url'],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "has_deprecation": is_deprecation,
                    "chunk_type": "release_notes"
                }
            )
            documents.append(doc)
        
        for section_name, section_content in content.get('sections', {}).items():
            if section_content.strip():
                doc = Document(
                    page_content=f"Section: {section_name}\n\n{section_content}",
                    metadata={
                        "package": "numpy",
                        "version": version,
                        "url": content['url'],
                        "section_name": section_name,
                        "has_deprecation": True,
                        "chunk_type": "deprecation_section"
                    }
                )
                documents.append(doc)
        
        return documents
    
    def process_version(self, version: str) -> int:
        content = self.fetch_release_notes(version)
        if content:
            chunks = self.create_chunks(content, version)
            if chunks:
                self.chroma.add_documents(chunks)
                print(f"  Added {len(chunks)} chunks to Chroma")
                return len(chunks)
        return 0
    
    def run(self):
        print(f"\nProcessing {len(self.versions)} NumPy versions...")
        
        start_time = time.time()
        total_chunks = 0
        successful_versions = 0
        
        for i, version in enumerate(self.versions):
            print(f"[{i+1}/{len(self.versions)}] Processing NumPy {version}...", flush=True)
            
            chunks_added = self.process_version(version)
            if chunks_added > 0:
                total_chunks += chunks_added
                successful_versions += 1
            else:
                print("  No content found")
            
            time.sleep(0.5)
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"{'='*50}")
        print(f"Stats:")
        print(f"   - Versions processed: {successful_versions}/{len(self.versions)}")
        print(f"   - Total chunks created: {total_chunks}")
        print(f"   - Time elapsed: {elapsed_time/60:.1f} minutes")
        print(f"   - Database location: data/docs/chroma_db/")
        
        print(f"\n Verifying database...")
        doc_count = self.chroma._collection.count()
        print(f"   - Documents in Chroma: {doc_count}")
        print(f"{'='*50}")

if __name__ == "__main__":
    processor = NumpyDocsProcessor()
    processor.run()
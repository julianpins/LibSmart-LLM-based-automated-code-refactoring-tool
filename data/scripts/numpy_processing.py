import requests
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import time
from pathlib import Path
import torch
import re
from typing import List, Dict, Set


class NumpyDocScraper:
    def __init__(self):
        self.base_url = "https://numpy.org/doc/stable/release/"
        self.versions = [
            "2.3.0", "2.2.0", "2.1.0", "2.0.0",
            "1.26.0", "1.25.0", "1.24.0", "1.23.0", "1.22.0", 
            "1.21.0", "1.20.0", "1.19.0", "1.18.0", "1.17.0", "1.16.0"
        ]
        
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.embeddings = HuggingFaceEmbeddings(
            model_name = "BAAI/bge-base-en-v1.5",
            model_kwargs = {'device': device},
            encode_kwargs = {'normalize_embeddings': True}
        )
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1200,
            chunk_overlap = 200,
            separators = ["\n\n", "\n", ". ", " ", ""],
            length_function = len
        )
        
        db_path = Path("data/docs/chroma_db")
        db_path.mkdir(parents = True, exist_ok = True)
        
        self.chroma = Chroma(
            collection_name = "numpy_docs",
            persist_directory = str(db_path),
            embedding_function = self.embeddings
        )

    # Fetch and parse release notes
    def fetch_notes(self, version: str) -> BeautifulSoup:
        url = f"{self.base_url}{version}-notes.html"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return BeautifulSoup(resp.content, "html.parser")

    # Extract text preserving formatting
    def extract_text(self, elem) -> str:
        if isinstance(elem, NavigableString):
            return str(elem)
        
        if not isinstance(elem, Tag):
            return ""
        
        # Special handling for code elements
        if elem.name in ['code', 'tt', 'pre']:
            return elem.get_text()
        
        # Handle line breaks
        if elem.name == 'br':
            return '\n'
        
        # Block elements need newlines
        if elem.name in ['p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            content = ''.join(self.extract_text(child) for child in elem.children)
            return f"\n{content.strip()}\n"
        
        # Inline elements
        return ''.join(self.extract_text(child) for child in elem.children)

    # Find and extract relevant sections
    def extract_sections(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        sections = []
        main_content = soup.find('div', class_='rst-content') or soup.find('main')
        
        if not main_content or not isinstance(main_content, Tag):
            return sections
        
        current_section = None
        current_header = None
        current_content = []
        
        for elem in main_content.find_all(True):  # Find all tags
            if not isinstance(elem, Tag):
                continue
            
            # Skip deeply nested elements
            if elem.parent and elem.parent != main_content and elem.parent.name not in ['section', 'div']:
                continue
            
            if elem.name in ['h1', 'h2', 'h3']:
                # Save previous section
                if current_section and current_content:
                    text = '\n'.join(current_content).strip()
                    if text:
                        sections.append({
                            'type': current_section,
                            'header': current_header,
                            'content': text
                        })
                
                # Start new section
                header_text = elem.get_text(strip=True)
                header_lower = header_text.lower()
                
                if 'deprecat' in header_lower:
                    current_section = 'deprecations'
                elif 'expired' in header_lower:
                    current_section = 'expired'
                elif any(word in header_lower for word in ['remov', 'delet']):
                    current_section = 'removals'
                elif 'compatibility' in header_lower:
                    current_section = 'compatibility'
                elif 'change' in header_lower or 'api' in header_lower:
                    current_section = 'changes'
                else:
                    current_section = None
                
                current_header = header_text
                current_content = [f"### {header_text}"]
            
            elif current_section and elem.name in ['p', 'ul', 'ol', 'dl', 'blockquote', 'pre']:
                text = self.extract_text(elem).strip()
                if text and text not in current_content:
                    current_content.append(text)
        
        # Don't forget last section
        if current_section and current_content:
            text = '\n'.join(current_content).strip()
            if text:
                sections.append({
                    'type': current_section,
                    'header': current_header,
                    'content': text
                })
        
        return sections

    # Extract function names from text
    def extract_functions(self, text: str) -> Set[str]:
        patterns = [
            r'numpy\.([a-zA-Z_]\w*)',
            r'np\.([a-zA-Z_]\w*)',
            r'`([a-zA-Z_]\w*)`(?:\s+is|\s+has|\s+was|\s+deprecated)',
            r'``([a-zA-Z_]\w*)``',
            r':func:`~?numpy\.([a-zA-Z_]\w*)`',
            r':meth:`~?numpy\.([a-zA-Z_]\w*)`',
            r'([a-zA-Z_]\w*)\s+(?:is|has been|was)\s+deprecated'
        ]
        
        funcs = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            funcs.update(m for m in matches if len(m) > 2 and m not in ['the', 'been', 'was', 'has'])
        
        return funcs

    # Create focused chunks
    def create_chunks(self, section: Dict[str, str], version: str, url: str) -> List[Document]:
        content = section['content']
        if not content.strip():
            return []
        
        # For deprecation sections, try to split by individual deprecations
        if section['type'] in ['deprecations', 'expired', 'removals']:
            chunks = self.split_deprecations(content)
        else:
            chunks = self.splitter.split_text(content)
        
        docs = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            funcs = self.extract_functions(chunk)
            
            # Create focused chunk with context
            enhanced = f"NumPy {version} - {section['header']}\n\n{chunk}"
            
            metadata = {
                "version": version,
                "section_type": section['type'],
                "section_header": section['header'],
                "url": url,
                "chunk_id": i,
                "functions": ', '.join(sorted(funcs)[:10]),
                "num_functions": len(funcs),
                "has_deprecation": bool(re.search(r'deprecat', chunk, re.I)),
                "has_removal": bool(re.search(r'remov|delet', chunk, re.I)),
                "has_replacement": bool(re.search(r'use\s+\S+\s+instead|replaced?\s+by', chunk, re.I))
            }
            
            docs.append(Document(page_content = enhanced, metadata = metadata))
        
        return docs

    # Split deprecation sections
    def split_deprecations(self, text: str) -> List[str]:
        # Try to split by function-level deprecations
        lines = text.split('\n')
        chunks = []
        current = []
        
        for line in lines:
            if re.match(r'^[#\-\*]\s*`?\w+', line) or re.match(r'^\w+\s+(is|has been)\s+deprecated', line):
                if current:
                    chunks.append('\n'.join(current))
                    current = [line]
                else:
                    current.append(line)
            else:
                current.append(line)
        
        if current:
            chunks.append('\n'.join(current))
        
        # If no good splits found, use regular splitter
        if len(chunks) <= 1:
            return self.splitter.split_text(text)
        
        # Ensure chunks aren't too long
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > 1500:
                final_chunks.extend(self.splitter.split_text(chunk))
            else:
                final_chunks.append(chunk)
        
        return final_chunks

    # Process single version
    def process_version(self, version: str) -> int:
        print(f"Processing NumPy {version}...")
        
        try:
            soup = self.fetch_notes(version)
            sections = self.extract_sections(soup)
            
            if not sections:
                print(f"  No sections found")
                return 0
            
            all_docs = []
            url = f"{self.base_url}{version}-notes.html"
            
            for section in sections:
                docs = self.create_chunks(section, version, url)
                all_docs.extend(docs)
            
            if all_docs:
                self.chroma.add_documents(all_docs)
                print(f"  Stored {len(all_docs)} chunks from {len(sections)} sections")
                return len(all_docs)
            else:
                print(f"  No chunks created")
                return 0
                
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return 0

    # Main execution
    def run(self):
        total = 0
        
        for version in self.versions:
            count = self.process_version(version)
            total += count
            time.sleep(0.5)
        
        print(f"\nComplete. Total chunks: {total}")


if __name__ == "__main__":
    scraper = NumpyDocScraper()
    scraper.run()
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

def main():

    device = 'mps'

    print("Loading BGE embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("Loading Chroma vector store...")
    db = Chroma(
        collection_name="numpy_docs",
        persist_directory="data/docs/chroma_db",
        embedding_function=embeddings
    )

    all_docs = db.get()
    print(f"Total documents in collection: {len(all_docs['documents'])}")

    query = "np.find_common_type"

    print(f"Running similarity search for query: '{query}'\n")
    results = db.similarity_search(query, k=5)

    for i, doc in enumerate(results):
        print(f"\n--- Result #{i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content:\n{doc.page_content[:500]}...")

if __name__ == "__main__":
    main()

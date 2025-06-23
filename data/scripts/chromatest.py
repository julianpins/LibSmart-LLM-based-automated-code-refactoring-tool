from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

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

    query = "np.irr"

    print(f"Running similarity search for query: '{query}'\n")
    results = db.similarity_search_with_relevance_scores(query, k=3)

    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result #{i+1} ---")
        print(f"Score: {score:.4f}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content:\n{doc.page_content[:750]}...")

if __name__ == "__main__":
    main()

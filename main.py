from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from rag_pipeline import (
    split_documents,
    EmeddingManager,
    VectorStore,
    RAGRetriever,
    rag_simple,
    llm
)

BASE_DIR = Path(__file__).parent
VECTOR_STORE_PATH = str(BASE_DIR / "chroma_db")
PDF_DIR = "."

def build_and_save_vectorstore():
    emedding_manager = EmeddingManager()
    vectorstore = VectorStore(persist_directory=VECTOR_STORE_PATH)

    pdf_dir = Path(PDF_DIR)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to scan")

    new_files_found = 0

    for pdf_file in pdf_files:
        existing = vectorstore.collection.get(
            where={"source_file": pdf_file.name}
        )
        if existing and existing["ids"]:
            print(f"  Already indexed, skipping: {pdf_file.name}")
            continue

        print(f" New file found: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            chunks = split_documents(documents)
            texts = [doc.page_content for doc in chunks]
            embeddings = emedding_manager.generate_emeddings(texts)
            vectorstore.add_documents(chunks, embeddings)
            new_files_found += 1
            print(f" Added: {pdf_file.name} — {len(chunks)} chunks")

        except Exception as e:
            print(f" Error processing {pdf_file.name}: {e}")

    if new_files_found == 0:
        print("No new PDFs found. Vector store is up to date.")

    print(f" Total docs in store: {vectorstore.collection.count()}")


def load_retriever() -> RAGRetriever:
    print("Loading retriever from existing vector store...")
    emedding_manager = EmeddingManager()
    vectorstore = VectorStore(persist_directory=VECTOR_STORE_PATH)

    if vectorstore.collection.count() == 0:
        print("ChromaDB is empty, building vector store now...")
        build_and_save_vectorstore()

    rag_retriever = RAGRetriever(vectorstore, emedding_manager)
    print(" Retriever loaded successfully")
    return rag_retriever


if __name__ == "__main__":
    build_and_save_vectorstore()

    # paste this temporarily in main.py inside if __name__ == "__main__"
# to see ALL chunks from HR POLICIES and IT SECURITY

    retriever = load_retriever()

    print("\n--- ALL HR POLICIES CHUNKS ---")
    results = retriever.vector_store.collection.get(
        where={"source_file": "HR POLICIES.pdf"}
    )
    for i, doc in enumerate(results["documents"]):
        print(f"\n[Chunk {i+1}]")
        print(doc[:300])
        print("---")

    print("\n--- ALL IT AND SECURITY CHUNKS ---")
    results = retriever.vector_store.collection.get(
        where={"source_file": "IT AND SECURITY POLICIES.pdf"}
    )
    for i, doc in enumerate(results["documents"]):
        print(f"\n[Chunk {i+1}]")
        print(doc[:300])
        print("---")

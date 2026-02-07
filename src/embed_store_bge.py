from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

def create_vector_store_bge(
    chunks,
    persist_dir: str = "./chroma_bge"
):
    """
    Create a ChromaDB vector store using bge-m3 embeddings.

    Args:
        chunks (list): List of LangChain Document objects
        persist_dir (str): Directory to persist ChromaDB data

    Returns:
        Chroma: ChromaDB vector store instance
    """

    #1 Initialize BGE-M3 embeddings via Ollama
    embeddings = OllamaEmbeddings(
        model="bge-m3"
    )

    #2 Create (or load) ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectorstore
"""      
# --- TEST BLOCK ---
# Run this file directly to verify it works
if __name__ == "__main__":
    from load_pdf import load_pdf_file
    from chunker import split_documents
    import shutil  # Used to clean up old DB for testing

    test_pdf = "D:\RAG LANGCHAIN PROJECT\data\data_file.pdf"

    # Optional: Clear old database for a fresh test
    # if os.path.exists(DB_PATH):
    #     shutil.rmtree(DB_PATH)

    if os.path.exists(test_pdf):
        print("1. Loading PDF...")
        docs = load_pdf_file(test_pdf)

        print("2. Splitting Text...")
        chunks = split_documents(docs)

        print("3. Creating Chroma Database...")
        retriever = create_vector_store_bge(chunks)

        # Test the search
        query = "What is Version control ?"
        print(f"\n--- Testing Search for: '{query}' ---")
        results = retriever.invoke(query)

        if results:
            print(f" Success! Found {len(results)} relevant results.")
            print(f"Preview: {results[0].page_content[:100]}...")
        else:
            print(" No results found.")
 """

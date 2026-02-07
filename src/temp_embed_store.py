from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

def temp_create_vector_store_bge(chunks,persist_dir: str = "./chroma_bge"):
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
        model="nomic-embed-text"
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
if __name__ == "__main__":
    # Ensure you have your other modules (load_pdf, chunker) available
    try:
        from load_pdf import load_pdf_file
        from chunker import split_documents

        test_pdf = r"D:\RAG LANGCHAIN PROJECT\data\data_file.pdf"

        if os.path.exists(test_pdf):
            print("1. Loading PDF...")
            docs = load_pdf_file(test_pdf)

            print("2. Splitting Text...")
            chunks = split_documents(docs)

            print("3. Creating FAISS Database...")
            vector_store = create_vector_store_faiss(chunks)

            # Convert to retriever for testing
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})

            # Test the search
            query ="Who are the members of CMSC?"
            print(f"\n--- Testing Search for: '{query}' ---")

            # Note: FAISS objects don't have .invoke(), but the retriever does
            results = retriever.invoke(query)

            if results:
                print(f"✅ Success! Found {len(results)} relevant results.")
                print(f"Preview: {results[0].page_content[:150]}...")
            else:
                print("❌ No results found.")

    except ImportError:
        print("Could not import helper modules. Run this from your project root.")
"""
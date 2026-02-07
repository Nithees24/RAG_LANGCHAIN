import os
from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf_file(file_path: str):
    # 1. Check if file exists first
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return []

    print(f"--- Loading {file_path} ---")

    try:
        # PyMuPDF is optimized for speed and complex layouts for reading the PDF
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # 2. Check if the PDF was empty or readable
        if not docs:
            print("Error: PDF loaded but no text found")
            return []

        print(f"Successfully loaded {len(docs)} pages.")
        return docs

    except Exception as e:
        print(f"Unexpected Error loading PDF: {e}")
        return []


"""
******************************************
LOAD MODULE HAS BEEN TESTED AND CONFIGURED
******************************************

if __name__ == "__main__":
    # Create a dummy PDF or point to an existing one to test
    test_pdf = "D:\RAG LANGCHAIN PROJECT\data\data_file.pdf"  # Make sure this file exists in your project folder

    if not os.path.exists(test_pdf):
        print(f" Please put a PDF named '{test_pdf}' in this folder to test.")
    else:
        loaded_docs = load_pdf_file(test_pdf)
        if loaded_docs:
            print(f"Test Preview: {loaded_docs[1].page_content[:1000]}...")

"""
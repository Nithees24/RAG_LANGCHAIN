from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs):
    """
    """
    print("--- Splitting Documents ---")

    # Configure the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,  # Size of each chunk in characters
        chunk_overlap=80,  # Overlap to keep context between chunks
        length_function=len,
        is_separator_regex=False,
    )

    # Perform the split
    splits = text_splitter.split_documents(docs)

    if not splits:
        print(" Error: No chunks created.")
        return []

    print(f" Split {len(docs)} pages into {len(splits)} chunks.")

    return splits

"""

**********************************
CHUNK MODULE VERIFIED AND EXECUTED
**********************************

# --- TEST BLOCK ---
# Run this file directly to test if splitting works
if __name__ == "__main__":
    # We need to import the loader from Step 1 to test this
    from load_pdf import load_pdf_file
    import os

    test_pdf = "D:\RAG LANGCHAIN PROJECT\data\data_file.pdf"  # Ensure this file exists

    if os.path.exists(test_pdf):
        # 1. Load
        raw_docs = load_pdf_file(test_pdf)

        # 2. Split (Testing this module)
        if raw_docs:
            my_chunks = split_documents(raw_docs)

            # Show the first chunk content to verify
            print("\n--- Preview of First Chunk ---")
            print(my_chunks[0].page_content)
            print("------------------------------")
    else:
        print(f"File '{test_pdf}' not found. Please add it to test.")

"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Custom modules
from src.load_pdf import load_pdf_file
from src.temp_chunker import temp_split_documents
from src.temp_embed_store import temp_create_vector_store_bge


def run_rag_pipeline(pdf_path: str):
    print("\n--- Initializing RAG Pipeline ---")

    # Load & split PDF
    docs = load_pdf_file(pdf_path)
    if not docs:
        print("No documents loaded")
        return None

    chunks = temp_split_documents(docs)

    #  Vector store (BGE-M3 + ChromaDB)
    vector_store = temp_create_vector_store_bge(chunks)

    # Vector-only search (score-aware)
    def simple_search(query: str):
        # IMPORTANT: score-aware search
        results = vector_store.similarity_search_with_score(
            query,
            k=10  # recall width (increase if needed)
        )

        # Chroma similarity: LOWER score = BETTER match
        filtered_docs = []
        for doc, score in results:
            if score < 0.85:  # relevance threshold (tune 0.75–0.9)
                filtered_docs.append(doc)

        # Fallback: if nothing passed threshold, keep top 3 anyway
        if not filtered_docs:
            filtered_docs = [doc for doc, _ in results[:3]]

        # Deduplicate by content
        unique_docs = {
            doc.page_content: doc
            for doc in filtered_docs
        }

        # Final strict cap (context safety)
        return list(unique_docs.values())[:5]

    # LLM (LLaMA 3.2 – 3B)
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    #  Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are an expert analyst assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not present, say:
  "Not found in the provided document."

Context:
{context}

Question:
{question}
"""
    )

    #  Format context
    def format_docs(docs):
        if not docs:
            return "NO_RELEVANT_CONTEXT_FOUND"
        return "\n\n".join(doc.page_content for doc in docs)

    # Build RAG chain
    rag_chain = (
        {
            "context": RunnableLambda(simple_search)
                       | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# CLI entry point
if __name__ == "__main__":

    pdf_path = "D:/RAG LANGCHAIN PROJECT/data/data_file.pdf"

    chain = run_rag_pipeline(pdf_path)

    if chain:
        print("\n" + "=" * 55)
        print(" BOT READY")
        print("=" * 55 + "\n")

        while True:
            query = input("You: ").strip()
            if query.lower() in {"exit", "quit"}:
                break

            try:
                response = chain.invoke(query)
                print(f"\nBot:\n{response}")
                print("-" * 55)
            except Exception as e:
                print(f" Error: {e}")

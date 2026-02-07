from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest
import time
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

#1)
# Custom modules (Ensure these files exist in your folder)
from src.load_pdf import load_pdf_file
from src.chunker import split_documents
from src.embed_store_bge import create_vector_store_bge

#2)
def run_rag_pipeline(pdf_path):
    logger.info("Initializing RAG pipeline")
    # 1. Load & Split

    #2A)
    docs = load_pdf_file(pdf_path)
    if not docs:
        logger.error("No documents loaded from PDF")
        return None
    logger.info("Loaded %d documents", len(docs))

    #2B)
    chunks = split_documents(docs)
    logger.info("Split documents into %d chunks", len(chunks))
    # 2. Build Retrievers (Wide Net Strategy)
    # We fetch 20 results from each source to ensure we don't miss the answer.
    logger.info("Creating vector store using BGE-M3 embeddings") #BAAI/BGM3 Basic general embedding model 3

    #2C)
    #VECTOR STORE RETRIEVER
    vector_store = create_vector_store_bge(chunks)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 30})

    #KEYWORD  RETRIEVER
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 30

    # 3) Initialize Re-Ranker (The Quality Control Layer)
    # This runs locally and filters the 40 docs down to the best 5.
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./opt")

    # 4) Advanced Search Logic
    def smart_search(query: str):
        #4A)____VECTOR AND KEYWORD SEARCH
        logger.info("smart_search started")
        logger.info("Query: %s", query)

        start_time = time.perf_counter()

        # --- Vector retrieval ---
        vector_docs = vector_retriever.invoke(query)
        logger.info("Vector retriever returned %d docs", len(vector_docs))

        # --- Keyword retrieval ---
        keyword_docs = keyword_retriever.invoke(query)
        logger.info("BM25 retriever returned %d docs", len(keyword_docs))

        # --- Combine & deduplicate ---
        unique_docs = {
            doc.page_content: doc
            for doc in vector_docs + keyword_docs
        }
        candidate_docs = list(unique_docs.values())
        logger.info(
            "After deduplication: %d unique candidate docs",
            len(candidate_docs)
        )

        # 4B)--- Prepare passages for reranker ---
        passages = [
            {
                "id": str(i),
                "text": doc.page_content,
                "meta": doc.metadata
            }
            for i, doc in enumerate(candidate_docs)
        ]
        logger.info("Prepared %d passages for reranking", len(passages))

        # --- Rerank ---
        rerank_start = time.perf_counter()
        rerank_request = RerankRequest(
            query=query,
            passages=passages
        )
        ranked_results = ranker.rerank(rerank_request)
        rerank_time = time.perf_counter() - rerank_start

        logger.info(
            "Reranking completed in %.3f sec",
            rerank_time
        )

        # --- Select final top-N ---
        top_docs = []
        for result in ranked_results[:7]:
            original_doc = unique_docs.get(result["text"])
            if original_doc:
                top_docs.append(original_doc)

        #  FIX: define total_time BEFORE logging
        total_time = time.perf_counter() - start_time

        logger.info(
            "smart_search finished | final_docs=%d | total_time=%.3f sec",
            len(top_docs),
            total_time
        )

        # Optional DEBUG preview
        logger.debug(
            "Top chunk previews: %s",
            [doc.page_content[:80] for doc in top_docs]
        )

        return top_docs
    # 5. Initialize LLM
    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    # 6. Professional Prompt
    template = """You are an expert analyst assistant. 
    Answer the user question based STRICTLY on the context provided below.

    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 7. Helper to format text
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # 8. Build the Clean Chain
    rag_chain = (
            {
                "context": RunnableLambda(smart_search) | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain



if __name__ == "__main__":
    # UPDATE THIS PATH TO YOUR PDF
    pdf_path = "D:\RAG LANGCHAIN PROJECT\data\data_file.pdf"

    chain = run_rag_pipeline(pdf_path)


    if chain:
        print("\n" + "=" * 50)
        print("BOT READY")
        print("=" * 50 + "\n")

        while True:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]: break

            try:
                # Direct answer, no "Thinking..." noise
                start_total = time.perf_counter()

                start_llm = time.perf_counter()
                response = chain.invoke(query)
                llm_time = time.perf_counter() - start_llm

                total_time = time.perf_counter() - start_total

                print(f"\nBot:\n{response}")
                print(f"\nLLM time: {llm_time:.3f} sec")
                print(f"Total time: {total_time:.3f} sec")
                print("-" * 50)
            except Exception as e:
                print(f"Error: {e}")
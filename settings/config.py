#switch to use Embeddings and LLM API calls
USE_API_LLM=False
USE_API_EMBED = False

GEMINI_MODEL="gemini-2.5-flash"
GEMINI_EMBED_MODEL="models/gemini-embedding-2-preview"
GEMINI_TEMPERATURE=0.0

LOCAL_EMBED_MODEL="bge-m3:latest"
LOCAL_MODEL="llama3.2:3b"
LOCAL_TEMPERATURE=0.0
LOCAL_URL="http://localhost:11434/"

PINECONE_INDEX="pdf-rag-bge"
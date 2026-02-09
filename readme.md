Tokenization:


this is process of chunking the data for the process by LLM. Text tokenization is 
the fundamental process in Natural Language Processing (NLP) of breaking down a body of text into 
smaller, meaningful units called tokens

This is the first process we are doing here


Vector DB


FAAS:

Its converts the chunked tokens into numerical values known as embedding values. These values are stored in FAISS vectors known as vectors. Then the vectors are compared 
with similar vectors stored in FAISS index with KNN concept. Then the values are formulated and the nearest meaning is found to project the answer.
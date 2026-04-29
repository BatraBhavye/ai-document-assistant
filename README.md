# AI Document Assistant (Local & Private RAG Pipeline)

An enterprise-grade Retrieval-Augmented Generation (RAG) system engineered for secure, offline document intelligence. 

This project addresses critical bottlenecks in modern AI implementations: data privacy, LLM hallucinations, and retrieval accuracy. By executing the entire pipeline locally, it guarantees zero data leakage while providing verifiable, context-grounded answers.

## Performance & Evaluation
The system's reliability is mathematically validated using the **RAGAS (Retrieval Augmented Generation Assessment)** framework.
* **Faithfulness Score: 1.0** (The model relies 100% on the provided document, successfully eliminating hallucinations).
* **Answer Relevancy: ~0.76** (The generated responses accurately and directly address the user's queries).

## System Architecture
This pipeline moves beyond basic vector search by implementing a robust, multi-layered retrieval strategy:

1. **Semantic Chunking:** Utilizes LangChain's `RecursiveCharacterTextSplitter` to preserve paragraph and sentence structure, ensuring context is not destroyed during the embedding phase.
2. **Hybrid Indexing:** Simultaneously builds a Dense Vector Index (FAISS) for semantic understanding and a Sparse Keyword Index (BM25) for exact-match terminology retrieval.
3. **Fused Retrieval:** Implements Reciprocal Rank Fusion (RRF) to merge and rank results from both indexes, optimizing overall context accuracy.
4. **Grounded Generation:** Custom prompt engineering forces the local LLM to restrict its answers to the retrieved context and append page-level source citations.

## Technical Stack
* **Language:** Python
* **LLM Engine:** Ollama (Phi-3 for generation, Llama-3 for evaluation)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`, `nomic-embed-text`)
* **Frameworks & Libraries:** LangChain, rank-bm25, RAGAS, PyPDF
* **Frontend:** Streamlit

## Installation & Quick Start

**1. Clone the repository**

```bash
git clone [https://github.com/BatraBhavye/ai-document-assistant.git](https://github.com/BatraBhavye/ai-document-assistant.git)
cd ai-document-assistant
```

**2. Install dependencies** 

```Bash
pip install -r requirements.txt
```

**3. Initialize Local Models**
Ensure Ollama is installed on your system, then pull the required models:

```Bash
ollama pull phi3
ollama pull llama3
ollama pull nomic-embed-text
```

**4. Run the Application**

```Bash
streamlit run app.py
```

## Project Structure
**app.py:** The main Streamlit application handling the frontend UI and session state.

**utils.py:** The core engine handling PDF parsing, hybrid chunking, FAISS/BM25 indexing, and semantic search routing.

**evaluate.py:** An automated testing script utilizing RAGAS to generate and record performance metrics.

**evaluation_results.csv:** The output log of the system's faithfulness and relevancy scores.


## Future Roadmap
**Multi-Document Interoperability:** Expanding the vector store to query across entire directories of PDFs simultaneously.

**Optical Character Recognition (OCR):** Integrating Tesseract to handle scanned, non-searchable document formats.

**REST API Mode:** Decoupling the Streamlit frontend to deploy the backend as a headless API for broader enterprise integration.

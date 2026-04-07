# AI Document Assistant (RAG-Based QA System)

An intelligent document question-answering system built using **Retrieval-Augmented Generation (RAG)**.  
This project enables users to query PDF documents and receive accurate, context-aware responses using semantic search and local LLMs.

---

##  Key Features

-  PDF text extraction and preprocessing  
-  Smart chunking with overlap  
-  Semantic search using Sentence Transformers  
-  Fast similarity search with FAISS  
-  Local LLM integration using Ollama  
-  Context-aware, relevant answers  

---

##  How It Works

1. The PDF is parsed and converted into text  
2. Text is split into overlapping chunks  
3. Each chunk is converted into embeddings  
4. FAISS indexes the embeddings for fast retrieval  
5. User query is embedded and matched with relevant chunks  
6. Retrieved context is passed to the LLM  
7. LLM generates a final answer  

---

##  Tech Stack

- **Python**
- **Sentence Transformers**
- **FAISS**
- **Ollama (Local LLM)**
- **NumPy**

---

##  Project Structure

-ai-document-assistant/
-│── app.py
-│── utils.py
-│── requirements.txt
-│── data/
-│ └── sample.pdf

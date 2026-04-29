"""
Utility functions for PDF processing, chunking,
hybrid vector/keyword storage, and retrieval.
"""
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# Load embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_and_chunk_pdf(file_obj, chunk_size=800, overlap=100):
    reader = PdfReader(file_obj)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    document_chunks = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            chunks = splitter.split_text(text)
            for chunk in chunks:
                document_chunks.append({
                    "text": chunk,
                    "page": page_num + 1 
                })
                
    return document_chunks

def create_hybrid_store(document_chunks):
    """
    Creates BOTH a FAISS vector index and a BM25 keyword index.
    """
    texts = [chunk["text"] for chunk in document_chunks]
    
    # 1. FAISS Index (Semantic)
    embeddings = embedding_model.encode(texts)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings))
    
    # 2. BM25 Index (Keyword)
    # Basic tokenization: lowercase and split by spaces
    tokenized_corpus = [doc.lower().split() for doc in texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    return faiss_index, bm25_index, embeddings

def hybrid_search(query, document_chunks, faiss_index, bm25_index, top_k=3):
    """
    Combines FAISS and BM25 results using Reciprocal Rank Fusion (RRF).
    """
    # We retrieve extra chunks initially to allow for better fusion sorting
    retrieve_k = top_k * 2 
    
    # --- 1. FAISS Search ---
    query_embedding = embedding_model.encode([query])
    _, faiss_indices = faiss_index.search(np.array(query_embedding), retrieve_k)
    faiss_indices = faiss_indices[0].tolist()
    
    # --- 2. BM25 Search ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    # Get indices of highest scores
    bm25_indices = np.argsort(bm25_scores)[::-1][:retrieve_k].tolist()
    
    # --- 3. Reciprocal Rank Fusion (RRF) ---
    rrf_scores = {}
    k_rrf = 60 # Standard smoothing constant for RRF
    
    # Score FAISS results
    for rank, doc_idx in enumerate(faiss_indices):
        if doc_idx not in rrf_scores:
            rrf_scores[doc_idx] = 0
        rrf_scores[doc_idx] += 1 / (k_rrf + rank + 1)
        
    # Score BM25 results
    for rank, doc_idx in enumerate(bm25_indices):
        if doc_idx not in rrf_scores:
            rrf_scores[doc_idx] = 0
        rrf_scores[doc_idx] += 1 / (k_rrf + rank + 1)
        
    # Sort the chunks by their fused RRF score
    sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    
    # Return the top_k absolute best chunks
    results = [document_chunks[i] for i in sorted_indices[:top_k] if i < len(document_chunks)]
    return results
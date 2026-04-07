from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model once (important for performance)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# Create Vector Store (FAISS)
def create_vector_store(chunks):
    embeddings = embedding_model.encode(chunks)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    index.add(np.array(embeddings))
    
    return index, embeddings


#  Semantic Search
def semantic_search(query, chunks, index, top_k=3):
    query_embedding = embedding_model.encode([query])
    
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    results = [chunks[i] for i in indices[0]]
    return results
"""
Automated Evaluation Pipeline using RAGAS.
Tests the Hybrid Search RAG architecture for Faithfulness and Answer Relevancy.
"""

import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Import our custom backend engine
from utils import load_and_chunk_pdf, create_hybrid_store, hybrid_search
import ollama

def generate_test_data(pdf_path, questions):
    print("Loading Document and Building Hybrid Index...")
    chunks = load_and_chunk_pdf(pdf_path)
    faiss_index, bm25_index, _ = create_hybrid_store(chunks)
    
    # UPDATED: Using the strict column names required by new RAGAS versions
    data = {"user_input": [], "response": [], "retrieved_contexts": []}
    
    for q in questions:
        print(f"Testing Query: {q}")
        
        # 1. Retrieve Context
        relevant_chunks = hybrid_search(q, chunks, faiss_index, bm25_index)
        context_texts = [chunk["text"] for chunk in relevant_chunks]
        context_str = "\n".join(context_texts)
        
        # 2. Generate Answer (Using your fast Phi-3 model)
        system_prompt = f"Answer the user's question using ONLY the context provided below.\n\nCONTEXT:\n{context_str}"
        
        response = ollama.chat(
            model="phi3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q}
            ]
        )
        answer = response["message"]["content"]
        
        # 3. Store in RAGAS format
        data["user_input"].append(q)
        data["response"].append(answer)
        data["retrieved_contexts"].append(context_texts) 
        
    return Dataset.from_dict(data)

def main():
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample.pdf")
    
    test_questions = [
        "What is the exact submission date and location, and what is the penalty for late submissions?",
        "What are the specific instructions for writing and submitting the assignment?",
        "According to Assignment 1, what exact fields need to be included when designing the feedback form?"
    ]
    
    print("--- Starting Pipeline Generation ---")
    dataset = generate_test_data(pdf_path, test_questions)
    
    print("\n--- Starting RAGAS Evaluation ---")
    # UPDATED: Using llama3 as the judge because it follows JSON formatting instructions better than phi3
    evaluator_llm = ChatOllama(model="llama3", format="json") 
    evaluator_embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    df = result.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    
    print("\n=== FINAL EVALUATION SCORES ===")
    display_cols = [col for col in ['user_input', 'faithfulness', 'answer_relevancy'] if col in df.columns]
    print(df[display_cols])
    
    print(f"\nResults successfully saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
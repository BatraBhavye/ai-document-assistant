import os
import ollama
from utils import load_pdf, chunk_text, create_vector_store, semantic_search


def get_pdf_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data", "sample.pdf")


def build_context(chunks):
    """Limit context size to avoid model crashes"""
    return " ".join(chunks[:2])[:1500]


def main():
    print("Loading document...")

    # Load PDF
    pdf_path = get_pdf_path()
    text = load_pdf(pdf_path)

    # Chunk text
    chunks = chunk_text(text)
    print(f"Chunks created: {len(chunks)}")

    # Create vector store
    index, _ = create_vector_store(chunks)
    print("Vector store ready.")

    # User input
    question = input("\nAsk a question about the document: ")

    # Retrieve relevant chunks
    relevant_chunks = semantic_search(question, chunks, index)

    # Build context
    context = build_context(relevant_chunks)

    # Generate response
    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "user",
                "content": f"""
You are an AI assistant. Answer based only on the given context.

Context:
{context}

Question:
{question}

Answer:
"""
            }
        ]
    )

    print("\nAnswer:\n")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
import streamlit as st
import ollama
from utils import load_and_chunk_pdf, create_hybrid_store, hybrid_search

# -- Page Configuration --
st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="centered")

st.title("📄 AI Document Assistant")
st.markdown("Upload a secure PDF and ask questions. Powered by **Local RAG (Ollama + FAISS + BM25)**.")

# -- Session State Management --
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

# -- Sidebar: File Upload & Processing --
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Parsing and building hybrid indexes..."):
            chunks = load_and_chunk_pdf(uploaded_file)
            st.session_state.chunks = chunks
            
            # Create BOTH Vector and Keyword Stores
            f_index, b_index, _ = create_hybrid_store(chunks)
            st.session_state.faiss_index = f_index
            st.session_state.bm25_index = b_index
            
            st.success(f"Processed successfully! Created {len(chunks)} hybrid-searchable chunks.")

# -- Main Chat Interface --
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    
    if st.session_state.faiss_index is None:
        st.warning("Please upload and process a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve context using Hybrid Search
        with st.spinner("Executing Hybrid Search (Vector + Keyword)..."):
            relevant_chunks = hybrid_search(
                prompt, 
                st.session_state.chunks, 
                st.session_state.faiss_index, 
                st.session_state.bm25_index
            )
            
            context = ""
            for i, chunk in enumerate(relevant_chunks):
                context += f"--- Source {i+1} (Page {chunk['page']}) ---\n{chunk['text']}\n\n"

        # Generate response
        system_prompt = f"""
        You are a highly accurate enterprise AI assistant. 
        Answer the user's question using ONLY the context provided below. 
        
        CRITICAL INSTRUCTION: You MUST cite your sources in your answer using the page numbers provided in the context. 
        Format your citations like this: "According to the document (Page X)..." or place [Page X] at the end of the sentence.
        
        If the answer is not contained in the context, say "I cannot find the answer in the provided document." 
        Do not hallucinate.

        CONTEXT:
        {context}
        """

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                response = ollama.chat(
                    model="phi3",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response["message"]["content"]
                st.markdown(answer)
                
                with st.expander("View Retrieved Source Context"):
                    st.write(context)

        st.session_state.messages.append({"role": "assistant", "content": answer})
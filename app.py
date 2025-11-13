import streamlit as st
from dotenv import load_dotenv
from rag_utils import extract_text_from_pdf, chunk_text
from qdrant_setup import get_qdrant_store, create_fresh_vector_store
from langgraph_agent import build_rag_agent
import os

load_dotenv()
st.set_page_config(page_title="🧠 LangGraph PDF Chat")

st.title("🧠 LangGraph-powered PDF Chat")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent" not in st.session_state:
    st.session_state.agent = build_rag_agent()
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# Show current PDF status
if st.session_state.current_pdf:
    st.info(f"📄 Currently loaded: {st.session_state.current_pdf}")
else:
    st.info("📄 No PDF loaded yet. Upload a PDF and click 'Load into Vector DB' to get started.")

if pdf_file and st.button("🔁 Load into Vector DB"):
    with st.spinner("Extracting & indexing..."):
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(text)
        
        # Create a fresh vector store for the new PDF (replaces any existing content)
        vector_store = create_fresh_vector_store()
        vector_store.add_texts(chunks)
        
        # Rebuild the agent with the new vector store
        st.session_state.agent = build_rag_agent()
        
        # Clear previous chat history since we have new content
        st.session_state.chat_history = []
        
        # Update current PDF tracker
        st.session_state.current_pdf = pdf_file.name
        
        st.success(f"📚 PDF content embedded and indexed! Added {len(chunks)} text chunks.")
        st.info("💡 You can now ask questions about the PDF content!")
        st.info(f"📄 Loaded content from: {pdf_file.name}")
        st.rerun()  # Refresh the UI to show the updated status

query = st.text_input("Ask something from the PDF")

if query:
    agent = st.session_state.agent
    result = agent.invoke({"question": query})
    response = result.get("response", "❌ No response")
    st.session_state.chat_history.append((query, response))

for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")

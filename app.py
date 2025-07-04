import streamlit as st
from dotenv import load_dotenv
from rag_utils import extract_text_from_pdf, chunk_text
from qdrant_setup import get_qdrant_store
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

if pdf_file and st.button("🔁 Load into Vector DB"):
    with st.spinner("Extracting & indexing..."):
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(text)
        qdrant = get_qdrant_store()
        qdrant.add_texts(chunks)
        st.success("📚 PDF content embedded and indexed in Qdrant!")

query = st.text_input("Ask something from the PDF")

if query:
    agent = st.session_state.agent
    result = agent.invoke({"question": query})
    response = result.get("response", "❌ No response")
    st.session_state.chat_history.append((query, response))

for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")

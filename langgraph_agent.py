from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from qdrant_setup import get_qdrant_store
import os
from typing import TypedDict

class GraphState(TypedDict):
    question: str
    response: str

def build_rag_agent():
    retriever = get_qdrant_store().as_retriever()
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    def query_node(state):
        question = state["question"]
        result = qa_chain.run(question)
        return {"response": result}

    builder = StateGraph(GraphState)  # ✅ FIXED
    builder.add_node("rag_query", query_node)
    builder.set_entry_point("rag_query")
    builder.set_finish_point("rag_query")

    app = builder.compile()
    return app

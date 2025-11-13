from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from qdrant_setup import get_qdrant_store
import os
from typing import TypedDict

class GraphState(TypedDict):
    question: str
    response: str

def build_rag_agent():
    retriever = get_qdrant_store().as_retriever()
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a simple RAG prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the following context:
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """)
    
    # Create a chain with prompt, llm, and output parser
    output_parser = StrOutputParser()

    def query_node(state):
        question = state["question"]
        
        # Retrieve relevant documents using the correct method
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Use the chain to get response
        chain = prompt | llm | output_parser
        result = chain.invoke({"context": context, "question": question})
        
        return {"response": result}

    builder = StateGraph(GraphState) 
    builder.add_node("rag_query", query_node)
    builder.set_entry_point("rag_query")
    builder.set_finish_point("rag_query")

    app = builder.compile()
    return app

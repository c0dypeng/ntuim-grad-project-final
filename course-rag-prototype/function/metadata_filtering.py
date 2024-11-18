from CourseSearch import CourseSearch
# dependencies for streamlit and langchain
import streamlit as st
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate

# dependencies for system
import asyncio


# ********** FILTER PROMPT SETUP **********
system = """You are an expert at converting user questions into database queries. \
You have access to a database of courses that are offered in NTU. \
Given a question, return a database query optimized to retrieve the most relevant results. \
If you decide to not return anything for "text", make "text" the original prompt. \
If there are acronyms or words you are not familiar with, do not try to rephrase them."""
filter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# ********** FILTER PROMPT SETUP **********




async def get_answer_multilingual_e5_metadataFiltering(llm, k, query: str) -> str:
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    pc = Pinecone()
    index_name = "ntuim-course"
    embedded_query = embeddings.embed_query(query)
    structured_llm = llm.with_structured_output(CourseSearch)
    query_analyzer = filter_prompt | structured_llm
    result = await query_analyzer.ainvoke({"question": query})
    filter = result.getFilter() if result else None
    vectorstore = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=k, filter=filter)
    if not docs:
        docs = []
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = await asyncio.to_thread(chain.run, input_documents=docs, question=query)
    return answer

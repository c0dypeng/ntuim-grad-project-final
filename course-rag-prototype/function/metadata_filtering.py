from CourseSearch import CourseSearch
# dependencies for streamlit and langchain
import streamlit as st
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings

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




async def get_answer_text_embedding_3_large_metadataFiltering(llm, k, query: str) -> str:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
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

    # data filtering
    # make shorten the page content to 700 characters
    for doc in docs:
        doc.page_content = doc.page_content[:700]
    # make embedding_text in matadata to be empty
    for doc in docs:
        doc.metadata["embedding_text"] = ""
    # make text in matadata to be empty
    for doc in docs:
        doc.metadata["text"] = ""

    if not docs:
        docs = []
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = await asyncio.to_thread(chain.run, input_documents=docs, question=query)
    return answer

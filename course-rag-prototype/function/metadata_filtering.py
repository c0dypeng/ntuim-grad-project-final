from CourseSearch import CourseSearch
# dependencies for streamlit and langchain
import streamlit as st
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain.schema import Document
import pandas as pd
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

def prepare_documents_with_separation(docs):
    prepared_docs = []
    for i, doc in enumerate(docs, 1):
        metadata_str = ', '.join(f"{key}: {value}" for key, value in doc.metadata.items())
        formatted_content = (
            f"[Document {i}]\n"
            f"Metadata: {metadata_str}\n"
            f"Content:\n{doc.page_content}\n"
        )
        prepared_docs.append(Document(page_content=formatted_content, metadata=doc.metadata))
    return prepared_docs



async def get_answer_multilingual_e5_metadataFiltering_reordering(llm, k, prompt,  query: str) -> str:
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    pc = Pinecone()
    index_name = "ntuim-course"

    structured_llm = llm.with_structured_output(CourseSearch)
    query_analyzer = filter_prompt | structured_llm
    result = await query_analyzer.ainvoke({"question": query})
    filter = result.getFilter() if result else None
    if(filter):
        df = pd.DataFrame.from_dict(filter, orient='index')
        # If you want to reset the index to have a default integer index
        df.reset_index(inplace=True)

        # Rename the columns if necessary
        df.columns = ['元資料', '篩選條件']
        
        st.dataframe(df)
    else:
        st.write("No filter returned")

    vectorstore = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=k, filter=filter)
    if not docs:
        answer = "不好意思，根據您的篩選條件，我們找不到符合的課程。有可能是我們沒有正確辨識出您的要求，您可以修改字句後再問一次；也有可能是資料庫中沒有符合您需求的課程。"
        return answer
    reordering = LongContextReorder()
    docs_reordered = reordering.transform_documents(docs)
    docs_reordered = prepare_documents_with_separation(docs_reordered)

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    answer = await asyncio.to_thread(chain.run, input_documents=docs_reordered, query=query)
    
    return answer

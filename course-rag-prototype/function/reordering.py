# custom code for the streamlit app
from taide_chat import taide_llm
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
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings



# dependencies for system
import asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

stuff_prompt_override = """你是一個了解台大課程的人，請謹慎、有禮貌但親切地給予協助，這對使用者而言非常重要。以下是系統找到的相關資訊:
-----
{context}
-----
請根據系統提供的資訊回答以下問題，請你以「以下是我找到的資訊」開頭。若以上資訊與問題無關，請忽略以上資訊:
{query}"""
prompt_template = PromptTemplate(
    template=stuff_prompt_override, input_variables=["context", "query"]
)
# ********** PROMPT SETUP **********

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

"""
Add reordering, promptTemplate, and prepare_documents_with_separation
"""
async def get_answer_text_embedding_3_large_reordering(llm, k, query: str) -> str:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
    index_name = "ntuim-course"
    vectorstore = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=k)

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

    reordering = LongContextReorder()
    docs_reordered = reordering.transform_documents(docs)
    docs_reordered = prepare_documents_with_separation(docs_reordered)

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)
    answer = await asyncio.to_thread(chain.run, input_documents=docs_reordered, query=query)
    return answer
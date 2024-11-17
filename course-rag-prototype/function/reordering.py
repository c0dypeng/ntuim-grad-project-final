# dependencies for streamlit and langchain
import streamlit as st
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain.schema import Document


# dependencies for system
import asyncio


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
async def get_answer_multilingual_e5_reordering(llm, k, prompt, query: str) -> str:
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    index_name = "ntuim-course"
    vectorstore = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=k)
    print("docs: ", docs)
    reordering = LongContextReorder()
    docs_reordered = reordering.transform_documents(docs)
    print("docs_reordered: ", docs_reordered)
    docs_reordered = prepare_documents_with_separation(docs_reordered)

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    answer = await asyncio.to_thread(chain.run, input_documents=docs_reordered, query=query)
    return answer
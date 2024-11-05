# dependencies for streamlit and langchain
import streamlit as st

# dependencies for system
import asyncio

from taide_chat import taide_llm

from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain.schema import Document

from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

k = 5

llm = taide_llm # change this use different LLM provider

stuff_prompt_override = """你是一個了解台大課程的人，請謹慎、有禮貌但親切地給予協助，這對使用者而言非常重要。以下是系統找到的相關資訊:
-----
{context}
-----
請根據系統提供的資訊回答以下問題，請你以「以下是我找到的資訊」開頭。若以上資訊與問題無關，請忽略以上資訊:
{query}"""
prompt_template = PromptTemplate(
    template=stuff_prompt_override, input_variables=["context", "query"]
)

chain_rag = load_qa_chain(llm, chain_type="stuff")
chain_rag_reordering = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)
chain_normal = load_qa_chain(llm)

long_context_reorder = LongContextReorder()

# ********** FUNCTIONS **********

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


async def get_answer_multilingual_e5(query: str) -> str:
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    index_name = "ntuim-course"
    vectorstore = PineconeVectorStore(
    index_name=index_name, embedding=embeddings)
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=k)
    answer = await asyncio.to_thread(chain_rag.run, input_documents=docs, question=query)
    return answer

"""
Add reordering, promptTemplate, and prepare_documents_with_separation
"""
async def get_answer_multilingual_e5_reordering(query: str) -> str:
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    index_name = "ntuim-course"
    vectorstore = PineconeVectorStore(
    index_name=index_name, embedding=embeddings)
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=k)
    docs_reordered = long_context_reorder.transform_documents(docs)
    docs_reordered = prepare_documents_with_separation(docs_reordered)

    answer = await asyncio.to_thread(chain_rag_reordering.run, input_documents=docs_reordered, query=query)
    return answer

async def get_answer_without_rag(query: str) -> str:
    answer = await asyncio.to_thread(chain_normal.run, input_documents=[], question=query)
    return answer

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

async def main(query: str):
    answer_multilingual_e5 = await get_answer_multilingual_e5(query)
    answer_multilingual_e5_reordering = await get_answer_multilingual_e5_reordering(query)
    answer_without_rag = await get_answer_without_rag(query)

    st.session_state.chat_history.append({
        "query": query,
        "answer_multilingual_e5": answer_multilingual_e5,
        "answer_multilingual_e5_reordering": answer_multilingual_e5_reordering,
        "answer_without_rag": answer_without_rag
    })

st.title('Query Answering Application')
query = st.chat_input('Enter your query:')
# add "query: " to the input query
# query = 'query: ' + query

if query:
    asyncio.run(main(query))

for entry in st.session_state.chat_history:
    st.chat_message("user").write(entry["query"])
    st.chat_message("assistant").write(f'**Answer (multilingual-e5-large)**: {entry["answer_multilingual_e5"]}')
    st.chat_message("assistant").write(f'**Answer (multilingual-e5-large with reordering)**: {entry["answer_multilingual_e5_reordering"]}')
    st.chat_message("assistant").write(f'**Answer (without RAG)**: {entry["answer_without_rag"]}')
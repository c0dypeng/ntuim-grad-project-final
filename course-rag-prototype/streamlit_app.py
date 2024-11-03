# custom code for the streamlit app
from taide_chat import taide_llm

# dependencies for streamlit and langchain
import streamlit as st
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain.schema import Document


# dependencies for system
import asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

k = 5

llm = taide_llm # change this use different LLM provider

# ********** PROMPT SETUP **********
stuff_prompt_override = """你是一個了解台大課程的人，請謹慎、有禮貌但親切地給予協助，這對使用者而言非常重要。以下是相關資訊:
-----
{context}
-----
請根據以上的資訊回答以下問題，並預設以上是你自己找到的資料:
{query}"""
prompt = PromptTemplate(
    template=stuff_prompt_override, input_variables=["context", "query"]
)
# ********** PROMPT SETUP **********


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
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = await asyncio.to_thread(chain.run, input_documents=docs, question=query)
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

    reordering = LongContextReorder()
    docs_reordered = reordering.transform_documents(docs)
    docs_reordered = prepare_documents_with_separation(docs_reordered)

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    answer = await asyncio.to_thread(chain.run, input_documents=docs_reordered, query=query)
    return answer

async def get_answer_without_rag(query: str) -> str:
    chain = load_qa_chain(llm)
    answer = await asyncio.to_thread(chain.run, input_documents=[], question=query)
    return answer

# ********** FUNCTIONS **********


async def main(query: str):
    answer_multilingual_e5 = await get_answer_multilingual_e5(query)
    answer_multilingual_e5_reordering = await get_answer_multilingual_e5_reordering(query)
    answer_without_rag = await get_answer_without_rag(query)
    
    st.write(f'**Answer (multilingual-e5-large)**: {answer_multilingual_e5}')
    st.write(f'**Answer (multilingual-e5-large with reordering)**: {answer_multilingual_e5_reordering}')
    st.write(f'**Answer (without RAG)**: {answer_without_rag}')

st.title('Query Answering Application')
query = st.text_input('Enter your query:')
# add "query: " to the input query
# query = 'query: ' + query

if st.button('Get Answer'):
    asyncio.run(main(query))



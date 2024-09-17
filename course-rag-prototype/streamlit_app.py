# custom code for the streamlit app
from taide_chat import taide_llm

# dependencies for streamlit and langchain
import streamlit as st
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.chains.question_answering import load_qa_chain

# dependencies for system
import asyncio
from dotenv import load_dotenv

load_dotenv()

k = 1

llm = taide_llm # change this use different LLM provider


async def get_answer_multilingual_e5(query: str) -> str:
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    index_name = "ntuim-course"
    vectorstore = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=k)
    chain = load_qa_chain(llm, chain_type="map_reduce")
    answer = await asyncio.to_thread(chain.run, input_documents=docs, question=query)
    return answer


async def get_answer_without_rag(query: str) -> str:
    chain = load_qa_chain(llm)
    answer = await asyncio.to_thread(chain.run, input_documents=[], question=query)
    return answer


async def main(query: str):
    answer_multilingual_e5 = await get_answer_multilingual_e5(query)
    answer_without_rag = await get_answer_without_rag(query)
    
    st.write(f'Answer (multilingual-e5-large): {answer_multilingual_e5}')
    st.write(f'Answer (without RAG): {answer_without_rag}')

st.title('Query Answering Application')
query = st.text_input('Enter your query:')
# add "query: " to the input query
# query = 'query: ' + query

if st.button('Get Answer'):
    asyncio.run(main(query))



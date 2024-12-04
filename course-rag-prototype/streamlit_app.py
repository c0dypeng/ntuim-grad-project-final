# custom code for the streamlit app
from taide_chat import taide_llm
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI 

import streamlit as st
from langchain.chains.question_answering import load_qa_chain

# dependencies for system
import asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

from langchain_openai import OpenAIEmbeddings

# import functions
from function.reordering import get_answer_reordering
from function.simple_rag import get_answer_simple_rag
from function.metadata_filtering import get_answer_metadataFiltering
from function.only_llm import get_answer_without_rag
from function.simple_rag_agent import get_answer_simple_rag_agent
import os
load_dotenv()

k = 5

# llm = taide_llm # change this use different LLM provider
# llm = OpenAI()

llm = ChatOpenAI(model_name="gpt-4")

chat_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = chat_history

async def main(query: str):
    for entry in st.session_state.chat_history:
        st.chat_message("user").markdown(entry["query"])
        st.chat_message("assistant").markdown(f'**Assistant**: {entry["answer"]}')

    st.chat_message("user").markdown(query)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    # answer_simple_rag = await get_answer_simple_rag(embeddings, llm, k, query)
    # answer_reordering = await get_answer_reordering(embeddings, llm, k, query)
    # answer_metadataFiltering = await get_answer_metadataFiltering(embeddings, llm, k, query)
    # answer_without_rag = await get_answer_without_rag(llm, query)
    with st.spinner("Loading..."):
        answer = await get_answer_simple_rag_agent(embeddings, llm, k, query)
    st.chat_message("assistant").markdown(f'**Assistant**: {answer}')

    st.session_state.chat_history.append({
        "query": query,
        # "answer_text_embedding_3_large": answer_simple_rag,
        # "answer_text_embedding_3_large_reordering": answer_reordering,
        # "answer_text_embedding_3_large_metadataFiltering": answer_metadataFiltering,
        "answer": answer,
    })


st.title(body=':blue[NTU Course Search]', anchor=False)
st.caption('歡迎使用台大課程搜尋助手！有任何課程相關的問題嗎？')
st.caption('請輸入您的問題，我們將會為您提供相關的課程資訊。')
st.divider()
query = st.chat_input('Enter your query:')
# add "query: " to the input query
# query = 'query: ' + query

if query:
    asyncio.run(main(query))
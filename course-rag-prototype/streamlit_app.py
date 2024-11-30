# custom code for the streamlit app
from taide_chat import taide_llm
from langchain_openai import OpenAI

import streamlit as st
from langchain.chains.question_answering import load_qa_chain

# dependencies for system
import asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


# dependencies for system
import asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# import functions
from function.reordering import get_answer_multilingual_e5_reordering
from function.simple_rag import get_answer_multilingual_e5
from function.metadata_filtering import get_answer_multilingual_e5_metadataFiltering
from function.only_llm import get_answer_without_rag

from function.simple_rag_agent import get_answer_multilingual_e5_agent
load_dotenv()

k = 5

# llm = taide_llm # change this use different LLM provider
llm = OpenAI()

chat_history = []
st.session_state.chat_history = chat_history

async def main(query: str):
    answer_multilingual_e5 = await get_answer_multilingual_e5(llm, k, query)
    answer_multilingual_e5_reordering = await get_answer_multilingual_e5_reordering(llm, k, query)
    # answer_multilingual_e5_metadataFiltering = await get_answer_multilingual_e5_metadataFiltering(llm, k, query)
    answer_without_rag = await get_answer_without_rag(llm, query)

    answer_multilingual_e5_agent = await get_answer_multilingual_e5_agent(llm, k, query)

    chat_history.append({
        "query": query,
        "answer_multilingual_e5": answer_multilingual_e5,
        "answer_multilingual_e5_reordering": answer_multilingual_e5_reordering,
        # "answer_multilingual_e5_metadataFiltering": answer_multilingual_e5_metadataFiltering,
        "answer_without_rag": answer_without_rag,
        "answer_multilingual_e5_agent": answer_multilingual_e5_agent
    })
    
    # # legacy
    # st.write(f'**Answer (multilingual-e5-large)**: {answer_multilingual_e5}')
    # st.write(f'**Answer (multilingual-e5-large with reordering)**: {answer_multilingual_e5_reordering}')
    # st.write(f'**Answer (multilingual-e5-large with metadataFiltering)**: {answer_multilingual_e5_metadataFiltering}')
    # st.write(f'**Answer (without RAG)**: {answer_without_rag}')
    # st.write(f'**Answer (multilingual-e5-large agent)**: {answer_multilingual_e5_agent}')

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
    # st.chat_message("assistant").write(f'**Answer (multilingual-e5-large with metadataFiltering)**: {entry["answer_multilingual_e5_metadataFiltering"]}')
    st.chat_message("assistant").write(f'**Answer (without RAG)**: {entry["answer_without_rag"]}')
    st.chat_message("assistant").write(f'**Answer (multilingual-e5-large agent)**: {entry["answer_multilingual_e5_agent"]}')
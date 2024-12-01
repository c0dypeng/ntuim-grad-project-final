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
from function.reordering import get_answer_text_embedding_3_large_reordering
from function.simple_rag import get_answer_text_embedding_3_large
from function.metadata_filtering import get_answer_text_embedding_3_large_metadataFiltering
from function.only_llm import get_answer_without_rag

from function.simple_rag_agent import get_answer_text_embedding_3_large_agent
load_dotenv()


k = 5

# llm = taide_llm # change this use different LLM provider
llm = OpenAI()

chat_history = []
st.session_state.chat_history = chat_history

async def main(query: str):
    answer_text_embedding_3_large = await get_answer_text_embedding_3_large(llm, k, query)
    answer_text_embedding_3_large_reordering = await get_answer_text_embedding_3_large_reordering(llm, k, query)
    # answer_text_embedding_3_large_metadataFiltering = await get_answer_text_embedding_3_large_metadataFiltering(llm, k, query)
    answer_without_rag = await get_answer_without_rag(llm, query)
    # answer_text_embedding_3_large_agent = await get_answer_text_embedding_3_large_agent(llm, k, query)

    chat_history.append({
        "query": query,
        "answer_text_embedding_3_large": answer_text_embedding_3_large,
        "answer_text_embedding_3_large_reordering": answer_text_embedding_3_large_reordering,
        # "answer_text_embedding_3_large_metadataFiltering": answer_text_embedding_3_large_metadataFiltering,
        "answer_without_rag": answer_without_rag,
        # "answer_text_embedding_3_large_agent": answer_text_embedding_3_large_agent
    })

st.title('NTU Course Search')
query = st.chat_input('Enter your query:')
# add "query: " to the input query
# query = 'query: ' + query

if query:
    asyncio.run(main(query))

for entry in st.session_state.chat_history:
    st.chat_message("user").write(entry["query"])
    st.chat_message("assistant").write(f'**Answer (text_embedding_3_large)**: {entry["answer_text_embedding_3_large"]}')
    st.chat_message("assistant").write(f'**Answer (text_embedding_3_large with reordering)**: {entry["answer_text_embedding_3_large_reordering"]}')
    # st.chat_message("assistant").write(f'**Answer (text_embedding_3_large with metadataFiltering)**: {entry["answer_text_embedding_3_large_metadataFiltering"]}')
    st.chat_message("assistant").write(f'**Answer (without RAG)**: {entry["answer_without_rag"]}')
    # st.chat_message("assistant").write(f'**Answer (text_embedding_3_large agent)**: {entry["answer_text_embedding_3_large_agent"]}')
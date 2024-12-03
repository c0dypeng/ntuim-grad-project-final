# custom code for the streamlit app
from taide_chat import taide_llm
from langchain_openai import OpenAI

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
load_dotenv()


k = 5

# llm = taide_llm # change this use different LLM provider
llm = OpenAI()

chat_history = []
st.session_state.chat_history = chat_history

async def main(query: str):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
    answer_simple_rag = await get_answer_simple_rag(embeddings, llm, k, query)
    answer_reordering = await get_answer_reordering(embeddings, llm, k, query)
    # answer_metadataFiltering = await get_answer_metadataFiltering(embeddings, llm, k, query)
    answer_without_rag = await get_answer_without_rag(llm, query)

    chat_history.append({
        "query": query,
        "answer_text_embedding_3_large": answer_simple_rag,
        "answer_text_embedding_3_large_reordering": answer_reordering,
        # "answer_text_embedding_3_large_metadataFiltering": answer_metadataFiltering,
        "answer_without_rag": answer_without_rag,
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
# custom code for the streamlit app
# from taide_chat import taide_llm

import streamlit as st
from langchain.chains.question_answering import load_qa_chain

# dependencies for system
import asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# dependencies for system
import asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# import functions
from function.reordering import get_answer_multilingual_e5_reordering
from function.simple_rag import get_answer_multilingual_e5
from function.metadata_filtering import get_answer_multilingual_e5_metadataFiltering_reordering
from function.only_llm import get_answer_without_rag
load_dotenv()

k = 5

# llm = taide_llm # change this use different LLM provider
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

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


async def main(query: str):
    # answer_multilingual_e5 = await get_answer_multilingual_e5(llm, k, query)
    # answer_multilingual_e5_reordering = await get_answer_multilingual_e5_reordering(llm, k, prompt, query)
    answer_multilingual_e5_metadataFiltering = await get_answer_multilingual_e5_metadataFiltering_reordering(llm, k, prompt, query)
    # answer_without_rag = await get_answer_without_rag(llm, query)
    
    # st.write(f'**Answer (multilingual-e5-large)**: {answer_multilingual_e5}')
    # st.write(f'**Answer (multilingual-e5-large with reordering)**: {answer_multilingual_e5_reordering}')
    st.write(f'**Answer (multilingual-e5-large with metadataFiltering)**: {answer_multilingual_e5_metadataFiltering}')
    # st.write(f'**Answer (without RAG)**: {answer_without_rag}')

st.title('Query Answering Application')
query = st.text_input('Enter your query:')
# add "query: " to the input query
# query = 'query: ' + query

if st.button('Get Answer'):
    asyncio.run(main(query))



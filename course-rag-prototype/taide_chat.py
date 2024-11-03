from dotenv import load_dotenv
import os
import requests
host = "https://td.nchc.org.tw/api/v1"

from langchain_openai import ChatOpenAI


load_dotenv()  # take environment variables from .env.


def get_token():
    username = os.getenv("TAIDE_EMAIL")
    password = os.getenv("TAIDE_PASSWORD")
    r = requests.post(
        host+"/token", data={"username": username, "password": password})
    token = r.json()["access_token"]
    return token


taide_llm = ChatOpenAI(
    model="TAIDE/a.2.0.0",
    temperature=0,
    max_tokens=4000,
    timeout=None,
    max_retries=2,
    openai_api_base="https://td.nchc.org.tw/api/v1/",
    openai_api_key=get_token(),    
)

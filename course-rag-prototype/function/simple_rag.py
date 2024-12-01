from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings


# dependencies for system
import asyncio


async def get_answer_text_embedding_3_large(llm, k, query: str) -> str:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
    index_name = "ntuim-course"
    vectorstore = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)
    docs = await asyncio.to_thread(vectorstore.similarity_search, query=query, k=k)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = await asyncio.to_thread(chain.run, input_documents=docs, question=query)
    return answer
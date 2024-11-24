from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.agents import initialize_agent
from langchain.tools import BaseTool
import asyncio
from typing import Any

async def get_answer_multilingual_e5_agent(llm, k, query: str) -> str:
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    index_name = "ntuim-course"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    class VectorStoreSearchTool(BaseTool):
        name: str = "VectorStoreSearch"
        description: str = "search similar documents in vector store"
        vectorstore: Any
        k: int

        def _run(self, query: str) -> str:
            docs = self.vectorstore.similarity_search(query=query, k=self.k)
            return "\n\n".join([doc.page_content for doc in docs])
        
        async def _arun(self, query: str) -> str:
            docs = await asyncio.to_thread(
                self.vectorstore.similarity_search, query=query, k=self.k
            )
            return "\n\n".join([doc.page_content for doc in docs])
    
    search_tool = VectorStoreSearchTool(vectorstore=vectorstore, k=k)
    tools = [search_tool]
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=False
    )
    
    answer = await asyncio.to_thread(agent.run, query)
    
    return answer

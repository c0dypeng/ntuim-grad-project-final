from langchain_pinecone import PineconeVectorStore
from langchain.agents import initialize_agent
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import create_react_agent, AgentExecutor
import asyncio
from typing import Any
from get_template import get_language_specific_template

class VectorStoreSearchTool(BaseTool):
    name: str = "NtuCourseSearch"
    description: str = "搜尋台大課程"
    vectorstore: Any
    k: int

    def _run(self, query: str) -> str:
        docs = self.vectorstore.similarity_search(query=query, k=self.k)
        return "\n\n".join([doc.page_content for doc in docs])
    
    async def _arun(self, query: str) -> str:
        docs = await asyncio.to_thread(
            self.vectorstore.similarity_search, query=query, k=self.k
        )

        # data filtering
        # make shorten the page content to 700 characters
        for doc in docs:
            doc.page_content = doc.page_content[:700]
        # make embedding_text in matadata to be empty
        for doc in docs:
            doc.metadata["embedding_text"] = ""
        # make text in matadata to be empty
        for doc in docs:
            doc.metadata["text"] = ""

        return "\n\n".join([doc.page_content for doc in docs])

async def get_answer_simple_rag_agent(embedding, llm, k, query: str) -> str:    
    embeddings = embedding
    index_name = "ntuim-course"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    
    search_tool = VectorStoreSearchTool(vectorstore=vectorstore, k=k)
    tools = [search_tool]
    template = get_language_specific_template(query)

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=2)

    answer = await asyncio.to_thread(agent_executor.invoke, {'input': query})
    print(answer)
    return answer["output"]
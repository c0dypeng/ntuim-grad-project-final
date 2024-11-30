from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain.agents import initialize_agent
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import create_react_agent, AgentExecutor
import asyncio
from typing import Any

TEMPLATE = """你是一個了解台大課程的人，請謹慎、有禮貌但親切地給予協助，這對使用者而言非常重要。

{tools}

To use a tool, please use the following format:
'''
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
'''

當你收集到所有信息後，請根據訊息給予使用者課程選擇上的建議，或者回答使用者的問題。

'''
Thought: Do I need to use a tool? No
Final Answer: [The AI's response]
'''

Begin!

Instructions: {input}
{agent_scratchpad}
"""

async def get_answer_multilingual_e5_agent(llm, k, query: str) -> str:    
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    index_name = "ntuim-course"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    class VectorStoreSearchTool(BaseTool):
        name: str = "NtuCourseSearch"
        description: str = "search similar documents about course info in vector store"
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

    prompt = PromptTemplate.from_template(TEMPLATE)
    agent = create_react_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    answer = await asyncio.to_thread(agent_executor.invoke, {'input': query})
    print(answer)
    return answer
from langchain_pinecone import PineconeVectorStore
# from langchain.agents import initialize_agent
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.document_transformers import LongContextReorder
import asyncio
from CourseSearch import CourseSearch
from typing import Any
from .get_template import get_language_specific_template
from .metadata_filtering import get_metadata_filter
from langchain.schema import Document

def prepare_documents_with_separation(docs):
    prepared_docs = []
    printOnce = False
    for i, doc in enumerate(docs, 1):
        metadata_str = ', '.join(f"{key}: {value}" for key, value in doc.metadata.items())
        formatted_content = (
            f"[Document {i}]\n"
            f"Metadata: {metadata_str}\n"
            f"Content:\n{doc.page_content}\n"
        )[:700]
        if not printOnce:
            print(formatted_content)
            printOnce = True
        prepared_docs.append(Document(page_content=formatted_content, metadata=doc.metadata))
    return prepared_docs

class VectorStoreSearchTool(BaseTool):
    name: str = "NtuCourseSearch"
    description: str = "搜尋台大課程"
    vectorstore: PineconeVectorStore
    k: int
    filter: dict

    def _run(self, query: str) -> str:
        # try:
        #     docs = self.vectorstore.similarity_search(query=query, k=self.k, filter=self.filter)
        # except:
        docs = self.vectorstore.similarity_search(query=query, k=self.k)
        reordering = LongContextReorder()
        docs = reordering.transform_documents(docs)

        # data filtering
        # make embedding_text in matadata to be empty
        for doc in docs:
            doc.metadata["embedding_text"] = ""
        # make text in matadata to be empty
        for doc in docs:
            doc.metadata["text"] = ""
        
        docs = prepare_documents_with_separation(docs)
        return "\n\n".join([doc.page_content for doc in docs])
    
    async def _arun(self, query: str) -> str:
        docs = self.vectorstore.similarity_search(query=query, k=self.k)
        reordering = LongContextReorder()
        docs = reordering.transform_documents(docs)

        # data filtering
        # make embedding_text in matadata to be empty
        for doc in docs:
            doc.metadata["embedding_text"] = ""
        # make text in matadata to be empty
        for doc in docs:
            doc.metadata["text"] = ""
        
        docs = prepare_documents_with_separation(docs)
        return "\n\n".join([doc.page_content for doc in docs])

async def get_answer_simple_rag_agent(embedding, llm, k, query: str) -> str:    
    embeddings = embedding
    index_name = "ntuim-course"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    filter = await get_metadata_filter(llm, query)
    print(filter)

    search_tool = VectorStoreSearchTool(vectorstore=vectorstore, k=k, filter=filter)
    tools = [search_tool]
    template = get_language_specific_template(query)

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=2)

    answer = await asyncio.to_thread(agent_executor.invoke, {'input': query})
    print(answer)
    return answer["output"]
from langchain.chains.question_answering import load_qa_chain

# dependencies for system
import asyncio


async def get_answer_without_rag(llm, query: str) -> str:
    chain = load_qa_chain(llm)
    answer = await asyncio.to_thread(chain.run, input_documents=[], question=query)
    return answer

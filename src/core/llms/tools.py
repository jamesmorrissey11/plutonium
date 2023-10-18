import openai
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.vectorstores import DeepLake


def create_web_search_tool(
    serper_api_key: str, tool_name: str, tool_description: str
) -> Tool:
    web_search = GoogleSerperAPIWrapper(
        serper_api_key=serper_api_key,
    )
    web_search_tool = Tool(
        name=tool_name,
        func=web_search.run,
        description=tool_description,
    )
    return web_search_tool


def create_vectorstore_search_tool(
    model_dir: str, tool_name: str, tool_description: str
) -> Tool:
    retrieval_llm = OpenAI(temperature=0)
    db = DeepLake(
        dataset_path=model_dir,
        embedding=OpenAIEmbeddings(),
        read_only=True,
    )
    vectorstore_retriever = RetrievalQA.from_chain_type(
        llm=retrieval_llm, retriever=db.as_retriever(), chain_type="stuff"
    )
    vectorstore_search_tool = Tool(
        name=tool_name,
        func=vectorstore_retriever.run,
        description=tool_description,
    )
    return vectorstore_search_tool

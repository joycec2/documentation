from llama_index.core import ListIndex, SimpleDirectoryReader, VectorStoreIndex, Settings
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama

from utils import extract_numbered_questions

class BaseAgent:
    def __init__(self, prompt_template: str, model):
        self.model = model
        self.template = prompt_template

    def query(self, query: str):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_model(self):
        return self.model
    
    def get_template(self):
        return self.template

class ReasoningAgent(BaseAgent):
    def __init__(self, model):
        super().__init__(model)

    # we now use langchain model
    def query(self, query: str):
        chain = self.template | self.model
        result = chain.invoke({"query_str": query})
        content = result.content
        questions = extract_numbered_questions(content)
        return questions

class RAGAgent(BaseAgent):
    def __init__(self, index: VectorStoreIndex, model: Ollama, vectors: ChromaVectorStore):
        super().__init__(model)
        self.vectors = vectors

    def query(self, query: str):
        response = self.index.query(query)
        return response

class Text2SQLAgent(BaseAgent):
    def __init__(self, index: VectorStoreIndex, model: Ollama, query_engine):
        super().__init__(model)
        self.query_engine = query_engine

    def query(self, query: str):
        sql_query = self._nl_to_sql(query)
        result = self.query_engine.execute(sql_query)
        return result

    def _nl_to_sql(self, natural_language: str) -> str:
        return f"SELECT * FROM data WHERE column = '{natural_language}'"

from llama_index.core import ListIndex, SimpleDirectoryReader, VectorStoreIndex, Settings
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb

class BaseAgent:
    def __init__(self, index: VectorStoreIndex, model: Ollama):
        self.model = model
        self.index = index

    def query(self, query: str):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_index(self):
        return self.index
    
    def get_model(self):
        return self.model

class ReasoningAgent(BaseAgent):
    def __init__(self, index: VectorStoreIndex, model: Ollama):
        super().__init__(index, model)

    def query(self, query: str):
        steps = self._decompose_query(query)
        responses = [self.index.query(step) for step in steps]
        return "\n".join(responses)

    def _decompose_query(self, query: str):
        return [query]

class RAGAgent(BaseAgent):
    def __init__(self, index: VectorStoreIndex, model: Ollama, vectors: ChromaVectorStore):
        super().__init__(index, model)
        self.vectors = vectors

    def query(self, query: str):
        response = self.index.query(query)
        return response

class Text2SQLAgent(BaseAgent):
    def __init__(self, index: VectorStoreIndex, model: Ollama, query_engine):
        super().__init__(index, model)
        self.query_engine = query_engine

    def query(self, query: str):
        sql_query = self._nl_to_sql(query)
        result = self.query_engine.execute(sql_query)
        return result

    def _nl_to_sql(self, natural_language: str) -> str:
        return f"SELECT * FROM data WHERE column = '{natural_language}'"

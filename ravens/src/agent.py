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

class RAGAgent(BaseAgent):
    def __init__(self, index: VectorStoreIndex, model: Ollama, vectors: ChromaVectorStore):
        super().__init__(index, model)
        self.index = index
        self.model = model
        self.vectors = vectors

    def query(self, query: str):
        response = self.index.query(query)
        return response




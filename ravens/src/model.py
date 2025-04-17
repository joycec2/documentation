from typing import List, Union
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:
    OpenAIEmbedding = None
    OllamaEmbedding = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class EmbeddingService:
    def __init__(self, provider: str = "sentence-transformers", model_name: str = "all-MiniLM-L6-v2"):
        self.provider = provider.lower()

        if self.provider == "sentence-transformers":
            if SentenceTransformer is None:
                raise ImportError("Please install sentence-transformers to use this provider.")
            self.model = SentenceTransformer(model_name)
        elif self.provider == "openai":
            if OpenAIEmbedding is None:
                raise ImportError("Please install llama-index with OpenAI support.")
            self.model = OpenAIEmbedding(model=model_name)
        elif self.provider== "ollama":
            if OllamaEmbedding is None:
                raise ImportError("Please install llama-index with Ollama support.")
            self.model = OllamaEmbedding(model_name=model_name)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]

        if self.provider == "sentence-transformers":
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        else:
            return self.model.get_text_embedding_batch(texts)


class VectorStore:
    def __init__(self, collection_name: str = "default", embedding_function=None):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    def add(self, ids: List[str], texts: List[str], metadatas: List[dict] = None):
        self.collection.add(documents=texts, ids=ids, metadatas=metadatas)

    def query(self, query_text: str, n_results: int = 5):
        return self.collection.query(query_texts=[query_text], n_results=n_results)
      
# Example usages: sentence-transformers, openai, and ollama

# embedder = EmbeddingService(provider="sentence-transformers")

# embedder = EmbeddingService(provider="openai", model_name="text-embedding-ada-002")

# embedder = EmbeddingService(provider="ollama", model_name="mistral")

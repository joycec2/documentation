"""
Provides wrapper classes for configuring Ollama language models using both 
LlamaIndex and LangChain interfaces.

Classes:
    - llamaindex_OllamaLLM: Configures and wraps LlamaIndex's Ollama integration.
    - langchain_OllamaLLM: Configures and wraps LangChain's Ollama integration.
"""

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from langchain_ollama import ChatOllama


class llamaindex_OllamaLLM:
    """
    Wrapper for configuring a LlamaIndex Ollama model.

    Args:
        model_name (str): The name of the model to use.
        temperature (float): Sampling temperature for the model.
        max_token (int): Maximum number of tokens to generate.
        request_timeout (float): Timeout for API requests.
    """

    def __init__(self, model_name: str, temperature: float = 0.2, max_token: int = 2048, request_timeout: float = 360.0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_token = max_token
        self.request_timeout = request_timeout
        self.llm = self.configure_llm(model_name, temperature, max_token, request_timeout)

    def configure_llm(self, model_name: str, temperature: float = 0.2, max_token: int = 2048, request_timeout: float = 360.0):
        """
        Configure the Ollama model using LlamaIndex.

        Returns:
            Ollama: Configured LlamaIndex Ollama model.
        """
        return Ollama(
            model=model_name,
            request_timeout=request_timeout,
            temperature=temperature,
            max_token=max_token
        )


class langchain_OllamaLLM:
    """
    Wrapper for configuring a LangChain Ollama model.

    Args:
        model_name (str): The name of the model to use.
        temperature (float): Sampling temperature for the model.
        request_timeout (float): Timeout for API requests.
    """

    def __init__(self, model_name: str, temperature: float = 0.2, request_timeout: float = 360.0):
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.llm = self.configure_llm(model_name, temperature, request_timeout)

    def configure_llm(self, model_name: str, temperature: float = 0.2, request_timeout: float = 360.0):
        """
        Configure the Ollama model using LangChain.

        Returns:
            ChatOllama: Configured LangChain Ollama model.
        """
        return ChatOllama(
            model=model_name,
            request_timeout=request_timeout,
            temperature=temperature
        )

    # Example preloaded model
    deepseek_32b = ChatOllama(model="deepseek-r1:32b", request_timeout=360.0, temperature=0.2)
    
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

class EmbeddingService:
    """
    Wrapper around OllamaEmbedding for generating vector embeddings from text.
    """
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.embedding = OllamaEmbedding(model_name=model_name)

    def get_text_embedding(self, text: str):
        """
        Generate a vector embedding for the given text.

        Args:
            text (str): Input text to embed.

        Returns:
            List[float]: Vector representation of the input text.
        """
        return self.embedding.get_text_embedding(text)


class VectorStore:
    """
    Wrapper around ChromaVectorStore for storing and querying embedded vectors.
    """
    def __init__(self, persist_directory: str = "./chroma_db"):
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.vector_store = ChromaVectorStore(chroma_client=chroma_client, collection_name="default")

    def add_documents(self, documents: list[dict]):
        """
        Add documents with text and metadata to the vector store.

        Args:
            documents (list[dict]): List of documents with 'text' and optional 'metadata'.
        """
        self.vector_store.add(documents)

    def query(self, text: str, embedding_service: EmbeddingService, top_k: int = 5):
        """
        Query the vector store with embedded text and return top-k similar documents.

        Args:
            text (str): The text to embed and search with.
            embedding_service (EmbeddingService): The embedding service instance.
            top_k (int): Number of top documents to retrieve.

        Returns:
            List[dict]: Matched documents with similarity scores.
        """
        embedding = embedding_service.get_text_embedding(text)
        return self.vector_store.query(embedding, top_k=top_k)

# example usage
# Instantiate models using the custom OllamaLLM wrapper
# deepseek_7b_q8 = llamaindex_OllamaLLM(model_name="deepseek-r1:7b-qwen-distill-q8_0", temperature=0.1)
# deepseek_7b   = llamaindex_OllamaLLM(model_name="deepseek-r1:7b-qwen-distill-fp16", temperature=0.1)
# deepseek_8b   = llamaindex_OllamaLLM(model_name="deepseek-r1:8b-llama-distill-fp16", temperature=0.2)
# deepseek_14b  = llamaindex_OllamaLLM(model_name="deepseek-r1:14b-qwen-distill-fp16", temperature=0.1)
# deepseek_32b  = llamaindex_OllamaLLM(model_name="deepseek-r1:32b", temperature=0.2)

# codellama_7b   = llamaindex_OllamaLLM(model_name="codellama:7b-code-fp16", temperature=0.1)
# codellama_7b_q4 = llamaindex_OllamaLLM(model_name="codellama:7b-code", temperature=0.1)
# codellama_7b_q8 = llamaindex_OllamaLLM(model_name="codellama:7b-code-q8_0", temperature=0.1)
# codellama_13b_q4 = llamaindex_OllamaLLM(model_name="codellama:13b", temperature=0.1)

# llama3_1 = llamaindex_OllamaLLM(model_name="llama3.1:8b-text-fp16", temperature=0.1)

# deepseek_32b = langchain_OllamaLLM(model_name="deepseek-r1:32b", request_timeout=360.0, temperature=0.2)

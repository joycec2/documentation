from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import  Ollama
from llama_index.core import Settings

class LLM:
    def __init__(self, model_name: str, temperature: float = 0.2, max_token: int = 2048, request_timeout: float = 360.0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_token = max_token
        self.request_timeout = request_timeout
        self.llm = self.configure_llm(model_name, temperature, max_token, request_timeout)

    def configure_llm(self, model_name: str, temperature: float = 0.2, max_token: int = 2048, request_timeout: float = 360.0):
        """Configure the LLM settings."""
        return Ollama(
            model=model_name,
            request_timeout=request_timeout,
            temperature=temperature,
            max_token=max_token
        )
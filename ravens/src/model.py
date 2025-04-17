from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import  Ollama
from llama_index.core import Settings
from langchain_ollama import ChatOllama

class llamaindex_OllamaLLM:
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

class langchain_OllamaLLM:
    def __init__(self, model_name: str, temperature: float = 0.2, request_timeout: float = 360.0):
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.llm = self.configure_llm(model_name, temperature, request_timeout)

    def configure_llm(self, model_name: str, temperature: float = 0.2, request_timeout: float = 360.0):
        """Configure the LLM settings."""
        return ChatOllama(
            model=model_name,
            request_timeout=request_timeout,
            temperature=temperature
        )
    

    deepseek_32b = ChatOllama(model="deepseek-r1:32b", request_timeout=360.0, temperature=0.2)

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
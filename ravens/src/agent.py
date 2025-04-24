"""
Agent module.

Defines various types of agents for querying, reasoning, and interacting with vector databases
or SQL engines. Each agent extends BaseAgent and implements custom logic in the `query` method.

Classes:
    - BaseAgent: Abstract class defining the interface for all agents.
    - ReasoningAgent: Performs logical reasoning and multi-step inference.
    - RAGAgent: Uses vector databases and retrieval-augmented generation.
    - Text2SQLAgent: Converts natural language into SQL and queries databases.
"""

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
    """
    Abstract base class for agents.

    Handles shared attributes like the model and prompt template, and enforces implementation of the `query` method.
    """

    def __init__(self, prompt_template: str, model):
        """
        Initialize the agent with a prompt template and a model.

        Args:
            prompt_template (str): The prompt template to use for queries.
            model: A language model or callable chain for generating responses.
        """
        self.model = model
        self.template = prompt_template

    def query(self, query: str):
        """
        Abstract method to perform a query.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def get_model(self):
        """Return the model associated with this agent."""
        return self.model

    def get_template(self):
        """Return the prompt template used by this agent."""
        return self.template


class ReasoningAgent(BaseAgent):
    """
    Agent that performs logical reasoning and multi-step inference.

    Extends BaseAgent and utilizes a LangChain-style pipeline to extract numbered responses.
    """

    def __init__(self, model):
        """
        Initialize the ReasoningAgent with a language model pipeline.

        Args:
            model: A callable LangChain model or chain.
        """
        super().__init__(model)

    def query(self, query: str):
        """
        Execute a multi-step reasoning query.

        Args:
            query (str): The user input or problem to reason about.

        Returns:
            list[str]: A list of extracted numbered answers or steps.
        """
        chain = self.template | self.model
        result = chain.invoke({"query_str": query})
        content = result.content
        questions = extract_numbered_questions(content)
        return questions


class RAGAgent(BaseAgent):
    """
    Retrieval-Augmented Generation agent.

    Retrieves relevant context using vector embeddings and generates responses using a language model.
    """

    def __init__(self, index: VectorStoreIndex, model: Ollama, vectors: ChromaVectorStore):
        """
        Initialize the RAGAgent with a vector index, model, and vector store.

        Args:
            index (VectorStoreIndex): The RAG index used to retrieve relevant chunks.
            model (Ollama): The language model to synthesize responses.
            vectors (ChromaVectorStore): Underlying vector store for context retrieval.
        """
        super().__init__(model)
        self.index = index
        self.vectors = vectors

    def query(self, query: str):
        """
        Query the index and return a response based on retrieved context.

        Args:
            query (str): The user's query.

        Returns:
            str: The model's synthesized response based on context.
        """
        response = self.index.query(query)
        return response


class Text2SQLAgent(BaseAgent):
    """
    Agent that converts natural language into SQL queries.

    Interfaces with a SQL engine to execute the queries and return results.
    """

    def __init__(self, index: VectorStoreIndex, model: Ollama, query_engine):
        """
        Initialize the Text2SQLAgent with a model and query execution engine.

        Args:
            index (VectorStoreIndex): Placeholder for consistency; not currently used.
            model (Ollama): Language model used to help generate SQL syntax.
            query_engine: A database engine that can execute raw SQL strings.
        """
        super().__init__(model)
        self.query_engine = query_engine

    def query(self, query: str):
        """
        Convert a natural language query to SQL and execute it.

        Args:
            query (str): Natural language query.

        Returns:
            result: Result from the executed SQL query.
        """
        sql_query = self._nl_to_sql(query)
        result = self.query_engine.execute(sql_query)
        return result

    def _nl_to_sql(self, natural_language: str) -> str:
        """
        Naive converter from natural language to SQL (stub).

        Args:
            natural_language (str): The user's question.

        Returns:
            str: A hardcoded SQL query string based on the input.
        """
        return f"SELECT * FROM data WHERE column = '{natural_language}'"

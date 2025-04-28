"""
Coordinates agent interaction to process a user query into decomposed logic steps,
retrieves context using RAG, translates into SQL, and returns results in structured form.

Classes:
    - Service: Manages and runs a multi-agent query pipeline.
"""

from llm.agents.agent import ReasoningAgent, Text2SQLAgent, RAGAgent
from llm.agents.model import EmbeddingService, VectorStore

class Service:
    """
    Orchestrates reasoning, context retrieval, and SQL translation for user queries.

    Attributes:
        reasoning_agent (ReasoningAgent): Agent for query decomposition.
        text2sql_agent (Text2SQLAgent): Agent for SQL query generation and execution.
        rag_agent (RAGAgent): Agent for context retrieval using embeddings.

    Methods:
        run_query(user_query): Executes the query pipeline and returns structured output.
    """
    def __init__(self, model, index, vectors, query_engine):
        """
        Initialize the service with agents for reasoning, RAG, and SQL.

        Args:
            model: LLM or model chain used by agents.
            index: VectorStoreIndex used for retrieval and search.
            vectors: Vector store used by RAG agent.
            query_engine: SQL execution interface used by the Text2SQL agent.
        """
        self.reasoning_agent = ReasoningAgent(index=index, model=model)
        self.text2sql_agent = Text2SQLAgent(index=index, model=model, query_engine=query_engine)
        self.rag_agent = RAGAgent(index=index, model=model, vectors=vectors)

    def run_query(self, user_query: str) -> dict:
        """
        Execute the query processing pipeline.

        Decomposes the user query, retrieves relevant documents, generates SQL,
        executes the query, and returns all results.

        Args:
            user_query (str): The original natural language query.

        Returns:
            dict: A dictionary containing:
                - original_query (str)
                - subqueries (list[str])
                - retrieved_contexts (list[str])
                - sql_queries (list[str])
                - results (list[Any])
        """
        reasoning_steps = self.reasoning_agent._decompose_query(user_query)
        rag_contexts = [self.rag_agent.query(step) for step in reasoning_steps]
        sql_queries = [self.text2sql_agent._nl_to_sql(ctx) for ctx in rag_contexts]
        final_results = [self.text2sql_agent.query(ctx) for ctx in rag_contexts]

        return {
            "original_query": user_query,
            "subqueries": reasoning_steps,
            "retrieved_contexts": rag_contexts,
            "sql_queries": sql_queries,
            "results": final_results
        }

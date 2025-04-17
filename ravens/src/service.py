from agent import ReasoningAgent, Text2SQLAgent, RAGAgent
from model import EmbeddingService, VectorStore

class Service:
    def __init__(self, model, index, vectors, query_engine):
        self.reasoning_agent = ReasoningAgent(index=index, model=model)
        self.text2sql_agent = Text2SQLAgent(index=index, model=model, query_engine=query_engine)
        self.rag_agent = RAGAgent(index=index, model=model, vectors=vectors)

    def run_query(self, user_query: str) -> dict:
        #reasoning agent will break down complex query
        reasoning_steps = self.reasoning_agent._decompose_query(user_query)

        # RAG provides relevant context for each step
        rag_contexts = [self.rag_agent.query(step) for step in reasoning_steps]

        # generate SQL for each decomposed step
        sql_queries = [self.text2sql_agent._nl_to_sql(ctx) for ctx in rag_contexts]

        #execute and gather results
        final_results = [self.text2sql_agent.query(ctx) for ctx in rag_contexts]

        # construct UI-friendly response
        return {
            "original_query": user_query,
            "subqueries": reasoning_steps,
            "retrieved_contexts": rag_contexts,
            "sql_queries": sql_queries,
            "results": final_results
        }

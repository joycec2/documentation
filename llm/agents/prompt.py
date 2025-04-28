"""
Defines structured prompting strategies using LLMs for various use cases, including
retrieval-augmented generation, SQL query formulation, reasoning, and validation.

Classes:
    - BasePrompt: Abstract class for managing prompt templates.
    - RAGPrompt: Integrates retrieved context into the prompt.
    - Text2SQLPrompt: Structures prompts for SQL generation.
    - ReasoningPrompt: Guides LLM through logic using examples.
    - ValidationPrompt: Applies rule-based checking to LLM outputs.
"""

from llama_index.core import PromptTemplate

class BasePrompt:
    """
    Abstract base class for prompt construction and formatting.

    Methods:
        get_template(): Returns the base prompt string. Must be implemented by subclasses.
        format(query): Formats a query using the prompt template.
    """
    def __init__(self):
        self.template = self.get_template()

    def get_template(self) -> str:
        """
        Returns the prompt template string.
        """
        raise NotImplementedError("Subclasses must implement get_template().")

    def format(self, query: str):
        """
        Format the given query string using the prompt template.

        Args:
            query (str): The user query to embed in the template.

        Returns:
            str: A formatted prompt.
        """
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=query)


class RAGPrompt(BasePrompt):
    """
    Prompt class for retrieval-augmented generation (RAG).

    Adds contextual documents to the query for more grounded generation.
    """
    def get_template(self) -> str:
        return (
            "You are a helpful assistant with access to retrieved documents.\n"
            "Answer the user question using only the context below.\n\n"
            "Context:\n{query_str}\n\n"
            "Answer:"
        )

    def format_with_context(self, query: str, context_docs: list[str]):
        """
        Format the prompt with both query and retrieved context.

        Args:
            query (str): The user question.
            context_docs (list[str]): A list of context strings.

        Returns:
            str: A full prompt including embedded context.
        """
        context_block = "\n\n".join(context_docs)
        full_query = f"{context_block}\n\n{query}"
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)


class Text2SQLPrompt(BasePrompt):
    """
    Prompt class for natural language to SQL translation.

    Includes schema and example SQL conversions to guide generation.
    """
    def __init__(self, schema: str, examples: list[str]):
        self.schema = schema
        self.examples = examples
        super().__init__()

    def get_template(self) -> str:
        return (
            "You are an expert Text-to-SQL model.\n"
            "Given a schema and user question, write a correct SQL query.\n\n"
            "{query_str}\n\nSQL:"
        )

    def format_with_schema(self, query: str):
        """
        Format the query with schema and examples included.

        Args:
            query (str): The user's natural language query.

        Returns:
            str: A complete prompt with schema and examples.
        """
        examples_block = "\n".join(self.examples)
        full_query = (
            f"-- Schema:\n{self.schema}\n\n"
            f"-- Examples:\n{examples_block}\n\n"
            f"-- User Query:\n{query}"
        )
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)


class ReasoningPrompt(BasePrompt):
    """
    Prompt class for structured reasoning using examples.

    Designed for multi-step inference and logical breakdown.
    """
    def __init__(self, examples: list[str]):
        self.examples = examples
        super().__init__()

    def get_template(self) -> str:
        return (
            "You are an analytical assistant helping to break down complex problems.\n"
            "Use logical reasoning based on the examples provided and answer the question below.\n\n"
            "{query_str}\n\nAnswer:"
        )

    def format_with_examples(self, query: str):
        """
        Format the query with step-by-step reasoning examples.

        Args:
            query (str): The complex question to answer.

        Returns:
            str: Prompt with embedded reasoning examples.
        """
        examples_block = "\n".join(f"Example {i+1}:\n{ex}" for i, ex in enumerate(self.examples))
        full_query = f"{examples_block}\n\nQuestion:\n{query}"
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)


class ValidationPrompt(BasePrompt):
    """
    Prompt class for validating responses based on rule compliance.

    Ensures responses meet criteria for accuracy and format.
    """
    def __init__(self, rules: list[str]):
        self.rules = rules
        super().__init__()

    def get_template(self) -> str:
        return (
            "You are a validation assistant that ensures answers meet the following rules.\n"
            "Check if the response below satisfies each rule and explain any violations.\n\n"
            "{query_str}"
        )

    def format_with_rules(self, query: str):
        """
        Format the prompt with validation rules and a user response.

        Args:
            query (str): The generated response to evaluate.

        Returns:
            str: A prompt including the rule checklist and response.
        """
        rules_block = "\n".join(f"- {rule}" for rule in self.rules)
        full_query = f"Validation Rules:\n{rules_block}\n\nResponse to Validate:\n{query}"
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)


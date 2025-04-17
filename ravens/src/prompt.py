from llama_index.core import PromptTemplate

class BasePrompt:
    def __init__(self):
        self.template = self.get_template()

    def get_template(self) -> str:
        raise NotImplementedError("Subclasses must implement get_template().")

    def format(self, query: str):
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=query)


class RAGPrompt(BasePrompt):
    def get_template(self) -> str:
        return (
            "You are a helpful assistant with access to retrieved documents.\n"
            "Answer the user question using only the context below.\n\n"
            "Context:\n{query_str}\n\n"
            "Answer:"
        )

    def format_with_context(self, query: str, context_docs: list[str]):
        context_block = "\n\n".join(context_docs)
        full_query = f"{context_block}\n\n{query}"
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)


class Text2SQLPrompt(BasePrompt):
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
        examples_block = "\n".join(self.examples)
        full_query = (
            f"-- Schema:\n{self.schema}\n\n"
            f"-- Examples:\n{examples_block}\n\n"
            f"-- User Query:\n{query}"
        )
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)


class ReasoningPrompt(BasePrompt):
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
        examples_block = "\n".join(f"Example {i+1}:\n{ex}" for i, ex in enumerate(self.examples))
        full_query = f"{examples_block}\n\nQuestion:\n{query}"
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)


class ValidationPrompt(BasePrompt):
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
        rules_block = "\n".join(f"- {rule}" for rule in self.rules)
        full_query = f"Validation Rules:\n{rules_block}\n\nResponse to Validate:\n{query}"
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)

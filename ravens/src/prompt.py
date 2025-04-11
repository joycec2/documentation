from llama_index.core import PromptTemplate

class BasePrompt:
    def __init__(self, template: str):
        self.template = template

    def format(self, query:str):
        return PromptTemplate(template=self.template, input_variables="query_str") 
    
class RAGPrompt(BasePrompt):
    def __init__(self, template: str):
        super().__init__(template)

    def format_with_context(self, query: str, context_docs: list[str]):
        context_block = "\n\n".join(context_docs)
        full_query = f"{context_block}\n\n{query}"
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)

class Text2SQLPrompt(BasePrompt):
    def __init__(self, template: str, schema: str, examples: list[str]):
        super().__init__(template)
        self.schema = schema
        self.examples = examples

    def format_with_schema(self, query: str):
        examples_block = "\n".join(self.examples)
        full_query = f"-- Schema:\n{self.schema}\n\n-- Examples:\n{examples_block}\n\n-- User Query:\n{query}"
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)

class ReasoningPrompt(BasePrompt):
    def __init__(self, template: str, examples: list[str]):
        super().__init__(template)
        self.examples = examples

    def format_with_examples(self, query: str):
        examples_block = "\n".join(f"Example {i+1}:\n{ex}" for i, ex in enumerate(self.examples))
        full_query = f"{examples_block}\n\nQuestion:\n{query}"
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)

class ValidationPrompt(BasePrompt):
    def __init__(self, template: str, rules: list[str]):
        super().__init__(template)
        self.rules = rules

    def format_with_rules(self, query: str):
        """
        Formats the validation prompt by appending explicit validation rules.
        """
        rules_block = "\n".join(f"- {rule}" for rule in self.rules)
        full_query = f"Validation Rules:\n{rules_block}\n\nResponse to Validate:\n{query}"
        return PromptTemplate(template=self.template, input_variables=["query_str"]).format(query_str=full_query)

from llama_index.core import PromptTemplate

class BasePrompt:
    def __init__(self, template: str):
        self.template = template

    def format(self, query:str):
        return PromptTemplate(template=self.template, input_variables="query_str") 
    
class RAGPrompt(BasePrompt):
    def __init__(self, template: str):
        super().__init__(template)
        

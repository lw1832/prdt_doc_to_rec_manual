from project.doc_to_recommendation.llm.model.base_model import BaseModel
from project.doc_to_recommendation.llm.register.llm_register import LLM_REGISTER


@LLM_REGISTER.register_model("ollama_model")
class OllamaModel(BaseModel):

    def __init__(self, config: dict):
        super().__init__(config)
        self.tools = []

    def bind_tools(self, tools):
        self.tools = tools

    def agent_calls(self, text, image=None, prompt=None):
        pass
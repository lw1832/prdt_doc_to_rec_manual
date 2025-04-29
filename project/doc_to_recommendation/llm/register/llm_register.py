from project.doc_to_recommendation.llm.model.base_model import BaseModel


class LLMRegister:
    def __init__(self):
        self.register = {}

    def register_model(self, model_name: str):
        def decorator(model: BaseModel):
            if model_name not in self.register:
                self.register[model_name] = model
            return model
        return decorator

    def get_model(self, model_name):
        return self.register[model_name]

LLM_REGISTER = LLMRegister()
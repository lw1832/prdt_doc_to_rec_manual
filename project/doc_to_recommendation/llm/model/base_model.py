class BaseModel:
    def __init__(self, config: dict):
        self.config = config

    def bind_tools(self, tools):
        self.tools = tools

    def agent_calls(self, prompt, text, image):
        pass
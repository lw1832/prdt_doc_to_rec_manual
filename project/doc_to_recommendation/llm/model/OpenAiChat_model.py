from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from project.doc_to_recommendation.llm.model.base_model import BaseModel
from project.doc_to_recommendation.llm.register.llm_register import LLM_REGISTER
import os

os.environ["OPENAI_API_KEY"] = "sk-LGylSerFax4P4OiNZAKHyLQnf0VLHAcpltDsy4OehkPLXkUC"
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"

@LLM_REGISTER.register_model("open_ai_chat_model")
class OpenAIChatModel(BaseModel):

    def __init__(self, config: dict):
        super().__init__(config)
        model = config.get('model', 'gpt-4o')
        temperature = config.get('temperature', 0.3)
        self.model = ChatOpenAI(model=model, temperature=temperature)
        self.tools = []

    def bind_tools(self, tools):
        self.tools = tools
        self.model = self.model.bind_tools(tools)
        return self


    def agent_calls(self, text, image=None, prompt=None):
        # 构造输入信息
        call_message = []
        if prompt:
            call_message.append(SystemMessage(content=prompt))
        human_mes_content = [
            {"type": "text", "text": text},
        ]
        if image:
            human_mes_content.append(image)
        call_message.append(HumanMessage(content=human_mes_content))
        # 外呼llm
        return self.model.invoke(call_message)

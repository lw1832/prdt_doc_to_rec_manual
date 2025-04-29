import os
from typing import Optional

import project.doc_to_recommendation.llm.model
from project.doc_to_recommendation.llm.register.llm_register import LLM_REGISTER

os.environ["OPENAI_API_KEY"] = "sk-LGylSerFax4P4OiNZAKHyLQnf0VLHAcpltDsy4OehkPLXkUC"
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"

def get_model(model_name):
    return LLM_REGISTER.get_model(model_name)

def init_model(model_name: str, config: dict, tools: Optional[list]):
    model = get_model(model_name)
    return model(config=config).bind_tools(tools=tools)

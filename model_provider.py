import os
import openai
from zai import ZhipuAiClient
from dotenv import load_dotenv

load_dotenv()

class ModelProvider:
    """统一的模型调用 Provider 管理类"""

    def __init__(self):
        # 初始化 provider 客户端
        self.providers = {
            "openai": {
                "client": openai,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "api_base": os.getenv("OPENAI_API_BASE"),
                "api_type": "openai",
                "api_version": None,
            },
            "zhipu": {
                "client": ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY")),
            },
        }

    def chat_completion(self, model: str, system_prompt: str, user_content: str, provider: str = "openai", temperature: float = 0):
        """
        统一的聊天补全接口
        :param model: 模型 id
        :param system_prompt: 系统提示词
        :param user_content: 用户输入
        :param provider: 模型提供方，默认 openai
        :param temperature: 采样温度
        """
        if provider == "openai":
            cfg = self.providers["openai"]
            openai.api_key = cfg["api_key"]
            openai.api_base = cfg["api_base"]
            openai.api_type = cfg["api_type"]
            openai.api_version = cfg["api_version"]

            response = cfg["client"].ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content

        elif provider == "zhipu":
            client = self.providers["zhipu"]["client"]
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unsupported provider: {provider}")
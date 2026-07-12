# 将 API Key 配置到环境变量中
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# 创建客户端
def get_llm_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

# 可复用调用模块
def ask_llm(prompt: str) -> str:
    response = get_llm_client().chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=[
            {"role": "user", "content": prompt},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )

    return response.choices[0].message.content or ""

# 测试
if __name__ == "__main__":
    answer = ask_llm("请用一句话解释什么是 RAG。")
    print(answer)

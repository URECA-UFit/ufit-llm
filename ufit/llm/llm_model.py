import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def get_openai_model(temperature: float = 0.0, max_token: int = 256, streaming: bool = False):
    return ChatOpenAI(
        model_name="gpt-4",
        temperature=temperature,
        max_tokens=max_token,
        streaming=streaming,
    )

def get_anthropic_model(temperature: float = 0.0, max_token: int = 256, streaming: bool = False):
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=temperature,
        max_tokens_to_sample=max_token,
        streaming=streaming,
    )

def get_llm_model(temperature: float = 0.0, max_token: int = 256, streaming: bool = False):
    return get_anthropic_model(temperature,max_token,streaming)


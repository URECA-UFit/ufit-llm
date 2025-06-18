import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from ufit.services.formatter import generate_final_output

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#plans.json 경로
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'plans.json'))

#json 파일을 문서로 변환
with open(file_path, 'r', encoding='utf-8') as f:
    plans = json.load(f)

#문서 생성
docs = [
    Document(
        page_content=generate_final_output(plan),
        metadata={"name": plan["plan_name"], "idx": plan["idx"]}
    )
    for plan in plans
]

#임베딩 모델 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

#PGVector 연결 설정
PGVECTOR_CONNECTIONS_STRING = os.getenv("PGVECTOR_CONNECTIONS_STRING")
collection_name = "plans"

vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embedding_model,
    connection_string=PGVECTOR_CONNECTIONS_STRING,
    collection_name=collection_name
)

print("embedding success")
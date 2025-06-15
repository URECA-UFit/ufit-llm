# embedding_service.py
import os
from pymongo import MongoClient
from bson import ObjectId
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from ufit.services.formatter import generate_final_output
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "ufit"
COLLECTION_NAME = "rate_plan"
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTIONS_STRING")
COLLECTION_NAME_VECTOR = "plans"

mongo_client = MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]


# 테스트 시 덮어쓰기용 함수
def set_mongo_collection(mock_collection):
    global collection
    collection = mock_collection


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PGVector(
    embedding_function=embedding_model,
    connection_string=PGVECTOR_CONNECTION_STRING,
    collection_name=COLLECTION_NAME_VECTOR,
)


def embed_single_rateplan(rateplan_id: str):
    plan = collection.find_one({"_id": ObjectId(rateplan_id)})
    if not plan:
        print(f" MongoDB에 해당 요금제 ID {rateplan_id} 없음")
        return

    doc = Document(
        page_content=generate_final_output(plan),
        metadata={
            "mongo_id": str(plan["_id"]),
            "plan_name": plan.get("plan_name", ""),
            "idx": plan.get("idx", ""),
        },
    )

    vectorstore.add_documents([doc])
    print(f"요금제 {rateplan_id} 임베딩 완료")


def delete_rateplan_vector(rateplan_id: str):
    vectorstore.delete(filter={"mongo_id": rateplan_id})
    print(f"요금제 {rateplan_id} 벡터 삭제 완료")

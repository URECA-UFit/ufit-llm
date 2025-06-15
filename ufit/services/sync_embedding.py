"""
sync_embedding.py

이 스크립트는 MongoDB에 저장된 요금제 데이터를 PGVector에 임베딩하여
벡터 검색이 가능하도록 동기화하는 작업을 수행합니다.

- MongoDB와 PGVector 벡터 저장소 간의 요금제 데이터 싱크 맞춤
- 새로운 요금제가 MongoDB에 추가되었으면 → 벡터 임베딩 후 PGVector에 저장
- MongoDB에서 삭제된 요금제가 있으면 → PGVector에서도 제거

주의 ) 이 코드는 mongodb와 pgvector를 동기화하는 단발성 스크립트로 service, routes 코드와는 관계 없음
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from formatter import generate_final_output

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTIONS_STRING")

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "ufit"
COLLECTION_NAME = "rate_plan"
COLLECTION_NAME_VECTOR = "plans"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# MongoDB 연결
mongo_client = MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

# MongoDB에서 요금제 전체 조회
plans = list(collection.find({}))
mongo_ids = set(str(plan["_id"]) for plan in plans)

# 임베딩 모델
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# PGVector 초기화
vectorstore = PGVector(
    embedding_function=embedding_model,
    connection_string=PGVECTOR_CONNECTION_STRING,
    collection_name=COLLECTION_NAME_VECTOR,
)

# PGVector에서 저장된 모든 문서의 mongo_id 수집
existing_docs = vectorstore.similarity_search("임시질문", k=10000)
vector_mongo_ids = set(
    doc.metadata.get("mongo_id")
    for doc in existing_docs
    if doc.metadata.get("mongo_id")
)

# 신규 삽입할 문서 필터링
new_plans = [plan for plan in plans if str(plan["_id"]) not in vector_mongo_ids]

# 제거할 문서 필터링 (MongoDB에는 없고 PGVector에만 있는 것)
to_delete_ids = vector_mongo_ids - mongo_ids

# 신규 문서 임베딩
new_docs = [
    Document(
        page_content=generate_final_output(plan),
        metadata={
            "mongo_id": str(plan["_id"]),
            "plan_name": plan.get("plan_name", ""),
            "idx": plan.get("idx", ""),
        },
    )
    for plan in new_plans
]

if new_docs:
    vectorstore.add_documents(new_docs)
    print(f"신규 임베딩: {len(new_docs)}건")
else:
    print("신규 요금제 없음 (모두 이미 임베딩됨)")

# 삭제된 요금제 제거 (PGVector에만 있는 것)
if to_delete_ids:
    print(f"삭제할 요금제 {len(to_delete_ids)}건: PGVector에만 존재")

    for mongo_id in to_delete_ids:
        vectorstore.delete(filter={"mongo_id": mongo_id})
    print("삭제 완료")
else:
    print("PGVector와 MongoDB 동기화 완료 - 삭제 항목 없음")

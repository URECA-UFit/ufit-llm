# embedding_service.py
import os
from pymongo import MongoClient
from bson import ObjectId
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from ufit.services.formatter import generate_final_output
from dotenv import load_dotenv
from ufit.exceptions import (
    RatePlanNotFoundException,
    VectorCreateException,
    VectorDeleteException,
)
from ufit.dto.rateplan_request import callRatePlanRequest

from sqlalchemy import text, create_engine

load_dotenv()

DB_NAME = "ufit"
COLLECTION_NAME = "rate_plans"
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTIONS_STRING")
COLLECTION_NAME_VECTOR = "langchain_pg_embedding"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PGVector(
    embedding_function=embedding_model,
    connection_string=PGVECTOR_CONNECTION_STRING,
    collection_name=COLLECTION_NAME_VECTOR,
)

# 요금제 임베딩
def embed_single_rateplan(request: callRatePlanRequest):
    
    plan_data = {
        "rateplan_id": request.ratePlanId,
        "plan_name": request.planName,
        "summary": request.summary,
        "monthly_fee": request.monthlyFee,
        "discount_fee": request.discountFee,
        "data_allowance": request.dataAllowance,
        "voice_allowance": request.voiceAllowance,
        "sms_allowance": request.smsAllowance,
        "basic_benefit": request.basicBenefit,
        "discount_benefit": request.discountBenefit,
        "special_benefit": request.specialBenefit,
        "device_type": request.deviceType,
        "data_sharing": request.dataSharing,
        "social_category": request.socialCategory,
        "extra_data": request.extraData,
        "data_category": request.dataCategory,
    }
    mongo_id = plan_data["rateplan_id"]

    try:
        doc = Document(
            page_content=generate_final_output(plan_data),
            metadata={
                "mongo_id": mongo_id,
                "plan_name": plan_data.get("plan_name", ""),
            },
        )
        vectorstore.add_documents([doc])
    except Exception as e:
        print(f"[벡터 임베딩 에러] {e}")
        raise VectorCreateException(str(e))


def _acquire_pg_conn(store: PGVector):
    """vectorstore 안에 열린 커넥션이 없으면 엔진에서 새로 연결"""
    conn = getattr(store, "_conn", None) or getattr(store, "_connection", None)
    if conn is not None:
        return conn, False  # (커넥션, close_needed)

    # 일부 버전엔 _engine 속성, 없으면 DSN으로 새 엔진 생성
    engine = getattr(store, "_engine", None) or create_engine(
        PGVECTOR_CONNECTION_STRING
    )
    return engine.connect(), True  # 새로 열었으니 사용 뒤 close


def delete_rateplan_vector(rateplan_id: str):
    try:
        conn, should_close = _acquire_pg_conn(vectorstore)

        sql = text(
            """
            DELETE FROM langchain_pg_embedding
            WHERE cmetadata->>'mongo_id' = :mid
            """
        )

        with conn.begin():  # 트랜잭션
            result = conn.execute(sql, {"mid": str(rateplan_id)})

        print(f"[DEBUG] deleted rows → {result.rowcount}")
        if should_close:
            conn.close()

        if result.rowcount == 0:
            print("[주의] 해당 mongo_id 레코드가 존재하지 않습니다.")

    except Exception as e:
        print(f"[벡터 삭제 에러] {e}")
        raise VectorDeleteException(str(e))

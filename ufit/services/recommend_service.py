import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

from ufit.services.user_service import get_user_full_info
from ufit.dto.user_info import UserFullInfoDTO
from sqlalchemy.orm import Session
from pymongo.database import Database

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

PGVECTOR_CONNECTIONS_STRING = os.getenv("PGVECTOR_CONNECTIONS_STRING")
collection_name = "plans"

vectorstore = PGVector(
    embedding_function = embedding_model,
    collection_name = collection_name,
    connection_string = PGVECTOR_CONNECTIONS_STRING
)

def search_similar_plans(query: str, k: int = 2):
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


def make_recommend( user_id: int, base_prompt: str, postgre_db: Session, mongo_db: Database,):
    
    # 1) 사용자 정보 조회
    user_info: UserFullInfoDTO = get_user_full_info(user_id, postgre_db, mongo_db)
    

    # 3) LLM용 최종 프롬프트 조립
    prompt = (
        
        "다음은 사용자 정보입니다:\n"
        f"이메일: {user_info.email}\n"
        f"나이: {user_info.age}\n"
        f"성별: {user_info.gender}\n"
        f"현재 요금제: {user_info.rate_plan.plan_name}\n"
        f"기본요금: {user_info.rate_plan.monthly_fee}\n"
        f"데이터 사용 현황: {[u.usage_amount for u in user_info.data_usages]} (월별)\n"
        f"통화 사용 현황: {[u.usage_amount for u in user_info.call_usages]} (월별)\n"
        f"SMS 사용 현황: {[u.usage_amount for u in user_info.sms_usages]} (월별)\n"
        f"등록 디바이스: {[d.device_name + '(' + str(d.data_type) + ')' for d in user_info.devices]}\n\n"
        "위 사용자 정보에 기반해, 다음 기준에 맞춰 요금제를 추천해 주세요:\n"

        f"{base_prompt}"
    )
    
    # 4) LLM 호출 (예시)
    
    return search_similar_plans(prompt, 2)
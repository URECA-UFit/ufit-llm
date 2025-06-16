import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient
from ufit.services.recommend_service_hj import make_recommend

# .env 불러오기
load_dotenv()

# 테스트 값
TEST_USER_ID = 1
TEST_CHAT_ROOM_ID = 999

# PostgreSQL 연결
PG_URL = os.getenv("PGVECTOR_CONNECTIONS_STRING")
engine = create_engine(PG_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
pg_session = SessionLocal()

# MongoDB 연결
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client.get_database() 

# 테스트 실행
if __name__ == "__main__":
    try:
        print("📡 LLM 기반 요금제 추천 테스트 시작...\n")

        test_cases = [
            # (질문, 추천 여부, 설명)
            ("영상 스트리밍 많이 해요. 어떤 요금제가 좋을까요?", True, "추천 질문 + 유사도 높음"),
            ("나 강아지 좋아하는데, 요금제 추천해줘", True, "추천 질문 + 유사도 낮음"),
            ("나 넷플릭스 많이 봐", False, "비추천 질문 + 유사도 높음"),
            ("날씨가 어떤가요?", False, "비추천 질문 + 유사도 낮음"),
        ]

        for prompt, is_recommend, description in test_cases:
            print(f"\n🔹 테스트 케이스: {description}")
            print(f"💬 사용자 질문: {prompt}")

            response = make_recommend(
                user_id=TEST_USER_ID,
                base_prompt=prompt,
                chat_room_id=TEST_CHAT_ROOM_ID,
                postgre_db=pg_session,
                mongo_db=mongo_db,
                is_recommend_question=is_recommend
            )
            print("✅ 응답 결과:")
            print(response)

    except Exception as e:
        print(f"\n❌ 오류 발생:\n{e}")
    finally:
        pg_session.close()
        mongo_client.close()


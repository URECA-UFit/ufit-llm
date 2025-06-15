import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

from ufit.services.user_service import get_user_full_info
from ufit.dto.user_info import UserFullInfoDTO
from sqlalchemy.orm import Session
from pymongo.database import Database
from langchain_mongodb import MongoDBChatMessageHistory

from langchain_anthropic import ChatAnthropic
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from ufit.services.llm_answer_agent import get_prompt, get_llm_response
from typing import List


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

llm_model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0.7
)

PGVECTOR_CONNECTIONS_STRING = os.getenv("PGVECTOR_CONNECTIONS_STRING")
collection_name = "plans"

vectorstore = PGVector(
    embedding_function = embedding_model,
    collection_name = collection_name,
    connection_string = PGVECTOR_CONNECTIONS_STRING
)

def search_similar_plans(query: str, k: int = 2):
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [{"content": doc.page_content, "score": score} for doc, score in results]

def save_chat_bot_message(
    collection,
    content: str,
    owner: bool,
    chat_room_id: int,
    a_plan_id: int | None = None,
    b_plan_id: int | None = None,
):
    collection.insert_one({
        "content": content,
        "owner": owner,
        "chat_room_id": chat_room_id,
        "a_plan_id": a_plan_id,
        "b_plan_id": b_plan_id
    })


def make_recommend(
    user_id: int,
    base_prompt: str,
    chat_room_id: int,
    postgre_db: Session,
    mongo_db: Database,
    is_recommend_question: bool,  # 🚨 추천 질문 여부 주입된다고 가정!!
) -> str:
    # 1) 히스토리 저장소 생성
    session_id = str(chat_room_id)
    history = MongoDBChatMessageHistory(
        connection_string=os.getenv("MONGO_URI"),
        session_id=session_id,
        database_name=mongo_db.name,
        collection_name="message_histories"
    )

    # 2) RAG 검색
    retrieved = search_similar_plans(base_prompt, k=2)
    similarity_scores = [1 - item["score"] for item in retrieved]

    similarity_threshold = 0.4 ## 🚨 임시로 설정!!

    top1_similarity = similarity_scores[0] if similarity_scores else 0.0
    is_high_similarity = top1_similarity > similarity_threshold

    retrieved_block = "\n".join(
        f"- {item['content']} (유사도: {1 - item['score']:.2f})"
        for item in retrieved
    )
    # 3) 과거 메시지
    past_msgs: List[HumanMessage | AIMessage] = history.messages[-6:]

    # 4) 현재 메시지 기록
    history.add_user_message(base_prompt)

    # 5) 사용자 정보 생성
    user_info: UserFullInfoDTO = get_user_full_info(user_id, postgre_db, mongo_db)
    user_block = (
        "다음은 사용자 정보입니다:\n"
        f"- 이메일: {user_info.email}\n"
        f"- 나이: {user_info.age}\n"
        f"- 성별: {user_info.gender}\n"
        f"- 현재 요금제: {user_info.rate_plan.plan_name}\n"
        f"- 기본요금: {user_info.rate_plan.monthly_fee}\n\n"
    )

     # 6) 분기 처리
    if is_recommend_question and is_high_similarity:
        # Case 1: 추천 + 유사도 높음
        rendered_msgs = get_prompt(user_block, retrieved_block, base_prompt)
        llm_messages = past_msgs + rendered_msgs
        answer = get_llm_response(llm_messages)

    elif is_recommend_question and not is_high_similarity:
        # Case 2: 추천 + 유사도 낮음
        answer = (
            "죄송합니다. 고객님의 질문과 유사한 요금제를 찾기 어려웠습니다.\n"
            "보다 정확한 추천을 위해 다음과 같은 정보를 알려주시면 좋습니다:\n"
            "- 하루 평균 데이터 사용량\n"
            "- 통화 시간 또는 패턴\n"
            "- 예산 또는 요금 한도\n"
            "이 정보를 기반으로 최적의 요금제를 추천해드릴 수 있습니다!"
        )

    elif not is_recommend_question and is_high_similarity:
        # Case 3: 비추천 + 유사도 높음
        answer = (
            "해당 질문은 요금제 추천과는 직접적인 관련은 없지만,\n"
            "요금제 추천이 필요하시다면 언제든지 도와드릴 수 있습니다.\n"
            "'추천해줘' 또는 '나에게 맞는 요금제 알려줘'와 같은 질문을 주시면 안내해드리겠습니다."
        )

    else:
        # Case 4: 비추천 + 유사도 낮음
        answer = (
            "죄송합니다. 현재 질문은 통신 요금제와 직접적인 관련이 없어 도움드리기 어려운 점 양해 부탁드립니다.\n"
            "요금제 추천이나 통신 서비스 관련 문의가 있으시다면 언제든지 도와드리겠습니다!"
        )

    chat_bot_messages = mongo_db.get_collection("chat_bot_messages")

    # MongoDB에 사용자 질문 저장
    save_chat_bot_message(
        collection=chat_bot_messages,
        content=base_prompt,
        owner=True,
        chat_room_id=chat_room_id
    )

    # MongoDB에 응답 저장
    save_chat_bot_message(
        collection=chat_bot_messages,
        content=answer,
        owner=False,
        chat_room_id=chat_room_id ##🚨 추천된 요금제 plans_id 저장해야함!!
    )

    # 7) 응답 저장 및 반환
    history.add_ai_message(answer)
    return answer
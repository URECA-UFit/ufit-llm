import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

from ufit.services.user_service import get_user_full_info
from ufit.dto.user_info import UserFullInfoDTO
from sqlalchemy.orm import Session
from pymongo.database import Database
from langchain_community.chat_message_histories import MongoDBChatMessageHistory

from langchain_anthropic import ChatAnthropic
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import List


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

CLAUDE_API_KEY= os.getenv("CLAUDE_API_KEY")

if not CLAUDE_API_KEY:
    raise ValueError("CLAUDE_API_KEY is not set")
os.environ["ANTHROPIC_API_KEY"] = CLAUDE_API_KEY

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

llm_model = ChatAnthropic(
    model="claude-3-haiku-20240307",  # 또는 claude-3-sonnet-20240229, claude-3-opus-20240229
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

def make_recommend(
    user_id: int,
    base_prompt: str,
    chat_room_id: int,
    postgre_db: Session,
    mongo_db: Database,
) -> str:
    
    # 1) 히스토리 스토어
    session_id = str(chat_room_id)
    history = MongoDBChatMessageHistory(
        os.getenv("MONGO_URL"), session_id, mongo_db.name, "message_histories"
    )

    # 2) RAG: 임베딩 검색
    retrieved = search_similar_plans(base_prompt, 2)
    retrieved_block = "\n".join(f"- {t}" for t in retrieved)
    
    '''

    유사도를 통해 도메인 판별해 주세요~!
    
    '''

    # 3) 과거 6턴 가져오기
    past_msgs: List[HumanMessage | AIMessage] = history.messages[-6:]

    # 4) 유저 메시지 기록
    history.add_user_message(base_prompt)

    # 5) 사용자 정보 블록 생성
    user_info: UserFullInfoDTO = get_user_full_info(user_id, postgre_db, mongo_db)
    user_block = (
        "다음은 사용자 정보입니다:\n"
        f"- 이메일: {user_info.email}\n"
        f"- 나이: {user_info.age}\n"
        f"- 성별: {user_info.gender}\n"
        f"- 현재 요금제: {user_info.rate_plan.plan_name}\n"
        f"- 기본요금: {user_info.rate_plan.monthly_fee}\n\n"
    )

    # 6) PromptTemplate 정의
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are the best rate-plan recommendation chatbot. Always respond in Korean."
        ),
        HumanMessagePromptTemplate.from_template(
            "{user_block}"
            "다음 요금제 정보를 참고해 답변해 주세요:\n\n"
            "{retrieved_block}\n\n"
            "사용자 질문: {base_prompt}"
        ),
    ])

    # 7) 템플릿 렌더링 → 메시지 리스트
    rendered_msgs = prompt.format_prompt(
        user_block=user_block,
        retrieved_block=retrieved_block,
        base_prompt=base_prompt
    ).to_messages()

    # 8) 최종 LLM 입력 = 과거 히스토리 + 방금 렌더링된 메시지
    llm_messages: List[HumanMessage | AIMessage] = []
    llm_messages.extend(past_msgs)
    llm_messages.extend(rendered_msgs)

    # 9) LLM 호출
    response_msg = llm_model.invoke(llm_messages)
    answer = response_msg.content.strip()

    # 10) 답변 기록
    history.add_ai_message(answer)

    return answer
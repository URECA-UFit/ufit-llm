import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

from ufit.services.user_service import get_user_full_info
from ufit.dto.user_info import UserFullInfoDTO
from sqlalchemy.orm import Session
from pymongo.database import Database
from langchain_mongodb import MongoDBChatMessageHistory
from ufit.database.database import save_chat_bot_message

from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage

from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from itertools import islice
from langchain.schema import HumanMessage
from ufit.services.llm_answer_agent import get_prompt, get_llm_response
from ufit.dto.recommend import AnswerType, RecommendResponse, PlanDTO


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

if not CLAUDE_API_KEY:
    raise ValueError("CLAUDE_API_KEY is not set")
os.environ["ANTHROPIC_API_KEY"] = CLAUDE_API_KEY

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

llm_model = ChatAnthropic(
    model="claude-3-haiku-20240307",  # 또는 claude-3-sonnet-20240229, claude-3-opus-20240229
    temperature=0.7,
)

"""
llm_model = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.5,
    streaming=False
)
"""


PGVECTOR_CONNECTIONS_STRING = os.getenv("PGVECTOR_CONNECTIONS_STRING")
collection_name = "plans"

vectorstore = PGVector(
    embedding_function=embedding_model,
    collection_name=collection_name,
    connection_string=PGVECTOR_CONNECTIONS_STRING,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    raw_message: str
    user_info: UserFullInfoDTO
    is_recommend: bool
    plans: List[Dict]
    a_plan: PlanDTO
    b_plan: PlanDTO
    threshold_met: bool
    final_answer: str


def analyze_intent_node(state: State):

    last = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """당신은 최고의 요금제 챗봇입니다. 사용자의 질문이 확실하게 요금제 추천해 달라는 질문인지 판단해주세요.
            판단한 후에 반드시 true 혹은 false로만 답해주세요."""
            ),
            HumanMessagePromptTemplate.from_template(
                "{query}" "true 혹은 false로 대답해줘"
            ),
        ]
    )
    rendered = prompt.format_prompt(query=last).to_messages()
    resp = llm_model.invoke(rendered)
    # JSON 파싱 로직
    is_rec = "true" in resp.content.lower()
    return {"is_recommend": is_rec}


def search_plans_node(state: State, k: int = 2):
    # 히스토리에서 HumanMessage만 추출 (필요하다면 최근 N턴만)
    user_msgs = [
        msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)
    ]

    recent = list(islice(user_msgs, max(0, len(user_msgs) - 5), len(user_msgs)))

    # history로 문맥 포함 쿼리 생성
    context_query = " ".join(recent).strip()
    if not context_query:
        context_query = state["messages"][-1].content

    # 3) 벡터 검색
    docs_with_scores = vectorstore.similarity_search_with_score(context_query, k=k)

    # 4) plans 리스트에 원본 스코어까지 담기
    plans: List[Dict[str, Any]] = []
    for doc, raw_score in docs_with_scores:
        plans.append(
            {
                "content": doc.page_content,
                "raw_score": raw_score,
            }
        )

    # 5) threshold_met 판단 (scaled_score 기준)
    threshold = 0.8
    threshold_met = any(p["raw_score"] <= threshold for p in plans)

    return {"plans": plans, "threshold_met": threshold_met}


def request_info_response_node(state: State):
    resp = (
        "죄송합니다. 현재 질문은 통신 요금제와 직접적인 관련이 없어 도움드리기 어려운 점 양해 부탁드립니다.\n"
        "요금제 추천이나 통신 서비스 관련 문의가 있으시다면 언제든지 도와드리겠습니다!"
    )

    return {"final_answer": resp, "messages": [AIMessage(content=resp)]}


def not_found_rate_response_node(state: State):
    resp = (
        "죄송합니다. 고객님의 질문과 유사한 요금제를 찾기 어려웠습니다.\n"
        "보다 정확한 추천을 위해 다음과 같은 정보를 알려주시면 좋습니다:\n"
        "- 하루 평균 데이터 사용량\n"
        "- 통화 시간 또는 패턴\n"
        "- 예산 또는 요금 한도\n"
        "이 정보를 기반으로 최적의 요금제를 추천해드릴 수 있습니다!"
    )

    return {"final_answer": resp, "messages": [AIMessage(content=resp)]}


def recommend_response_node(state: State):

    # 대화 히스토리 직렬화
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in state["messages"]
    )
    # 후보 요금제 전체 블록 생성
    plan_blocks = []
    for idx, plan in enumerate(state["plans"], start=1):
        plan_blocks.append(f"---\n요금제 {idx} 정보:\n{plan['content'].strip()}\n---")

    plans_text = "\n\n".join(plan_blocks)
    prompt = get_prompt(
        history_text, state["user_info"], plans_text, state["raw_message"]
    )

    # LLM 호출 및 응답
    resp = get_llm_response(prompt)

    # for planId,name in enumerate(state["plans"], start=1):

    return {
        "final_answer": resp,
        "a_plan": PlanDTO(planId="a", name="a_name"),
        "b_plan": PlanDTO(planId="b", name="b_name"),
        "messages": [AIMessage(content=resp)],
    }


def suggest_response_node(state: State):
    resp = (
        "해당 질문은 요금제 추천과는 직접적인 관련은 없지만,\n"
        "요금제 추천이 필요하시다면 언제든지 도와드릴 수 있습니다.\n"
        "'추천해줘' 또는 '나에게 맞는 요금제 알려줘'와 같은 질문을 주시면 안내해드리겠습니다."
    )

    return {"final_answer": resp, "messages": [AIMessage(content=resp)]}


# StateGraph 초기화
graph_builder = StateGraph(State)

# 1) 노드 등록
graph_builder.add_node("analyze_intent", analyze_intent_node)
graph_builder.add_node("search_plans", search_plans_node)
graph_builder.add_node("not_found_rate_response", not_found_rate_response_node)
graph_builder.add_node("request_info_response", request_info_response_node)
graph_builder.add_node("recommend_response", recommend_response_node)
graph_builder.add_node("suggest_response", suggest_response_node)

# 2) 시작 노드 연결
graph_builder.add_edge(START, "analyze_intent")
graph_builder.add_edge("analyze_intent", "search_plans")


# 3) SEARCH → 분기: (is_recommend, threshold_met) 네 가지 경로
def search_branch(state: State):

    if state["is_recommend"] == True and state["threshold_met"] == False:
        return "condition1"
    elif state["is_recommend"] == True and state["threshold_met"] == True:
        return "condition2"
    elif state["is_recommend"] == False and state["threshold_met"] == False:
        return "condition3"
    elif state["is_recommend"] == False and state["threshold_met"] == True:
        return "condition4"


graph_builder.add_conditional_edges(
    "search_plans",
    search_branch,
    {
        "condition1": "not_found_rate_response",
        "condition2": "recommend_response",
        "condition3": "request_info_response",
        "condition4": "suggest_response",
    },
)

# 4) 최종 응답 노드들 → END
for terminal in [
    "not_found_rate_response",
    "recommend_response",
    "request_info_response",
    "suggest_response",
]:
    graph_builder.add_edge(terminal, END)

# 5) 그래프 컴파일
ufit_graph = graph_builder.compile()


def make_recommend(
    user_id: int,
    base_prompt: str,
    chat_room_id: int,
    postgre_db: Session,
    mongo_db: Database,
):

    # 1) 과거 히스토리 로드
    session_id = str(chat_room_id)
    history = MongoDBChatMessageHistory(
        connection_string=os.getenv("MONGO_URI"),
        session_id=session_id,
        database_name=mongo_db.name,
        collection_name="message_histories",
    )

    history.add_user_message(base_prompt)

    # 2) 유저 정보 조회
    user_info = get_user_full_info(user_id, postgre_db, mongo_db)

    # 3) initial_state 설정
    initial_state: State = {
        "messages": history.messages,
        "raw_message": base_prompt,
        "user_info": user_info,
        "is_recommend": False,
        "plans": [],
        "a_plan": PlanDTO(planId="", name=""),
        "b_plan": PlanDTO(planId="", name=""),
        "threshold_met": False,
        "final_answer": "",
    }
    # 4) 그래프 실행
    result_state = ufit_graph.invoke(initial_state)

    chat_bot_messages = mongo_db.get_collection("chat_bot_messages")

    save_chat_bot_message(
        collection=chat_bot_messages,
        content=base_prompt,
        owner=True,
        chat_room_id=chat_room_id,
    )

    message_id = save_chat_bot_message(
        collection=chat_bot_messages,
        content=result_state["final_answer"],
        owner=False,
        chat_room_id=chat_room_id,
        # a_plan_id= a
        # b_plan_id= b
    )

    history.add_ai_message(result_state["final_answer"])

    a_plan_id = result_state["a_plan"].planId
    b_plan_id = result_state["b_plan"].planId

    if not a_plan_id and not b_plan_id:
        answer_type = AnswerType.GENERAL
    else:
        answer_type = AnswerType.RECOMMEND

    answer = RecommendResponse(
        messageId=str(message_id),
        answer=result_state["final_answer"],
        answerType=answer_type,
        recommendPlans=[result_state["a_plan"], result_state["b_plan"]],
    )

    return answer

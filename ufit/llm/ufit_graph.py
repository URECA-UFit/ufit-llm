# <Ufit_Graph 시각화>
#
# [START]
#    |
#    v
# [is_safe_query_node]
#    ├─ if False ──> [respond_to_unsafe_query_node] ──> [END]
#    └─ if True ──> [rewrite_query_node]
#                        |
#                        v
#              [is_rateplan_related_query_node]
#                ├─ if False ──> [respond_to_unrelated_rateplan_query_node] ──> [END]
#                └─ if True ──> [is_recommendation_intent_node]
#                                 ├─ if False ──> [respond_to_non_recommendation_intent_node] ──> [END]
#                                 └─ if True ──> [respond_to_recommendation_intent_node] ──> [END]


import os,json

from ufit.services.user_service import stringify_user_full_info
from ufit.dto.user_info import UserFullInfoDTO


from langchain_community.vectorstores import PGVector
from langchain.schema import AIMessage
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from ufit.dto.recommend import PlanDTO
from langchain_teddynote.graphs import visualize_graph
from ufit.llm.ai_model import get_anthropic_model, embedding_model
from ufit.llm.ufit_graph_prompt import (
    get_safe_query_prompt,
    get_rateplan_related_prompt,
    get_recommendation_intent_prompt,
    get_recommendation_prompt,
    get_non_recommendation_prompt,
    get_other_carrier_prompt
)

PGVECTOR_CONNECTIONS_STRING = os.getenv("PGVECTOR_CONNECTIONS_STRING")
collection_name = "plans"

vectorstore = PGVector(
    embedding_function = embedding_model,
    collection_name = collection_name,
    connection_string = PGVECTOR_CONNECTIONS_STRING
)

num_of_recommend_plan = 2 
respond_to_unsafe_query = "죄송합니다. 해당 요청은 서비스 이용 정책에 따라 처리할 수 없습니다.\n다른 질문을 해주세요."
respond_to_unrelated_rateplan_query = "죄송합니다. 요금제와 관련없는 질문은 답변을 드릴 수가 없네요.\n요금제와 관련된 질문을 해주세요."
respond_to_other_carrier_query = "죄송합니다. 저는 LG U+ 요금제에 한해 상담을 제공하고 있습니다.\n타 통신사 관련 문의는 답변드릴 수 없습니다."
respond_to_non_recommendation_intent_if_llm_error= "요금제에 대해 궁금한 점이 있으신가요? 자세히 말씀해 주시면 추천도 도와드릴 수 있어요."


# -------- 그래프 상태 정의 --------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    content: str
    rewriten_content: str
    is_safe: bool
    is_other_carrier: bool
    is_rateplan_related: bool
    is_recommendation_intent: bool
    user_info: UserFullInfoDTO
    a_plan: PlanDTO
    b_plan: PlanDTO
    answer: str


# -------- 노드 함수 정의 --------
# 금칙어 처리하는 노드(욕설, 개인정보 추출 등)
def is_safe_query_node(state: State):

    prompt = get_safe_query_prompt(input=state["content"])
    response = get_anthropic_model(temperature=0.0, max_token=100).invoke(prompt.to_messages())
    
    try:
        result = json.loads(response.content)
        is_safe = result["is_safe"]
    except Exception as e:
        # 파싱을 실패할 수 있어 기본적으로 unsafe 처리한다.
        is_safe = False

    return {
        "is_safe": is_safe
    }


# 금칙어 질문에 응답 반환하는 노드(정적 대답)"""
def respond_to_unsafe_query_node(state: State):
    return {
        "answer": respond_to_unsafe_query,
        "messages": [AIMessage(content=respond_to_unsafe_query)]
    }


# 멀티턴을 위해 사용자 질문을 정제하는 노드
# 멀티턴 성능 개선을 위해 두 번 비동기로 api호출을 시도한다. 그리고 더 나은 답변을 LLM에 선택하게 한다. (Self-Evaluation Prompting 기법)

def rewrite_query_node(state: State):
    import asyncio, json
    from ufit.llm.ufit_graph_prompt import get_rewrite_query_prompt, get_self_evaluation_prompt
    from ufit.llm.ai_model import get_openai_model

    messages = state["messages"]

    async def _rewrite_once(messages, temperature):
        prompt = get_rewrite_query_prompt(messages)
        return await get_openai_model(temperature).ainvoke(prompt.to_messages())

    async def _run():
        try:
            # 멀티턴 정제 두 번 수행
            rewrite_1, rewrite_2 = await asyncio.gather(
                _rewrite_once(messages, temperature=0.35),
                _rewrite_once(messages, temperature=0.42),
            )

            # JSON 파싱 시도
            try:
                q1 = json.loads(rewrite_1.content)["rewritten_question"]
                q2 = json.loads(rewrite_2.content)["rewwritten_question"]
            except Exception:
                fallback = rewrite_1.content if "rewritten_question" not in rewrite_1.content else q1
                return {"rewriten_content": fallback}

            # Self-Evaluation Prompt 구성
            selection_prompt = get_self_evaluation_prompt(q1, q2)
            selection_response = await get_openai_model(temperature=0.0).ainvoke(selection_prompt.to_messages())

            try:
                choice = json.loads(selection_response.content)["chosen"]
                rewriten_query = q1 if choice == "Question 1" else q2
            except Exception:
                rewriten_query = q1

            return {"rewriten_content": rewriten_query}
        except Exception as e:
            # 예외 발생 시 fallback
            return {"rewriten_content": state["content"]}

    # 비동기 함수 실행
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(_run())
    loop.close()

    return result


# 질문이 요금제 관련인지 판단하는 노드
def is_rateplan_related_query_node(state: State):
    content = state["rewriten_content"]

    prompt = get_rateplan_related_prompt(content)
    response = get_anthropic_model(temperature=0.3, max_token=100).invoke(prompt.to_messages())

    try:
        result = json.loads(response.content)
        is_related = result["is_rateplan_related"]
    except Exception:
        is_related = False  # 파싱 실패 시 기본값

    return {
        "is_rateplan_related": is_related
    }


# 요금제 관련 없는 질문에 응답 반환하는 노드(정적 대답)"""
def respond_to_unrelated_rateplan_query_node(state: State):
    return {
        "answer": respond_to_unrelated_rateplan_query,
        "messages": [AIMessage(content=respond_to_unrelated_rateplan_query)]
    }


# 추천 의도가 있는지 판단하는 노드
def is_recommendation_intent_node(state: State):
    prompt = get_recommendation_intent_prompt(state["rewriten_content"])
    response = get_anthropic_model(temperature=0.0, max_token=100).invoke(prompt.to_messages())
    try:
        result = json.loads(response.content)
        is_recommendation_intent = result.get("is_recommendation_intent", False)
    except Exception:
        is_recommendation_intent = False  # 파싱 실패 시 기본값

    return {
        "is_recommendation_intent": is_recommendation_intent
    }


# 추천 의도가 없을 경우의 응답하는 노드(LLM 대답)
def respond_to_non_recommendation_intent_node(state: State):
    prompt = get_non_recommendation_prompt(state["rewriten_content"])
    response = get_anthropic_model(temperature=0.05, max_token=1000).invoke(prompt.to_messages())

    try:
        content = response.content
    except Exception:
        content = respond_to_non_recommendation_intent_if_llm_error

    return {
        "answer": content,
        "message": [AIMessage(content=content)]
    }


# 추천 의도가 있을 경우 요금제 추천 응답하는 노드(LLM 대답)
def extract_plan_dto(doc, default_name):
    metadata = doc.metadata or {}
    return PlanDTO(
        planId=metadata.get("planId", f"unknown-{default_name}"),
        name=metadata.get("name", f"요금제 {default_name}")
    )

def respond_to_recommendation_intent_node(state: State):
    
    # Handle case where user_info might be None
    if state["user_info"] is None:
        user_text = "사용자 정보가 없습니다."
    else:
        user_text = stringify_user_full_info(state["user_info"])
    #retriever_text = f"사용자 정보: {user_text}\n\n 질문: {state["rewriten_content"]}"

    docs = vectorstore.similarity_search(state["rewriten_content"], num_of_recommend_plan)

    plan_texts = "\n\n".join(
    f"요금제 {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

    prompt = get_recommendation_prompt(user_text, plan_texts, state["rewriten_content"])

    response = get_anthropic_model(temperature=0.4, max_token=1000).invoke(prompt.to_messages())
    print(plan_texts)

    a_plan = extract_plan_dto(docs[0], "A") if len(docs) > 0 else PlanDTO(planId="", name="")
    b_plan = extract_plan_dto(docs[1], "B") if len(docs) > 1 else PlanDTO(planId="", name="")
    

    # 8. 결과 반환
    return {
        "answer": response.content,
        "a_plan": a_plan,
        "b_plan": b_plan,
        "messages": [AIMessage(content=response.content)],
    }


# 타 통신사 질문인지 판단하는 노드
def is_other_carrier_query_node(state: State):
    content = state["rewriten_content"]

    prompt = get_other_carrier_prompt(content)
    response = get_anthropic_model(temperature=0.3, max_token=100).invoke(prompt.to_messages())

    try:
        result = json.loads(response.content)
        is_other_carrier = result["is_other_carrier"]
    except Exception:
        is_other_carrier = False  # 파싱 실패 시 기본값

    return {
        "is_other_carrier": is_other_carrier
    }

# 타 통신사 질문에 응답 반환하는 노드(정적 대답)
def respond_to_other_carrier_query_node(state: State):
    return {
        "answer": respond_to_other_carrier_query,
        "messages": [AIMessage(content=respond_to_other_carrier_query)]
    }

graph_builder = StateGraph(State)

# -------- 노드 구성 --------
graph_builder.add_node("is_safe_query_node", is_safe_query_node)
graph_builder.add_node("rewrite_query_node", rewrite_query_node)
graph_builder.add_node("respond_to_unsafe_query_node", respond_to_unsafe_query_node)
graph_builder.add_node("is_other_carrier_query_node", is_other_carrier_query_node)
graph_builder.add_node("respond_to_other_carrier_query_node", respond_to_other_carrier_query_node)
graph_builder.add_node("is_rateplan_related_query_node", is_rateplan_related_query_node)
graph_builder.add_node("respond_to_unrelated_rateplan_query_node", respond_to_unrelated_rateplan_query_node)
graph_builder.add_node("is_recommendation_intent_node", is_recommendation_intent_node)
graph_builder.add_node("respond_to_non_recommendation_intent_node", respond_to_non_recommendation_intent_node)
graph_builder.add_node("respond_to_recommendation_intent_node", respond_to_recommendation_intent_node)

# -------- 분기 함수 구성 --------
def search_branch1(state: State):
    if (state["is_safe"] == True): return "safe"
    elif (state["is_safe"] == False): return "unsafe"

def search_branch2(state: State):
    if (state["is_other_carrier"] == True): return "other_carrier"
    elif (state["is_other_carrier"] == False): return "lg_uplus"

def search_branch3(state: State):
    if (state["is_rateplan_related"] == True): return "rateplan_related"
    elif (state["is_rateplan_related"] == False): return "rateplan_unrelated"

def search_branch4(state: State):
    if (state["is_recommendation_intent"] == True): return "recommendation_intent"
    elif (state["is_recommendation_intent"] == False): return "non_recommendation_intent"

# -------- 엣지 구성 --------
graph_builder.add_edge(START, "is_safe_query_node")

graph_builder.add_conditional_edges(
    "is_safe_query_node",
    search_branch1,
    path_map = {
        "unsafe": "respond_to_unsafe_query_node",
        "safe": "rewrite_query_node"
    },
)

graph_builder.add_edge("rewrite_query_node", "is_other_carrier_query_node")

graph_builder.add_conditional_edges(
    "is_other_carrier_query_node",
    search_branch2,
    path_map = {
        "other_carrier": "respond_to_other_carrier_query_node",
        "lg_uplus": "is_rateplan_related_query_node"
    },
)

graph_builder.add_conditional_edges(
    "is_rateplan_related_query_node",
    search_branch3,
    path_map = {
        "rateplan_unrelated": "respond_to_unrelated_rateplan_query_node",
        "rateplan_related": "is_recommendation_intent_node"
    },
)

graph_builder.add_conditional_edges(
    "is_recommendation_intent_node",
    search_branch4,
    path_map = {
        "non_recommendation_intent": "respond_to_non_recommendation_intent_node",
        "recommendation_intent": "respond_to_recommendation_intent_node"
    },
)

graph_builder.add_edge("respond_to_unsafe_query_node", END)
graph_builder.add_edge("respond_to_other_carrier_query_node", END)
graph_builder.add_edge("respond_to_unrelated_rateplan_query_node", END)
graph_builder.add_edge("respond_to_non_recommendation_intent_node", END)
graph_builder.add_edge("respond_to_recommendation_intent_node", END)

# -------- 그래프 완성 -------
ufit_graph = graph_builder.compile()


visualize_graph(ufit_graph)
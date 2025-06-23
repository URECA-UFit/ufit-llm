import os,json
from ufit.services.user_service import stringify_user_info, stringfiy_user_rate_plan
from ufit.dto.user_info import UserInfoDTO
from IPython.display import Image, display
from langchain_community.chat_message_histories import MongoDBChatMessageHistory

from langchain_core.messages import BaseMessage
from langchain_community.vectorstores import PGVector
from langchain.schema import AIMessage
from typing import Annotated, TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from ufit.dto.recommend import PlanDTO
from ufit.llm.llm_model import get_llm_model, embedding_model, get_openai_model
from ufit.llm.ufit_prompt import (
    get_safe_query_prompt,
    get_is_rateplan_related_prompt,
    get_unrelated_rateplan_prompt,
    get_is_recommendation_intent_prompt,
    get_recommendation_prompt,
    get_non_recommendation_prompt,
    get_other_carrier_prompt,
    get_rewrite_query_prompt
)

PGVECTOR_CONNECTIONS_STRING = os.getenv("PGVECTOR_CONNECTIONS_STRING")
collection_name = "plans"

vectorstore = PGVector(
    embedding_function = embedding_model,
    collection_name = collection_name,
    connection_string = PGVECTOR_CONNECTIONS_STRING
)

NUM_OF_RECOMMEND_PLAN = 2
MAX_TURNS = 5

respond_to_unsafe_query = "죄송합니다. 해당 요청은 서비스 이용 정책에 따라 처리할 수 없습니다.\n다른 질문을 해주세요."
respond_to_other_carrier_query = "죄송합니다. 저는 LG U+ 요금제에 한해 상담을 제공하고 있습니다.\n타 통신사 관련 문의는 답변드릴 수 없습니다."

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    history: MongoDBChatMessageHistory
    content: str
    rewriten_content: str
    is_safe: bool
    is_other_carrier: bool
    is_rateplan_related: bool
    is_recommendation_intent: bool
    is_my_recommend: bool
    user_info: UserInfoDTO
    a_plan: PlanDTO
    b_plan: PlanDTO
    answer: str

def is_safe_query_node(state: State):

    prompt = get_safe_query_prompt(input=state["content"])
    response = get_llm_model(temperature=0.0, max_token=100).invoke(prompt.to_messages())
    
    try:
        result = json.loads(response.content)
        is_safe = result["is_safe"]
    except Exception as e:
        is_safe = False

    return {
        "is_safe": is_safe
    }

def respond_to_unsafe_query_node(state: State):
    result = respond_to_unsafe_query
    return {
        "answer": result,
        "messages": [AIMessage(content=result)]
    }

def is_other_carrier_query_node(state: State):
    content = state["rewriten_content"]

    prompt = get_other_carrier_prompt(content)
    response = get_llm_model(temperature=0.0, max_token=100).invoke(prompt.to_messages())

    try:
        result = json.loads(response.content)
        is_other_carrier = result["is_other_carrier"]
    except Exception:
        is_other_carrier = False 

    return {
        "is_other_carrier": is_other_carrier
    }

def respond_to_other_carrier_query_node(state: State):
    return {
        "answer": respond_to_other_carrier_query,
        "messages": [AIMessage(content=respond_to_other_carrier_query)]
    }

def rewrite_query_node(state: State):
    prompt = get_rewrite_query_prompt(state["messages"][-5:],state["content"])
    response = get_llm_model(temperature=0.0, max_token=500).invoke(prompt)
    
    rewriten_content = response.content
    return {
        "rewriten_content": rewriten_content
    }

def is_rateplan_related_query_node(state: State):
    content = state["rewriten_content"]
    
    prompt = get_is_rateplan_related_prompt(content)
    response = get_llm_model(temperature=0.0, max_token=100).invoke(prompt.to_messages())
    
    try:
        result = json.loads(response.content)
        is_related = result["is_rateplan_related"]
    except Exception:
        is_related = False  

    return {
        "is_rateplan_related": is_related
    }

def respond_to_unrelated_rateplan_query_node(state: State):
    prompt = get_unrelated_rateplan_prompt(state["rewriten_content"])

    response = get_llm_model(temperature=0.5, max_token=800).invoke(prompt)

    result = response.content
    
    return {
        "answer": result,
        "messages": [AIMessage(content=result)]
    }

def is_recommendation_intent_node(state: State):
    prompt = get_is_recommendation_intent_prompt(state["rewriten_content"])
    response = get_llm_model(temperature=0.0, max_token=100).invoke(prompt.to_messages())
    try:
        result = json.loads(response.content)
        is_recommendation_intent = result.get("is_recommendation_intent", False)
        is_my_recommend = result.get("is_my_recommend", False)
    except Exception:
        is_recommendation_intent = False  


    return {
        "is_recommendation_intent": is_recommendation_intent,
        "is_my_recommend": is_my_recommend
    }

def respond_to_non_recommendation_intent_node(state: State):
    prompt = get_non_recommendation_prompt(state["rewriten_content"])
    response = get_llm_model(temperature=0.05, max_token=1000).invoke(prompt.to_messages())
    content = response.content
    
    state["history"].add_user_message(state["content"])
    state["history"].add_ai_message(response.content)

    return {
        "answer": content,
        "message": [AIMessage(content=content)]
    }

def extract_plan_dto(doc, default_name):
    metadata = doc.metadata or {}
    return PlanDTO(
        planId=metadata.get("mongo_id", f"요금제ID {default_name}"),
        name=metadata.get("plan_name", f"요금제 {default_name}")
    )

def respond_to_recommendation_intent_node(state: State):
    
    if state["user_info"] is None:
        user_text = "비회원"
        user_rate_plan_text = "비회원"
    else:
        user_text = stringify_user_info(state["user_info"])
        user_rate_plan_text = stringfiy_user_rate_plan(state["user_info"])

    if state["is_my_recommend"] and state["user_info"] is not None:
        retriever_text = f"{state['rewriten_content']} ({user_text})"
    else:
        retriever_text = state["rewriten_content"]

    docs = vectorstore.similarity_search(retriever_text, NUM_OF_RECOMMEND_PLAN)
    
    plan_texts = "\n\n".join(
    f"요금제 {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

    prompt = get_recommendation_prompt(user_text, user_rate_plan_text, plan_texts, state["rewriten_content"])

    response = get_llm_model(temperature=0.2, max_token=2000).invoke(prompt.to_messages())
    
    a_plan = extract_plan_dto(docs[0], "없음") if len(docs) > 0 else PlanDTO(planId="", name="")
    b_plan = extract_plan_dto(docs[1], "없음") if len(docs) > 1 else PlanDTO(planId="", name="")

    
    state["history"].add_user_message(state["content"])

    return {
        "answer": response.content,
        "a_plan": a_plan,
        "b_plan": b_plan,
        "messages": [AIMessage(content=response.content)],
    }


graph_builder = StateGraph(State)

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

graph_builder.add_edge(START, "is_safe_query_node")

graph_builder.add_conditional_edges(
    "is_safe_query_node",
    search_branch1,
    path_map={
        "unsafe": "respond_to_unsafe_query_node",
        "safe": "is_other_carrier_query_node"
    },
)

graph_builder.add_conditional_edges(
    "is_other_carrier_query_node",
    search_branch2,
    path_map={
        "other_carrier": "respond_to_other_carrier_query_node",
        "lg_uplus": "rewrite_query_node"
    },
)

graph_builder.add_edge("rewrite_query_node", "is_rateplan_related_query_node")

graph_builder.add_conditional_edges(
    "is_rateplan_related_query_node",
    search_branch3,
    path_map={
        "rateplan_unrelated": "respond_to_unrelated_rateplan_query_node",
        "rateplan_related": "is_recommendation_intent_node"
    },
)

graph_builder.add_conditional_edges(
    "is_recommendation_intent_node",
    search_branch4,
    path_map={
        "non_recommendation_intent": "respond_to_non_recommendation_intent_node",
        "recommendation_intent": "respond_to_recommendation_intent_node"
    },
)

graph_builder.add_edge("respond_to_unsafe_query_node", END)
graph_builder.add_edge("respond_to_other_carrier_query_node", END)
graph_builder.add_edge("respond_to_unrelated_rateplan_query_node", END)
graph_builder.add_edge("respond_to_non_recommendation_intent_node", END)
graph_builder.add_edge("respond_to_recommendation_intent_node", END)

ufit_graph = graph_builder.compile()

display(Image(ufit_graph.get_graph().draw_mermaid_png()))
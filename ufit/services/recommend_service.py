import os

from ufit.services.user_service import get_user_full_info
from sqlalchemy.orm import Session
from pymongo.database import Database
from langchain_mongodb import MongoDBChatMessageHistory
from ufit.database.database import save_chat_bot_message

from ufit.dto.recommend import AnswerType, RecommendResponse, PlanDTO
from ufit.llm.ufit_graph import State, ufit_graph

def run_ufit_graph(
    user_id: int,
    content: str,
    chat_room_id: int,
    postgre_db: Session,
    mongo_db: Database,
):
    # run graph
    session_id = str(chat_room_id)
    
    history = MongoDBChatMessageHistory(
        connection_string=os.getenv("MONGO_URI"),
        session_id=session_id,
        database_name=mongo_db.name,
        collection_name="message_histories",
    )

    user_info = get_user_full_info(user_id, postgre_db, mongo_db)

    initial_state: State = {
        "messages": history.messages,
        "content": content,
        "rewriten_content": content,
        "is_safe": False,
        "is_rateplan_related": False,
        "is_recommendation_intent": False,
        "user_info": user_info,
        "a_plan": PlanDTO(planId="", name=""),
        "b_plan": PlanDTO(planId="", name=""),
        "answer": "",
    }

    state = ufit_graph.invoke(initial_state)
    history.add_user_message(state["content"])
    history.add_ai_message(state["answer"])

    # response 가공
    chat_bot_messages = mongo_db.get_collection("chat_bot_messages")
    save_chat_bot_message(
        collection=chat_bot_messages,
        content=state["content"],
        owner=True,
        chat_room_id=chat_room_id
    )
    recommend_plans = []
    a_plan = state["a_plan"]
    b_plan = state["b_plan"]

    if a_plan.planId and a_plan.name:
        recommend_plans.append({"planId": a_plan.planId, "name": a_plan.name})
    if b_plan.planId and b_plan.name:
        recommend_plans.append({"planId": b_plan.planId, "name": b_plan.name})


    message_id = save_chat_bot_message(
        collection=chat_bot_messages,
        content=state["answer"],
        owner=False,
        chat_room_id=chat_room_id,
        recommend_plan=recommend_plans if recommend_plans else None
    )

    a_plan_id = state["a_plan"].planId
    b_plan_id = state["b_plan"].planId

    if not a_plan_id and not b_plan_id:
        answer_type = AnswerType.GENERAL
    else:
        answer_type = AnswerType.RECOMMEND

    print(state["answer"])

    answer = RecommendResponse(
        messageId=str(message_id),
        answer=state["answer"],
        answerType=answer_type,
        recommendPlans=[state["a_plan"], state["b_plan"]],
    )
    return answer

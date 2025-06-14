import os
from typing import List
from pymongo.database import Database
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from dotenv import load_dotenv
from bson.objectid import ObjectId

# 환경 변수 로드 및 LLM 세팅
load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise ValueError("CLAUDE_API_KEY is not set")
os.environ["ANTHROPIC_API_KEY"] = CLAUDE_API_KEY

llm_model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0.7
)


def get_user_queries_after_recommendation(mongo_db: Database, chat_room_id: int, recommendation_message_id_str: str, limit: int = 10) -> List[str]:
    """
    특정 추천 메시지 ID부터 미래(더 최신)로 올라가면서, 다음 챗봇 추천 메시지 전까지의 유저 메시지 최신 10개 추출.
    """
    print(f"[DEBUG] get_user_queries_after_recommendation 호출. chatRoom ID: {chat_room_id}, 시작 메시지 ID: {recommendation_message_id_str}")
    
    try:
        recommendation_obj_id = ObjectId(recommendation_message_id_str)
    except Exception as e:
        print(f"[DEBUG] 유효하지 않은 ObjectId: {e}")
        return []

    # 1. 해당 채팅방의 메시지 중, 시작점 메시지보다 이후이거나 같은 메시지를 _id 오름차순으로 가져옴
    # 이렇게 하면 시작 메시지부터 미래(더 최신)로 순회할 수 있음.
    messages_cursor = mongo_db.chat_bot_messages.find(
        {"chatRoom_id": chat_room_id, "_id": {"$gte": recommendation_obj_id}}
    ).sort("_id", 1) # 오름차순 정렬 (오래된 것부터 최신으로)
    
    messages = list(messages_cursor)
    print(f"[DEBUG] Raw messages from DB (시작점 포함, {len(messages)}개): ")
    for i, msg_item in enumerate(messages):
        print(f"  [DEBUG] {i+1}. {msg_item}")

    print("[DEBUG] === 메시지 필터링 시작 (시작 추천 메시지 이후 유저 질의 수집) ===")
    user_msgs_to_summarize = []
    found_start_recommendation = False

    for msg in messages:
        msg_content = msg.get("content", "")
        msg_owner = msg.get("owner", False)
        is_recommendation_msg = (not msg_owner and "추천" in msg_content)
        
        # 시작 메시지 ID를 찾았을 때부터 실제 필터링 시작
        if msg["_id"] == recommendation_obj_id:
            found_start_recommendation = True
            print(f"[DEBUG] 시작점 추천 메시지 발견! 내용: '{msg_content}'")
            continue # 이 메시지는 건너뛰고 다음 메시지부터 수집
        
        # 시작점 추천 메시지를 아직 찾지 못했으면 무시 (이전 메시지들)
        if not found_start_recommendation:
            print(f"[DEBUG] 시작점 메시지 이전에 발견된 메시지 (무시): Owner={msg_owner}, Content='{msg_content}'")
            continue

        print(f"[DEBUG] 현재 메시지 처리: Owner={msg_owner}, Content='{msg_content}', IsRecommendation='{is_recommendation_msg}'")

        # 챗봇 메시지에서 다음 추천 키워드가 있으면 순회 중단
        if is_recommendation_msg:
            print(f"[DEBUG] >>> 다음 추천 메시지 발견! 순회 중단. 내용: '{msg_content}'")
            break
        
        # 유저 메시지면 리스트에 추가
        if msg_owner:
            user_msgs_to_summarize.append(msg_content)
            print(f"[DEBUG] >>> 유저 메시지 추가: '{msg_content}', 현재 유저 메시지 개수: {len(user_msgs_to_summarize)}")
        
        if len(user_msgs_to_summarize) == limit:
            print(f"[DEBUG] >>> 유저 메시지 {limit}개 도달, 순회 중단.")
            break
            
    print("[DEBUG] === 메시지 필터링 종료 ===")
    print(f"[DEBUG] 최종 필터링된 유저 메시지 개수: {len(user_msgs_to_summarize)}")
    print(f"[DEBUG] 최종 필터링된 유저 메시지:\n{user_msgs_to_summarize}")
    
    # 수집된 메시지는 이미 오래된 것 -> 최신 순서 (오름차순 순회) 이므로 그대로 반환
    return user_msgs_to_summarize # 최종적으로 과거→최신 순서로 반환


def summarize_user_queries_with_llm(user_msgs: List[str]) -> str:
    """
    유저 메시지 리스트를 LLM(클로드)로 요약
    """
    if not user_msgs:
        return "요약할 유저 메시지가 없습니다."
    
    print(f"[DEBUG] 요약할 유저 메시지 ({len(user_msgs)}개): {user_msgs}")
    
    joined = "\n".join(user_msgs)
    
    print(f"[DEBUG] LLM 입력 Joined String:\n{joined}")
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "당신은 사용자의 대화 내용을 분석하여, 마치 사용자가 직접 자신의 통신 성향을 설명하는 것처럼 간결하고 자연스러운 하나의 문단으로 요약합니다. 대화체나 챗봇의 말투, 설명조의 표현을 사용하지 마세요. 오직 사용자의 관점에서 흐름이 자연스러운 요약 내용만을 반환합니다. 한국어로 답변하세요. 불법적이거나 폭력적인 질문에는 답변하지 마세요. **중요: 입력 데이터에 있는 내용만을 요약하고, 추측하지 마세요. '사용자는~', '따라서~'와 같은 설명조 표현을 사용하지 마세요.**"
        ),
        HumanMessagePromptTemplate.from_template(
            "아래는 사용자의 최근 대화 기록입니다:\n\n{user_queries}\n\n이 대화를 바탕으로 사용자가 어떤 통신 성향을 가지고 있는지, 마치 사용자가 자신의 성향을 한 번에 설명하는 것처럼 자연스러운 하나의 문단으로 요약해주세요. 예를 들어, '저는 평소 유튜브 시청이 많아서 데이터를 자주 사용하고, 통화량은 많지 않은 편입니다.'와 같이 부드러운 연결을 가진 문장을 사용해야 합니다. **입력 데이터에 있는 내용만을 요약하고, 추측하지 마세요. '사용자는~', '따라서~'와 같은 설명조 표현을 사용하지 마세요.**"
        ),
    ])
    rendered_msgs = prompt.format_prompt(user_queries=joined).to_messages()
    
    print(f"[DEBUG] LLM에 전달될 최종 메시지:\n{rendered_msgs}")
    
    response_msg = llm_model.invoke(rendered_msgs)
    
    print(f"[DEBUG] LLM 원시 응답:\n{response_msg}")
    
    return response_msg.content.strip()

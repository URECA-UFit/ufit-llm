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
    try:
        recommendation_obj_id = ObjectId(recommendation_message_id_str)
    except Exception as e:
        return []

    messages_cursor = mongo_db.chat_bot_messages.find(
        {"chatRoom_id": chat_room_id, "_id": {"$gte": recommendation_obj_id}}
    ).sort("_id", 1)
    
    messages = list(messages_cursor)
    for i, msg_item in enumerate(messages):
        print(f"  [DEBUG] {i+1}. {msg_item}")

    user_msgs_to_summarize = []
    found_start_recommendation = False

    for msg in messages:
        msg_content = msg.get("content", "")
        msg_owner = msg.get("owner", False)
        is_recommendation_msg = (not msg_owner and "추천" in msg_content)
        
        if msg["_id"] == recommendation_obj_id:
            found_start_recommendation = True
            continue
        
        if not found_start_recommendation:
            continue

        if is_recommendation_msg:
            break
        
        if msg_owner:
            user_msgs_to_summarize.append(msg_content)
        
        if len(user_msgs_to_summarize) == limit:
            break
            
    return user_msgs_to_summarize


def summarize_user_queries_with_llm(user_msgs: List[str]) -> str:
    if not user_msgs:
        return "요약할 유저 메시지가 없습니다."
    
    joined = "\n".join(user_msgs)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "당신은 사용자의 대화 내용을 분석하여, 마치 사용자가 직접 자신의 통신 성향을 설명하는 것처럼 간결하고 자연스러운 하나의 문단으로 요약합니다. 대화체나 챗봇의 말투, 설명조의 표현을 사용하지 마세요. 오직 사용자의 관점에서 흐름이 자연스러운 요약 내용만을 반환합니다. 한국어로 답변하세요. 불법적이거나 폭력적인 질문에는 답변하지 마세요. **중요: 입력 데이터에 있는 내용만을 요약하고, 추측하지 마세요. '사용자는~', '따라서~'와 같은 설명조 표현을 사용하지 마세요.**"
        ),
        HumanMessagePromptTemplate.from_template(
            "아래는 사용자의 최근 대화 기록입니다:\n\n{user_queries}\n\n이 대화를 바탕으로 사용자가 어떤 통신 성향을 가지고 있는지, 마치 사용자가 자신의 성향을 한 번에 설명하는 것처럼 자연스러운 하나의 문단으로 요약해주세요. 예를 들어, '저는 평소 유튜브 시청이 많아서 데이터를 자주 사용하고, 통화량은 많지 않은 편입니다.'와 같이 부드러운 연결을 가진 문장을 사용해야 합니다. **입력 데이터에 있는 내용만을 요약하고, 추측하지 마세요. '사용자는~', '따라서~'와 같은 설명조 표현을 사용하지 마세요.**"
        ),
    ])
    rendered_msgs = prompt.format_prompt(user_queries=joined).to_messages()
    
    response_msg = llm_model.invoke(rendered_msgs)
    
    return response_msg.content.strip()

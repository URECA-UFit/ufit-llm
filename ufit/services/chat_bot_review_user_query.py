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
    temperature=0.3
)

# 컬렉션 이름 상수 정의
CHAT_BOT_MESSAGES_COLLECTION = "chat_bot_messages"

def get_user_queries_after_recommendation(mongo_db: Database, chat_room_id: int, recommendation_message_id_str: str, limit: int = 10) -> List[str]:
    try:
        recommendation_obj_id = ObjectId(recommendation_message_id_str)
    except Exception as e:
        return []

    messages_cursor = mongo_db[CHAT_BOT_MESSAGES_COLLECTION].find(
        {"chat_room_id": chat_room_id, "_id": {"$lte": recommendation_obj_id}}
    ).sort("_id", -1)
    
    messages = list(messages_cursor)
    user_msgs_to_summarize = []
    found_start_recommendation = False

    for idx, msg in enumerate(messages):
        msg_content = msg.get("content", "")
        msg_owner = msg.get("owner", False)

        if msg["_id"] == recommendation_obj_id:
            found_start_recommendation = True
            continue
        
        if not found_start_recommendation:
            continue

        # break 조건: 챗봇의 RECOMMENDATION_LIST만
        is_previous_bot_recommendation = not msg_owner and "RECOMMENDATION_LIST" in msg_content
        if is_previous_bot_recommendation:
            break
        
        if msg_owner:
            user_msgs_to_summarize.append(msg_content)
        
        if len(user_msgs_to_summarize) == limit:
            break
            
    print(f"[요약 전] LLM에 전달될 메시지 리스트: {user_msgs_to_summarize}", flush=True)
    return user_msgs_to_summarize


def summarize_user_queries_with_llm(user_msgs: List[str]) -> str:
    if not user_msgs:
        return "요약할 유저 메시지가 없습니다."
    
    joined = "\n".join(user_msgs)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "당신은 사용자의 대화 내용을 분석하여, 마치 사용자가 직접 자신의 통신 성향을 설명하는 것처럼 간결하고 자연스러운 하나의 문단으로 요약합니다. 대화체나 챗봇의 말투, 설명조의 표현을 사용하지 마세요. 오직 사용자의 관점에서 흐름이 자연스러운 요약 내용만을 반환합니다. 한국어로 답변하세요. 불법적이거나 폭력적인 질문에는 답변하지 마세요. **중요: 입력 데이터에 있는 내용만을 요약하고, 추측하지 마세요. '사용자는~', '따라서~'와 같은 설명조 표현을 사용하지 마세요.**"
            "[예시]"
            "[질문1] 나는 유튜브를 많이 봐"
            "[질문2] 나는 데이터가 좀 부족해"
            "[질문3] 현재 데이터가 30GB인 요금제를 쓰고 있어"
            "[질문4] 나에게 맞는 요금제 추천해줘"
            "[답변] 유튜브를 많아봐서 데이터가 부족해. 현재 요금제는 30GB 데이터를 쓸 수 있는데 이 요금제 말고 알맞는 요금제를 추천해줘\n"
            "예시처럼 답변하세요"
        ),
        HumanMessagePromptTemplate.from_template(
            "아래는 사용자의 최근 대화 기록입니다:\n\n{user_queries}\n\n이 대화를 바탕으로 사용자가 어떤 통신 성향을 가지고 있는지, 마치 사용자가 자신의 성향을 한 번에 설명하는 것처럼 자연스러운 하나의 문단으로 요약해주세요. **입력 데이터에 있는 내용만을 요약하고, 추측하지 마세요. '사용자는~', '따라서~'와 같은 설명조 표현을 사용하지 마세요.**"
            "반드시 요금제 또는 통신 성향 관련 질문만 요약하세요. 그렇지 않은 질문은 요약하지마세요."
            "[예시1]"
            "[질문1] 나는 멍청해"
            "[질문2] 나는 데이터를 많이 써"
            "[질문3] 오늘 날씨가 좋다"
            "[질문4] 5만원 이하 요금제 추천해줘"
            "[답변] 데이터를 많이 쓰는 편이야. 5만원 이하로 알맞은 요금제를 추천해줘\n"
            "반드시 예시와 비슷하게 답변하세요"
            "[예시]"
            "[질문1] 나는 유튜브를 많이 봐"
            "[질문2] 나는 데이터가 좀 부족해"
            "[질문3] 현재 데이터가 30GB인 요금제를 쓰고 있어"
            "[질문4] 나에게 맞는 요금제 추천해줘"
            "[답변] 유튜브를 많아봐서 데이터가 부족해. 현재 요금제는 30GB 데이터를 쓸 수 있는데 이 요금제 말고 알맞는 요금제를 추천해줘\n"
            "예시처럼 답변하세요"
        ),
    ])
    rendered_msgs = prompt.format_prompt(user_queries=joined).to_messages()
    
    response_msg = llm_model.invoke(rendered_msgs)
    
    return response_msg.content.strip()
from langchain.schema import HumanMessage
from ufit.services.llm_answer_agent import get_prompt, get_llm_response

# 테스트용 질문들 (같은 의미, 다른 말투)
test_queries = [
    "데이터 많이 쓰는데 요금제 뭐가 좋을까요?",
    "나 데이터 많이 씀. 뭐 쓰면 됨?",
    "저는 데이터를 자주 사용하는 편인데 추천해주실 수 있나요?",
    "매일 영상 보는데요, 어떤 요금제가 좋습니까?",
    "영상 시청 많이 해요. 가성비 요금제 추천해줘.",
    "나한테 맞는 요금제 알려줄래요?"
]

# 사용자/요금제 정보는 고정 
dummy_user_block = """
다음은 사용자 정보입니다:
- 이메일: test@lg.com
- 나이: 25
- 성별: 남성
- 현재 요금제: 5G 라이트+
- 기본요금: 45000원
"""

# 후보 요금제 정보도 고정
dummy_retrieved_block = """
- 5G 프리미어 에센셜 요금제: 월 85,000원에 150GB + 5Mbps 속도제한 (유사도: 0.82)
- 5G 슬림+ 요금제: 월 45,000원에 10GB + 1Mbps 속도제한 (유사도: 0.79)
"""

# 테스트 실행
for query in test_queries:
    messages = get_prompt(
        user_block=dummy_user_block,
        retrieved_block=dummy_retrieved_block,
        base_prompt=query
    )

    response = get_llm_response(messages)
    
    print(f"🧾 질문: {query}")
    print("📨 응답:")
    print(response)
    print("-" * 80)

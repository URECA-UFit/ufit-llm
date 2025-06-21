import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from formatter import generate_final_output
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise ValueError("CLAUDE_API_KEY is not set")

# Claude LLM 인스턴스
llm_model = ChatAnthropic(
    api_key=CLAUDE_API_KEY,
    model="claude-3-haiku-20240307",
    temperature=0
)

# 메타데이터 추출 프롬프트 (benefit_keywords 하나로 합침)
EXTRACT_METADATA_PROMPT = """
아래 데이터에서 메타데이터(필드명:값)를 JSON 형태로 추출해줘.
필드명은 반드시 다음 중에서만 선택: social_category, data_category, device_type, data_sharing, benefit_keywords
- data_category는 해당 필드가 없으면 무조건 빈 문자열로, 값이 있다면 반드시 아래 값 중 하나만 저장해줘. (그 외 값이 들어가면 빈 문자열로 저장)
  - 'web, kakaotalk'
  - 'web, kakaotalk, music'
  - 'web, kakaotalk, music, video, game'
- data_sharing은 값에 '가능'이라는 단어가 들어가면 '가능'으로, '불가능'은 그대로 '불가능'으로, 그 외는 빈 문자열로 저장해줘.
- benefit_keywords는 basic_benefit, discount_benefit, special_benefit에서 추출한 모든 주요 혜택 키워드를 하나의 리스트로 합쳐서 저장해줘. (예: [\"로밍 혜택\", \"월정액 할인\"])
- 없는 값은 빈 값으로 넣어줘.
데이터:
{data}
"""

def extract_metadata_with_claude(plan: dict) -> dict:
    prompt = EXTRACT_METADATA_PROMPT.format(data=json.dumps(plan, ensure_ascii=False, indent=2))
    messages = [HumanMessage(content=prompt)]
    response = llm_model.invoke(messages).content.strip()
    # JSON만 추출
    import re
    import json as pyjson
    try:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            metadata = pyjson.loads(match.group(0))
        else:
            metadata = pyjson.loads(response)
    except Exception:
        metadata = {}
    return metadata

# plans_0619.json 경로
file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "plans_0619.json")
)

# json 파일을 문서로 변환
with open(file_path, "r", encoding="utf-8") as f:
    plans = json.load(f)

docs = []
for plan in plans:
    metadata = extract_metadata_with_claude(plan)
    # mongo_id와 plan_name 추가
    metadata["mongo_id"] = plan["_id"]
    metadata["plan_name"] = plan["plan_name"]
    docs.append(
        Document(
            page_content=generate_final_output(plan),
            metadata=metadata
        )
    )

# 전체 메타데이터 집합 추출 및 저장
all_metadata = {}
for doc in docs:
    for k, v in doc.metadata.items():
        if isinstance(v, list):
            all_metadata.setdefault(k, set()).update(v)
        else:
            all_metadata.setdefault(k, set()).add(v)
# set을 list로 변환 (None은 빈 문자열로 변환)
all_metadata = {
    k: sorted("" if v is None else v for v in v_set)
    for k, v_set in all_metadata.items()
}
with open(os.path.join(os.path.dirname(file_path), "metadata_keys.json"), "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)


# 임베딩 모델 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# PGVector 연결 설정
PGVECTOR_CONNECTIONS_STRING = os.getenv("PGVECTOR_CONNECTIONS_STRING")
collection_name = "plans"

vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embedding_model,
    connection_string=PGVECTOR_CONNECTIONS_STRING,
    collection_name=collection_name,
)

print("embedding success")

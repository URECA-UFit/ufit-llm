# <history_aware_chain>
# 사용자와의 대화 이력을 바탕으로,
# "현재 질문을 이해하고 검색할 수 있는 형태의 쿼리로 변환"하여
# 벡터 검색기(retriever)에 넘겨줄 수 있도록 LLM에 프롬프트를 구성하는 것.
# 멀티턴 성능 향상을 위해 사용한다.

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ufit.services.embedding import vectorstore
from ufit.llm.ai_model import get_anthropic_model, get_openai_model

retriever = vectorstore.as_retriever

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', "{input}")
    ("user", "Given the above conversation, generate a search query to look up information relevant to the last question.")
]
)

history_aware_chain = create_history_aware_retriever(
    llm=get_openai_model(temperature=0.2,max_token=800),
    retriever=retriever,
    prompt = prompt
)


from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.chain.tools import get_live_race, get_past_race, get_round_race, search_regulations

tools = [
    search_regulations,
    get_live_race,
    get_past_race,
    get_round_race
]

prompt = ChatPromptTemplate.from_messages([
    ('system', """당신은 f1 전문 챗봇입니다.
     사용자의 질문에 따라 적절한 도구를 선택해 답변하세요.
     답변은 한국어로만 하세요.
     규정 관련: search_regulations
     - 현재 실시간 레이스 관련: get_live_race
     - 과거 시즌 기록: get_past_race
     - 특정 라운드 기록: get_round_race
     """),
     MessagesPlaceholder(variable_name="chat_history", optional=True),
     ("human", "{input}"),
     MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-4o-mini") # 임의로 선정
agent = create_openai_functions_agent(llm, tools, prompt)
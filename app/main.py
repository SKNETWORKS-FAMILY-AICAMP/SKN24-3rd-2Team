# app/main.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from src.chain.agent import agent, tools


def run():
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    chat_history = []
    print("🏎️  F1 챗봇 시작 (종료: exit)\n")

    while True:
        query = input("질문: ").strip()

        if not query:
            continue
        if query.lower() == "exit":
            print("종료합니다.")
            break

        result = agent_executor.invoke({
            "input": query,
            "chat_history": chat_history,
        })

        answer = result["output"]
        print(f"\n답변: {answer}\n")

        # 대화 히스토리 누적
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))


if __name__ == "__main__":
    run()
from agent_graph.graph import graph
from agent_graph.state import AgentState

def main():

    # 사용자 입력 받기
    user_question = input("사용자 질문을 입력하세요: ").strip()
    input_state = AgentState(

        # Input
        question_id = "USER1",
        question_text = user_question,

    )

    # Graph 실행
    output_state = graph.invoke(input_state)

    # 결과 출력
    print("Graph 실행 결과:")
    print(output_state["answer"])
    print("인용 정보:")
    print(output_state["citations"])


if __name__ == "__main__":
    main()

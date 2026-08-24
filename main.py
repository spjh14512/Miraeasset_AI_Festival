from agent_graph.graph import graph
from agent_graph.state import AgentState
from agent_graph.tools import format_citations

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
    print(output_state["ai_answer"].answer)
    print("인용 정보:")
    for citation in format_citations(output_state["ai_answer"].citation):
        print("- " + citation)


if __name__ == "__main__":
    main()

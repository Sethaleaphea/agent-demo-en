import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from demo_re import run

# export LANG=en_US.UTF-8
# export LC_ALL=en_US.UTF-8

# 加载环境变量
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

# 初始化 LLM
chatLLM = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="deepseek-r1",
)

# 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]
    node_name: str


# 意图识别函数
def stu_intent(llm, student_msg: str):
    messages = [
        SystemMessage(f'''你是一名教师，你正在与学生交流，你需要根据学生最新的一句话判断其意图：
                        1. 如果学生在询问问题，希望获得解答，则输出“答疑”。
                        2. 如果学生是在提供题目或希望系统生成题目，则输出“生题”。
                        3. 以上两种均不属于输出“闲聊”。
                        学生最新一句话：{student_msg}'''),
    ]
    return llm.invoke(messages).content

# 答疑函数
def qa(llm, student_msg: str):
    """
    答疑函数，调用 demo_re.py 的 run 函数处理用户输入。
    """
    run(student_msg)
    return "答疑完成"


# 出题函数
def qg(llm, student_msg: str):
    """
    出题函数，调用出题系统
    """
    return "生题完成"

# 闲聊
def chat(llm, student_msg: str):
    """
    闲聊函数，直接调用大模型生成回复
    """
    messages = [
        SystemMessage(content="你是一个友好的助手，负责与学生进行日常闲聊。请根据学生的输入，生成自然、友好的回复。"),
        HumanMessage(content=student_msg)
    ]
    response = llm.invoke(messages)
    return response.content
    # return "闲聊完成"

# 构建图
def GlobalGraph():
    graph_builder = StateGraph(State)

    # 意图识别 Agent
    def stu_intent_agent(state: State):
        student_msg = state["messages"][-1].content  # 获取学生最新消息
        intent = stu_intent(chatLLM, student_msg)
        return {"messages": [AIMessage(content=intent)], "node_name": "stu_intent_agent"}

    # 答疑 Agent
    def qa_agent(state: State):
        student_msg = state["messages"][-1].content
        return {"messages": [AIMessage(content=qa(chatLLM, student_msg))], "node_name": "qa_agent"}

    # 出题 Agent
    def qg_agent(state: State):
        student_msg = state["messages"][-1].content
        return {"messages": [AIMessage(content=qg(chatLLM, student_msg))], "node_name": "qg_agent"}

    # 闲聊 Agent
    def chat_agent(state: State):
        student_msg = state["messages"][-1].content
        return {"messages": [AIMessage(content=chat(chatLLM, student_msg))], "node_name": "chat_agent"}

    # 添加节点
    graph_builder.add_node("stu_intent_agent", stu_intent_agent)  # 意图识别
    graph_builder.add_node("qa_agent", qa_agent)  # 答疑
    graph_builder.add_node("qg_agent", qg_agent)  # 出题
    graph_builder.add_node("chat_agent", chat_agent)  # 闲聊

    # 添加边
    graph_builder.add_edge(START, "stu_intent_agent")

    # 路由逻辑
    def router(state: State):
        intent = state["messages"][-1].content  # 获取意图识别的输出
        if intent == "答疑":
            return "qa_agent"
        elif intent == "生题":
            return "qg_agent"
        else:
            return "chat_agent"

    graph_builder.add_conditional_edges(
        "stu_intent_agent",
        router,
        {"qa_agent": "qa_agent", "qg_agent": "qg_agent","chat_agent": "chat_agent"}
    )

    graph_builder.add_edge("qa_agent", END)
    graph_builder.add_edge("qg_agent", END)
    graph_builder.add_edge("chat_agent", END)

    global_graph = graph_builder.compile()

    def draw_graph():
        from IPython.display import Image, display
        img_data = global_graph.get_graph().draw_mermaid_png()
        with open("global_graph.png", "wb") as f:
            f.write(img_data)
        display(Image(img_data))
    # draw_graph()

    return global_graph



if __name__ == "__main__":
    global_graph = GlobalGraph()

    while True:
        user_message = input("请输入：")

        # 初始化输入
        events = global_graph.stream(
            {
                "messages": [AIMessage(content=user_message)],
                "node_name": "__start__",
            },
            stream_mode="values",
        )

        # 遍历每一步的 event
        for event in events:
            if "messages" in event and isinstance(event["messages"], list):
                # 输出当前经过的 agent 名称
                print(f'--- {event["node_name"]} ---')
                # 输出当前 agent 返回的内容
                print(event["messages"][-1].content)






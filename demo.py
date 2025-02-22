from langchain_openai import ChatOpenAI
import os
import json
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
    AIMessage,
)

from rag_demo import RagApplication, ApplicationConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig, DenseRetriever

def init_rag():
    app_config = ApplicationConfig()
    embedding_model_path = "./bge-large-zh-v1.5"
    retriever_config = DenseRetrieverConfig(
        model_name_or_path=embedding_model_path,
        dim=1024,
        index_path='/home/dalhxwlyjsuo/guest/result/indexs/dense_cache/fassis.index')
    app_config.retriever_config = retriever_config
    application = RagApplication(app_config)
    application.init_vector_store()
    return application
rag_tool = init_rag()

def RAG(query: str):
    return rag_tool.get_rag_content(query)

# deepseek-r1
chatLLM = ChatOpenAI(
    api_key="sk-9f91f7f5e8eb4dfabb71c9df5f72e7d2",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="deepseek-r1",
)
# chatLLM = ChatOpenAI(
#     model_name = "gpt-4o", 
#     temperature = 0,
#     openai_api_key = "sk-EQY6oy2D9Brb2IWxymHTnmeFz5unIyobzHtjDgFs2ZwEPjmY",
#     base_url="https://api.chatanywhere.tech/v1"
#     )
# messages = [
#     # SystemMessage(""),
#     SystemMessage('''你是一个教师，你正在与你的学生交流，你需要判断学生的语句中是否存在问题。
#                     你需要判断的问题包括：逻辑错误、事实错误。
#                     逻辑错误：学生的语句中存在逻辑错误，比如矛盾、不合理的推理等。
#                     事实错误：学生的语句中存在事实错误，比如错误的数据、错误的结论、与事实不符的判断等。
#                     如果存在问题，你需要给出合理的建议。
#                     学生最新一句话：你好'''),
# ]
# response = chatLLM.invoke(messages)
# print(response.content)


from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages

# 获取学生画像
def stu_img(llm, history_msg: str, student_id: str):
    if student_id == "A":
        stu_img = "学生A的画像"
    elif student_id == "B":
        stu_img = "学生B的画像"
    else:
        messages = [
            SystemMessage('''你是一个擅长判断学生知识水平的教师，你的任务是根据和学生的历史记录来判断学生的知识水平，并使用100字左右给出你的看法。
                    历史对话记录：''' + history_msg),
        ]
        stu_img = llm.invoke(messages).content
    
    stu_img_content = stu_img
    return stu_img

# check 学生对话
def check_msg(llm, history_msg: str, student_msg: str):
    if history_msg == "":
        messages = [
            SystemMessage('''你是一个教师，你正在与你的学生交流，你需要判断学生的语句中是否存在问题。
                    你需要判断的问题包括：逻辑错误、事实错误。
                    逻辑错误：学生的语句中存在逻辑错误，比如矛盾、不合理的推理等。
                    事实错误：学生的语句中存在事实错误，比如错误的数据、错误的结论、与事实不符的判断等。
                    如果不存在问题，你只需要回复“无问题”。
                    如果存在问题，你需要给出合理的建议。
                    学生最新一句话：''' + student_msg),
        ]
    else:
        messages = [
            SystemMessage('''你是一个教师，你正在与你的学生交流，你需要根据和学生的历史对话，以及当前学生语言来判断，学生的语句中是否存在问题。
                    你需要判断的问题包括：逻辑错误、事实错误。
                    逻辑错误：学生的语句中存在逻辑错误，比如矛盾、不合理的推理等。
                    事实错误：学生的语句中存在事实错误，比如错误的数据、错误的结论、与事实不符的判断等。
                    如果不存在问题，你只需要回复“无问题”。
                    如果存在问题，你需要给出合理的建议。
                    历史对话记录：''' + history_msg + '''学生最新一句话：''' + student_msg),
        ]
    
    return llm.invoke(messages).content

# 问题改写agent
def question_rewrite(llm, history_msg: str, student_msg: str):
    if history_msg == "":
        return student_msg
    messages = [
        SystemMessage('''你是一个语言学家，你需要根据一段历史对话记录和学生最新的一段话，来改写学生的最新提问，确保改写得到的话能表达学生的想法，并且独立于对话记录。
                历史对话记录：''' + history_msg + '''学生最新一句话：''' + student_msg),
    ]
    return llm.invoke(messages).content

# 矛盾改写
def worng_question_rewrite(llm, history_msg: str, student_msg: str, wrong_msg: str):
    # worng_msg = "修改意见"
    # 推测用户意图，解答 返回
    messages = [
        SystemMessage('''你是一个语言学家，你需要根据一段历史对话记录和学生最新的一段话，来改写学生的最新提问，确保改写得到的话能表达学生的想法，并且独立于对话记录。
                历史对话记录：''' + history_msg + '''学生最新一句话：''' + student_msg),
    ]
    return llm.invoke(messages).content

# 路由 意图识别
def route_intent(llm, student_msg: str):
    messages = [
        SystemMessage('''你是一个教师，你正在与你的学生交流，你需要根据当前学生最新一句话按下面的流程判断学生当前意图。
                        1. 判断学生是否在提问，如果不是，输出“日常对话”。
                        2. 判断学生的疑问是否仅局限于知识概念不明概念，如果是，输出“概念性问题”。
                        3. 以上两种均不属于输出“习题解答”。
                        学生最新一句话：''' + student_msg),
    ]
    return llm.invoke(messages).content
    
# 答疑
def answer(llm, student_msg: str, stu_img_content: str):
    messages = [
        SystemMessage('''你是一个教师，你正在与你的学生交流，你需要根据查阅到的资料解答学生的问题。
                分步骤给出解答，确保学生能够理解。
                学生的问题：''' + student_msg),
    ]
    return llm.invoke(messages).content

# 概念性问题回答
def concept_answer(llm, student_msg: str, stu_img_content: str):
    #! 使用RAG获取外部知识
    content = RAG(student_msg)
    messages = [
        SystemMessage('''你是一个教师，你正在与你的学生交流，你需要向学生解答他不懂的概念性问题。
                学生的问题：''' + student_msg),
    ]
    return llm.invoke(messages).content

# 日常对话 引导学生提问
def daily_conversation(llm, student_msg: str, stu_img_content: str):
    messages = [
        SystemMessage("你是一个教师，你正在与你的学生交流，你的任务是对学生的对话进行回应，在你的回复中应该引导学生提出问题。"),
        HumanMessage(student_msg)
    ]
    return llm.invoke(messages).content

# 图的构建
class State(TypedDict):
    messages: Annotated[list, add_messages]
    node_name: str

config = {"configurable": {"thread_id": "1"}}
configA = {"configurable": {"thread_id": "2"}}
configB = {"configurable": {"thread_id": "3"}}

# 学生标签 TODO UI 设置选项 A B
student_id = "A"
# 历史记录 学生对话 自动记录·
history_msg = {"A": "", "B": ""}
user_message = ""
# 改写后的问题
rewrite_user_message = ""
# 学生画像
stu_img_content = ""

def GraphBuilder():
    graph_builder = StateGraph(State)

    def stu_img_agent(state: State):
        return {"messages": [AIMessage(content=stu_img(chatLLM, history_msg[student_id], student_id))], "node_name": "stu_img_agent"}

    def check_msg_agent(state: State):
        return {"messages": [AIMessage(content=check_msg(chatLLM, history_msg[student_id], user_message))], "node_name": "check_msg_agent"}

    def question_rewrite_agent(state: State):
        return {"messages": [AIMessage(content=question_rewrite(chatLLM, history_msg[student_id], user_message))], "node_name": "question_rewrite_agent"}
    
    def wrong_question_rewrite_agent(state: State):
        return {"messages": [AIMessage(content=worng_question_rewrite(chatLLM, history_msg[student_id], user_message, state["messages"][-1].content))],
                "node_name": "wrong_question_rewrite_agent"}

    def route_intent_agent(state: State):
        rewrite_user_message = state["messages"][-1].content
        return {"messages": [AIMessage(content=route_intent(chatLLM, rewrite_user_message))], "node_name": "route_intent_agent"}

    def answer_agent(state: State):
        return {"messages": [AIMessage(content=answer(chatLLM, rewrite_user_message, stu_img_content))], "node_name": "answer_agent"}

    def concept_answer_agent(state: State):
        return {"messages": [AIMessage(content=concept_answer(chatLLM, rewrite_user_message, stu_img_content))], 
                "node_name": "concept_answer_agent"}
    
    def daily_conversation_agent(state: State):
        return {"messages": [AIMessage(content=daily_conversation(chatLLM, rewrite_user_message, stu_img_content))], 
                "node_name": "daily_conversation_agent"}
    
    def add_node():
        graph_builder.add_node("stu_img_agent", stu_img_agent)
        graph_builder.add_node("check_msg_agent", check_msg_agent)
        graph_builder.add_node("question_rewrite_agent", question_rewrite_agent)
        graph_builder.add_node("wrong_question_rewrite_agent", wrong_question_rewrite_agent)
        graph_builder.add_node("route_intent_agent", route_intent_agent)
        graph_builder.add_node("answer_agent", answer_agent)
        graph_builder.add_node("concept_answer_agent", concept_answer_agent)
        graph_builder.add_node("daily_conversation_agent", daily_conversation_agent)

    add_node()

    graph_builder.add_edge(START, "stu_img_agent")
    graph_builder.add_edge("stu_img_agent", "check_msg_agent")

    # 路由 判断 上一步 是否check出问题
    def rewrite_router(state):
        messages = state["messages"]
        last_message = messages[-1]

        if "无问题" in last_message.content:
            return "question_rewrite_agent"
        return "wrong_question_rewrite_agent"
    
    graph_builder.add_conditional_edges(
        "check_msg_agent",
        rewrite_router, 
        {"question_rewrite_agent": "question_rewrite_agent", "wrong_question_rewrite_agent": "wrong_question_rewrite_agent"},
    )
    graph_builder.add_edge("question_rewrite_agent", "route_intent_agent")
    graph_builder.add_edge("wrong_question_rewrite_agent", END)

    def router(state):
        messages = state["messages"]
        last_message = messages[-1]

        if "概念性问题" in last_message.content:
            return "concept_answer_agent"
        elif "日常对话" in last_message.content:
            return "daily_conversation_agent"
        return "answer_agent"

    graph_builder.add_conditional_edges(
        "route_intent_agent",
        router,
        # {"continue": "chart_generator", "call_tool": "call_tool", END: END},
        {"concept_answer_agent": "concept_answer_agent", "daily_conversation_agent": "daily_conversation_agent", "answer_agent": "answer_agent"},
    )

    graph_builder.add_edge("concept_answer_agent",END)
    graph_builder.add_edge("daily_conversation_agent",END)
    graph_builder.add_edge("answer_agent",END)

    # 记忆
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    graph = graph_builder.compile(
        checkpointer=memory
    )

    def draw_graph():
        from IPython.display import Image, display
        img_data = graph.get_graph().draw_mermaid_png()
        with open("output.png", "wb") as f:
            f.write(img_data)
        display(Image(img_data))
    draw_graph()
    
    return graph

def get_config():
    if student_id == 'A':
        return configA
    elif student_id == 'B':
        return configB
    return config

if __name__ == "__main__":
    graph = GraphBuilder() 
    # for s in graph.stream(
    #     {
    #         "messages": [
    #             HumanMessage(content="你好")
    #         ],
    #         "node_name": "__start__",
    #     },
    #     config,
    #     stream_mode="values",
    # ):
    #     if "__end__" not in s:
    #         print(s)
    #         print("----")
        
    while True:
        print ("请输入：")
        user_message = input()
        
        events = graph.stream(
            {
                "messages": [
                    HumanMessage(content=user_message)
                ],
                "node_name": "__start__",
            },
            get_config(),
            stream_mode="values"
        )
        for event in events:
            # print(event)
            if "messages" in event:
                # event["messages"][-1].pretty_print()
                print('---' + event["node_name"] + '---')
                print(event["messages"][-1].content)
        
        history_msg[student_id] = history_msg[student_id] + "学生：" + user_message + "\n"
        history_msg[student_id] = history_msg[student_id] + "教师：" + event["messages"][-1].content + "\n"
        
        
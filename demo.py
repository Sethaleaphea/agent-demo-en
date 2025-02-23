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
        index_path='/home/dalhxwlyjsuo/guest/result/indexs/dense_cache/fassis.index')  # 知识库路径
    app_config.retriever_config = retriever_config
    application = RagApplication(app_config)
    application.init_vector_store()
    return application


# rag_tool = init_rag()

# def RAG(query: str):
#     return rag_tool.get_rag_content(query)

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
chatLLM = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="deepseek-r1",
)

# from rag_demo import RagApplication, ApplicationConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig, DenseRetriever


def RAG(query: str):
    step_back_prompt="""你是一位擅长从具体问题中提炼出更通用问题的专家，该通用问题能揭示回答具体问题所需的基本原理。
    你将会收到关于专升本教育中各个学科的问题，针对用户提出的具体问题，请你提炼出一个更抽象、更通用的问题，该问题是回答原问题所必须解决的核心问题。
    注意：如果遇到不认识的单词或缩略语，请不要尝试改写它们。请尽量编写简洁的问题。"""
    messages=[
        {"role": "system", "content": step_back_prompt},
        {'role': 'user', 'content': query}
    ]
    query_refined = chatLLM.invoke(messages).content
    print('query_refined:', query_refined)
    print('-----------RAG-------------')
    contents = retriever.retrieve(query=query_refined, top_k=5)
    print('---------------------------')
    contents = '\n'.join([content['text'] for content in contents])
    return contents


from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages

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

# 获取学生画像
def stu_img(llm, history_msg: str, student_id: str):
    if student_id == "A":
        stu_img = "这名学生在法学基础这门课的学习十分优秀，他在这方面的知识水平很高，对于他的提问，教师只需要稍加点拨，给出一点启发即可，他对于民法中的案例分析还不太熟练，需要适当练习。"
    elif student_id == "B":
        stu_img = "学生在法学基础这门课的学习比较一般，他对于宪法、刑法相关知识点尚有不明，但是他在计算机基础这门课表现十分优秀，尤其是数据库原理这部分。"
    else:
        # TODO 更新学生画像
        messages = [
            SystemMessage('''你是一个擅长判断学生知识水平的教师，你的任务是根据和学生的历史记录来判断学生的知识水平，并使用100字左右给出你的看法。
                    历史对话记录：''' + history_msg),
        ]
        stu_img = llm.invoke(messages).content

    global stu_img_content
    stu_img_content = stu_img
    return stu_img


# check 学生对话
def check_msg(llm, history_msg: str, student_msg: str):
    if history_msg == "":
        messages = [
            SystemMessage('''你是一个教师，你正在与你的学生交流，你需要判断学生的语句中是否存在问题。
                    你需要判断学生学生最新一句话是否存在知识点理解有误。
                    如果不存在问题，你只需要回复“无问题”。
                    如果存在问题，你需要给出合理的建议。
                    <注意事项>
                    你只需要判断学生的语句中是否存在问题，不需要回答学生的问题。
                    尽量输出“无问题”，除非学生的话明显与事实不符。
                    <学生最新一句话>
                    ''' + student_msg),
        ]
    else:
        messages = [
            SystemMessage('''你是一个教师，你正在与你的学生交流，你需要根据和学生的历史对话，以及当前学生语言来判断，学生的语句中是否存在问题。
                    你需要判断学生的最新一句话是否与之前的对话存在矛盾，或是学生最新一句话是否存在知识点理解有误。
                    如果不存在问题，你只需要回复“无问题”。
                    如果存在问题，你需要给出合理的建议。
                    <注意事项>
                    你只需要判断学生的语句中是否存在问题，不需要回答学生的问题。
                    尽量输出“无问题”，除非学生的话明显与事实不符，或存在明显前后矛盾。
                    <历史对话记录>
                    ''' + history_msg + '''
                    <学生最新一句话>
                    ''' + student_msg),
        ]

    return llm.invoke(messages).content


# 问题改写agent
def question_rewrite(llm, history_msg: str, student_msg: str):
    if history_msg == "":
        return student_msg
    messages = [
        SystemMessage('''你是一个语言学家，你需要根据一段历史对话记录和学生最新的一段话，来改写学生的最新提问，确保改写得到的话能表达学生的想法，并且独立于对话记录。
                <注意事项>
                你只需要改写学生的提问，不需要回答学生的问题。
                你需要尽量保留学生提问中的信息，并且增加一些必要的对话记录中提到的信息确保改写后的提问能够表达学生的意图。
                你需要确保没有对话记录的情况下，你的改写能够让人明白学生的想法。
                你只需要输出改写后的问题，而不需要输出你的分析过程。
                <历史对话记录>
                ''' + history_msg + '''
                <学生最新一句话>
                ''' + student_msg),
    ]
    return llm.invoke(messages).content

def worng_question_rewrite(llm, history_msg: str, student_msg: str, wrong_msg: str):
    # worng_msg = "修改意见"
    # 推测用户意图，解答 返回
    messages = [
        SystemMessage('''你是一个语言专家，你需要根据学生的最新一句话和历史对话记录，以及同事给你的修改意见，来指出学生提问中的问题，并给出合理的修改意见。
                      向他解释为什么他的错误理解，以提高他的知识水平。
                      <历史对话记录>
                      ''' + history_msg + '''
                      <学生最新一句话>
                      ''' + student_msg + '''
                      <修改意见>
                      ''' + wrong_msg),
    ]
    return llm.invoke(messages).content


# 路由 意图识别
def route_intent(llm, student_msg: str):
    messages = [
        SystemMessage('''你是一个教师，你正在与你的学生交流，你需要根据当前学生最新一句话按下面的流程判断学生当前意图。
                        1. 判断学生是否在和你讨论有关学科的问题或是提问，如果不是，输出“日常对话”。
                        2. 判断学生的疑问是否仅局限于知识概念不明，也就是说题目的难点在于知识概念的记忆和理解而非应用，如果是，输出“概念性问题”。
                        3. 以上两种均不属于输出“习题解答”。
                        学生最新一句话：''' + student_msg),
    ]
    return llm.invoke(messages).content

# 习题解答
def answer(llm, student_msg: str, stu_img_content: str):
    # 分步骤给出解答 以便学生提问和理解
    rag_prompt = '''你是一个教师，你正在与你的学生交流，你需要向学生解答他不懂的习题。请结合参考的上下文内容为学生解答。
                同时，学生画像反映了该学生的知识水平，请根据下面学生画像中反映出的知识水平，为该学生提供个性化、分层次的讲解，要求量身定制：回答时请结合学生的知识水平，采用适合的术语和解释深度。
                <注意事项>
                对于优秀的学生，你只需要给出一点启发即可。
                对于普通学生，你需要给出详细的思考步骤，以便学生能够理解。
                学生的问题：{question}
                学生画像：{stu_img}
                可参考的上下文：
                   ···
                   {context}
                   ···
                   有用的回答:"""
                '''
    contents = RAG(student_msg)
    prompt = rag_prompt.format(question=student_msg, stu_img=stu_img_content,context=contents)
    messages = [
        SystemMessage(prompt),
    ]
    return llm.invoke(messages).content


# 概念性问题回答
def concept_answer(llm, student_msg: str, stu_img_content: str):
    rag_prompt = '''你是一个教师，你正在与你的学生交流，你需要向学生解答他不懂的概念性问题。请结合参考的上下文内容回答用户不懂的概念性问题。
                同时，学生画像反映了该学生的知识水平，请根据下面学生画像中反映出的知识水平，为该学生提供个性化、分层次的讲解，要求量身定制：回答时请结合学生的知识水平，采用适合的术语和解释深度。
                <注意事项>
                对于优秀的学生，你只需要给出一点启发即可，对于普通学生，你需要给出详细的解答。
                你应该首先提取出学生问题中的关键概念，然后给出一个简明扼要的解释。
                再根据学生的知识水平，结合你给出的概念解释，解答问题。
                学生的问题：{question}
                学生画像：{stu_img}
                可参考的上下文：
                   ···
                   {context}
                   ···
                   有用的回答:"""
                '''
    contents = RAG(student_msg)
    # print('RAG-content')
    print(contents)
    prompt = rag_prompt.format(question=student_msg, stu_img=stu_img_content,context=contents)
    messages = [
        SystemMessage(prompt),
    ]
    return llm.invoke(messages).content


# 日常对话 引导学生提问
def daily_conversation(llm, student_msg: str, stu_img_content: str):
    messages = [
        SystemMessage('''你是一个优秀的教师，你正在与你的学生交流，你的任务是对学生的最新一句话进行回应，同时根据学生的学习状况引导学生向你提问，或是给他一些题目练习。
                      学生最新一句话：''' + student_msg + '''\n学生的学习情况：''' + stu_img_content),
        # HumanMessage(student_msg)
    ]
    return llm.invoke(messages).content


# 图的构建
class State(TypedDict):
    messages: Annotated[list, add_messages]
    node_name: str

def GraphBuilder():
    graph_builder = StateGraph(State)

    def stu_img_agent(state: State):
        return {"messages": [AIMessage(content=stu_img(chatLLM, history_msg[student_id], student_id))],
                "node_name": "stu_img_agent"}

    def check_msg_agent(state: State):
        return {"messages": [AIMessage(content=check_msg(chatLLM, history_msg[student_id], user_message))],
                "node_name": "check_msg_agent"}

    def question_rewrite_agent(state: State):
        return {"messages": [AIMessage(content=question_rewrite(chatLLM, history_msg[student_id], user_message))],
                "node_name": "question_rewrite_agent"}

    def wrong_question_rewrite_agent(state: State):
        return {"messages": [AIMessage(content=worng_question_rewrite(chatLLM, history_msg[student_id], user_message,
                                                                      state["messages"][-1].content))],
                "node_name": "wrong_question_rewrite_agent"}

    def route_intent_agent(state: State):
        global rewrite_user_message
        rewrite_user_message = state["messages"][-1].content
        return {"messages": [AIMessage(content=route_intent(chatLLM, rewrite_user_message))],
                "node_name": "route_intent_agent"}

    def answer_agent(state: State):
        return {"messages": [AIMessage(content=answer(chatLLM, rewrite_user_message, stu_img_content))], 
                "node_name": "answer_agent"}

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
        {"question_rewrite_agent": "question_rewrite_agent",
         "wrong_question_rewrite_agent": "wrong_question_rewrite_agent"},
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
        {"concept_answer_agent": "concept_answer_agent", "daily_conversation_agent": "daily_conversation_agent",
         "answer_agent": "answer_agent"},
    )

    graph_builder.add_edge("concept_answer_agent", END)
    graph_builder.add_edge("daily_conversation_agent", END)
    graph_builder.add_edge("answer_agent", END)

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
    index_path=r'D:/code/agent_demo/dense_cache'
    embedding_model_path = r"./bge-large-zh-v1.5"  # 随便写一个就可以
    retriever_config = DenseRetrieverConfig(
        model_name_or_path=embedding_model_path,
        dim=1024,
        index_path=index_path)
    
    from trustrag.modules.retrieval.embedding import FlagModelEmbedding
    embedding_generator = FlagModelEmbedding(embedding_model_path)
    retriever = DenseRetriever(retriever_config, embedding_generator)
    retriever.load_index(index_path)
    
    # contents = RAG('14周岁的中学生李亚楠独自在超市买东西，恰遇超市抽奖，中了5000元大奖，什么样的领奖方式是正确的')
    # print(contents)
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
        print("请输入：")
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
        
    #     # print("历史记录：")
    #     # print(history_msg[student_id])
    #     # print("----")
        
    # print('---------A-----------')
    # an2 = concept_answer(chatLLM, "14周岁的中学生李亚楠独自在超市买东西，恰遇超市抽奖，中了5000元大奖，什么样的领奖方式是正确的", stu_img(chatLLM, history_msg["A"], "A"))
    # print(an2)
    
    # print('---------B-----------')
    # an2 = concept_answer(chatLLM, "14周岁的中学生李亚楠独自在超市买东西，恰遇超市抽奖，中了5000元大奖，什么样的领奖方式是正确的", stu_img(chatLLM, history_msg["B"], "B"))
    # print(an2)
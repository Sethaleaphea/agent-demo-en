import os
from openai import OpenAI

#pip install trustrag 即可
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig, DenseRetriever
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-xxx",  # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
class ApplicationConfig():
    def __init__(self):
        self.retriever_config = None
        self.system_prompts = []

class RagApplication():
    def __init__(self, config):
        self.config = config
        self.retriever = DenseRetriever(self.config.retriever_config)
        self.rag_prompt = """
        请结合参考的上下文内容回答用户问题，如果上下文不能支撑你正确地回答用户问题，并且，你内部的参数知识也无法正确回答用户问题，那么回答不知道或者我无法根据参考信息回答。
               问题: {question}
               可参考的上下文：
               ···
               {context}
               ···
               有用的回答:"""
    def init_vector_store(self):
        print("init_vector_store ... ")
        index_path = self.config.retriever_config.index_path
        if os.path.exists(index_path):
            print("检测到已存在的向量数据库，直接加载...")
            self.load_vector_store()  # 直接加载已有索引
            return
    def load_vector_store(self):
        self.retriever.load_index(self.config.retriever_config.index_path)

    def rag_chat(self, question: str = '', top_k: int = 5,system_prompt_index: int = 0):
        step_back_prompt="""你是一位擅长从具体问题中提炼出更通用问题的专家，该通用问题能揭示回答具体问题所需的基本原理。
你将会收到关于专升本教育中各个学科的问题，针对用户提出的具体问题，请你提炼出一个更抽象、更通用的问题，该问题是回答原问题所必须解决的核心问题。
注意：如果遇到不认识的单词或缩略语，请不要尝试改写它们。请尽量编写简洁的问题。"""
        refined_query=client.chat.completions.create(
            model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
            messages=[
                {"role": "system", "content": step_back_prompt},
                {'role': 'user', 'content': question}
            ]
        )
        refined_query=refined_query.choices[0].message.content
        
        # ! 
        contents = self.retriever.retrieve(query=refined_query, top_k=top_k) #根据query检索topk个最相似的chunk
        contents = '\n'.join([content['text'] for content in contents])
        prompt = self.rag_prompt.format(question=question, context=contents)
        completion = client.chat.completions.create(
            model="deepseek-v3",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
            messages=[
                {"role": "system", "content": self.config.system_prompts[system_prompt_index]},
                {'role': 'user', 'content': question}
            ]
        )
        return completion.choices[0].message.content,contents

    def chat(self, query: str = '',system_prompt_index: int = 0): #不使用rag流程
        completion = client.chat.completions.create(
            model="deepseek-v3",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
            messages=[
                {"role": "system", "content": self.config.system_prompts[system_prompt_index]},
                {'role': 'user', 'content':query}
            ]
        )
        return completion.choices[0].message.content


if __name__ == '__main__':
    app_config = ApplicationConfig()
    embedding_model_path = "./data_process/models/bge-large-zh-v1.5" #随便写一个就可以
    retriever_config = DenseRetrieverConfig(
        model_name_or_path=embedding_model_path,
        dim=1024,
        index_path='/home/dalhxwlyjsuo/guest/result/indexs/dense_cache/fassis.index')
    app_config.retriever_config = retriever_config
    #针对不同用户，有不同的system_prompts
    app_config.system_prompts = ["你是一个专注于专升本教学的答疑助手", "你是一个通用知识问答助手"]
    application = RagApplication(app_config)
    application.init_vector_store()




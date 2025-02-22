import os
from openai import OpenAI
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
        # self.rag_prompt = """
        # 请结合参考的上下文内容回答用户问题，如果上下文不能支撑你正确地回答用户问题，并且，你内部的参数知识也无法正确回答用户问题，那么回答不知道或者我无法根据参考信息回答。
        #        问题: {question}
        #        可参考的上下文：
        #        ···
        #        {context}
        #        ···
        #        有用的回答:"""
    def init_vector_store(self):
        print("init_vector_store ... ")
        index_path = self.config.retriever_config.index_path
        if os.path.exists(index_path):
            print("检测到已存在的向量数据库，直接加载...")
            self.load_vector_store()  # 直接加载已有索引
            return
    def load_vector_store(self):
        self.retriever.load_index(self.config.retriever_config.index_path)
        
    def get_rag_content(self, question: str = '', top_k: int = 5,system_prompt_index: int = 0):
        contents = self.retriever.retrieve(query=question, top_k=top_k)
        contents = '\n'.join([content['text'] for content in contents])
        return contents

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
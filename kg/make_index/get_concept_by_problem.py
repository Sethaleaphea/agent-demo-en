import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import re
import json
from collections import defaultdict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    api_key="sk-EQY6oy2D9Brb2IWxymHTnmeFz5unIyobzHtjDgFs2ZwEPjmY",
    base_url="https://api.chatanywhere.tech/v1",
    model="gpt-4o-mini",
    temperature=0.35,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

class ConceptByProblem:
    def loader_faiss(self, faiss_index_path, embeddings_model_path):
        """
        加载 FAISS 索引
        :param faiss_index_path: FAISS 索引路径
        :param embeddings_model_path: 嵌入模型路径
        :return: FAISS 向量数据库对象
        """
        # 加载 FAISS 索引
        index = faiss.read_index(faiss_index_path + 'faiss_index.index')
        with open(faiss_index_path + "docstore.pkl", "rb") as f:
            docstore = pickle.load(f)

        with open(faiss_index_path + "index_to_docstore_id.pkl", "rb") as f:
            index_to_docstore_id = pickle.load(f)

        # 从加载的索引创建 FAISS 向量数据库
        vector_store_loaded = FAISS(
            embedding_function=HuggingFaceEmbeddings(model_name=embeddings_model_path),
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        return vector_store_loaded

    def __init__(self, q_index_path: str, c_index_path: str, neo4j_path: str, embeddings_model_path: str = "bge-large-zh-v1.5"):
        """
        初始化 ConceptByProblem 类
        :param q_index_path: 问题索引路径
        :param c_index_path: 概念索引路径
        :param neo4j_path: Neo4j 数据库路径
        :param embeddings_model_path: 嵌入模型路径
        """
        self.q_vector_store = self.loader_faiss(q_index_path, embeddings_model_path)
        self.c_vector_store = self.loader_faiss(c_index_path, embeddings_model_path)

        with open(neo4j_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.node_dict = {node["id"]: node for node in data["nodes"]}
        
        self.reverse_edges = defaultdict(list)
        for edge in data["edges"]:
            self.reverse_edges[edge["target"]].append(edge["source"])
        
        self.parent_to_children = defaultdict(list)
        self.child_to_parent = {}

        for edge in data["edges"]:
            if edge["type"] == "包含":
                parent, child = edge["source"], edge["target"]
                self.parent_to_children[parent].append(child)
                self.child_to_parent[child] = parent

    def match_best_son(self, father, sons, question):
        """
        匹配最好的儿子
        :father: 父亲节点
        :param sons: 儿子列表
        :param question: 问题
        :return: 最好的儿子
        """
        
        prompt = '''
-任务-
你是一个自然语言处理专家。你将会被给到一个问题、一个父节点和多个子节点的信息，你的任务是根据父节点和子节点的信息，分析出最符合问题的节点，并给出分析过程。

-输出-
输出内容为json 格式，包含以下字段：
- best_son: 一个数字，为最符合问题的节点id（从子节点和父节点中选择）
- analysis: 你是如何选择这个节点的？请详细描述你的分析过程。

    '''

        prompt = prompt.replace('{', '{{').replace('}', '}}')
        system_message_prompt = SystemMessagePromptTemplate.from_template(prompt)

        human_template="""
        -真实数据- 
        ######################
        问题：{question}
        父节点：{father}
        子节点：{input_text}
        ###################### 
        输出：
        """

        from langchain.chains import LLMChain
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_template])

        chain = LLMChain(llm=llm, prompt=chat_prompt)

        text = ""
        father_text = father['name'] + " " + father['properties']['description'] + " " + father['properties']['page_content']
        for node in sons:
            text += str(node['id']) + "、" + node['name'] + " " + node['properties']['description'] + " " + node['properties']['page_content'] + "\n"
        
        answer = chain.invoke({
            "question": question,
            "father": father_text,
            "input_text": text
            })
        # print(answer['text'])

        text = re.sub(r"^```json\s*|\s*```$", "", answer['text'].strip(), flags=re.IGNORECASE)
        json_data = json.loads(text)
        
        return json_data

    def get_concept_by_question(self, question, k=5):
        """
        根据问题获取概念
        :param question: 问题
        :param vector_store_loaded: 向量数据库对象
        :param k: 返回的结果数量
        :return: 概念列表
        """
        def find_lca(paths):
        # Transpose the list of paths, zip stops at shortest path
            zipped = list(zip(*paths))
            lca = []
            for level in zipped:
                if all(x == level[0] for x in level):
                    lca.append(level[0])
                else:
                    break
            return lca

        # sim_q = q_vector_store_loaded.similarity_search(question, k=k)
        sim_c = self.c_vector_store.similarity_search(question, k=k)

        result_list = []
        path_list = []
        for q in sim_c:
            print(q)
            path = q.metadata["path"]
            path_list.append(path)
        
        lac = find_lca(path_list)

        last_node_id = lac[-1]
        while True:
            sons = self.parent_to_children.get(last_node_id, [])
            # print("sons:", sons)
            if sons:
                son_list = []
                for son in sons:
                    son_list.append(self.node_dict[son])
                
                # print("son_list:", son_list)
                father = self.node_dict[last_node_id]
                last_node = self.match_best_son(father, son_list, question)
            else:
                break
            
            print (last_node)
            if last_node['best_son'] == last_node_id:
                break
            
            last_node_id = int(last_node['best_son'])
        
        last_node = self.node_dict[last_node_id]
        last_sons = self.parent_to_children.get(last_node_id, [])

        # 递归函数：向上查找路径
        def trace_to_root(node_id, visited=None):
            if visited is None:
                visited = set()
            if node_id in visited:
                return []  # 避免环
            visited.add(node_id)
            
            if node_id not in self.reverse_edges:
                return [node_id]  # 到达顶点
            # 默认选择第一条路径（如果多个前驱）
            parent = self.reverse_edges[node_id][0]
            return trace_to_root(parent, visited) + [node_id]
        
        def get_path(node_id, node_dict):
            # print(node_dict[node_id])

            path = trace_to_root(node_id)
            
            path = path[1:]  # 反转路径
            node1 = node_dict[path[0]]
            node2 = node_dict[path[1]]
            
            path_str = '第' + str(node1['properties']['num']) + '章' + node1['properties']['title'] + '\n'
            path_str += '第' + str(node2['properties']['num']) + '节' + node2['properties']['title'] + '\n'
            for p in path[2:]:
                node = node_dict[p]
                path_str += '----' + node['properties']['name'] + '----\n'
            return (last_node['properties']['page_content'], path_str)

        res = []
        res.append(get_path(last_node_id, self.node_dict))

        if last_sons:
            for son in last_sons:
                res.append(get_path(son, self.node_dict))

        return res
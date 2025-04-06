import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import json

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
        # self.c_vector_store = self.loader_faiss(c_index_path, embeddings_model_path)
        
        with open(neo4j_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.node_dict = {node["id"]: node for node in data["nodes"]}


    def get_concept_by_problem(self, problem: str, k: int = 3) -> list:
        """
        根据问题获取相关概念
        :param problem: 问题文本
        :param k: 相似问题数量
        :return: 概念列表
        """

        sim_q = self.q_vector_store.similarity_search(problem, k=k)

        source_list = []
        for q in sim_q:
            # 获取问题的ID
            import ast
            meatdata = q.metadata
            path = ast.literal_eval(meatdata["path"])
            for p in path:
                source_list.append(p)
        
        result_list = []
        
        from collections import Counter
        source_tuples = [tuple(x) for x in source_list]
        counter = Counter(source_tuples)
        sorted_unique_sources = [list(item) for item, count in counter.most_common()]
        
        for path in sorted_unique_sources:
            # 获取问题的ID
            source_id = path[-1]
            # print(path)
            c_node = self.node_dict[source_id]
            
            node1 = self.node_dict[path[0]]
            node2 = self.node_dict[path[1]]
            path_str = '第' + str(node1['properties']['num']) + '章 ' + node1['properties']['title'] + '\n'
            path_str += '第' + str(node2['properties']['num']) + '节 ' + node2['properties']['title'] + '\n'
            for p in path[2:]:
                node = self.node_dict[p]
                path_str += '----' + node['properties']['name'] + '----\n'
            
            result_list.append((c_node['properties']['name'] + c_node['properties']['page_content'], path_str))

        return result_list
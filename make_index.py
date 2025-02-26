import os
import json
import pickle
import pandas as pd
from tqdm import tqdm

from trustrag.modules.document.chunk import TextChunker
from trustrag.modules.document.txt_parser import TextParser
from trustrag.modules.document.utils import PROJECT_BASE
from trustrag.modules.generator.llm import GLM4Chat
from trustrag.modules.reranker.bge_reranker import BgeRerankerConfig, BgeReranker
from trustrag.modules.retrieval.bm25s_retriever import BM25RetrieverConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig,DenseRetriever
from trustrag.modules.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig

embedding_model_path = r"D:/Python/models/bge-large-zh-v1.5"
# def load_json_from_folder(folder_path):
#     """
#     Load JSON objects from all files in a folder into a single list.
    
#     Args:
#         folder_path (str): Path to the folder containing JSON files.
    
#     Returns:
#         list: A list of JSON objects.
#     """
#     documents = []
    
#    # 遍历文件夹中的所有文件
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.jsonl'):  # 检查是否为 JSONL 文件
#             file_path = os.path.join(folder_path, file_name)
            
#             # 打开 JSON 文件并读取每一行
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     data = json.loads(line.strip())  # 加载 JSON 数据
#                     contents = data.get('contents', None)  # 提取 `contents` 键的内容
#                     if isinstance(contents, str):  # 确保内容是字符串
#                         documents.append(contents)  # 添加到 documents 列表中
    
    # return documents

def load_json_from_json(folder_path):
    with open(folder_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    
    for item in data:
        documents.append(item['stem'] + item['analysis'])
        
    return documents

# 使用示例
folder_path = 'q2_output.json' 
documents = load_json_from_json(folder_path)
print(documents)
dense_config = DenseRetrieverConfig(
    model_name_or_path=embedding_model_path,
    dim=1024,
    index_path='qa_dense_cache'
)

from trustrag.modules.retrieval.embedding import FlagModelEmbedding
embedding_generator = FlagModelEmbedding(embedding_model_path)
dense_retriever = DenseRetriever(config=dense_config, embedding_generator=embedding_generator)


# Build the index
dense_retriever.build_from_texts(documents)
# Save the index
dense_retriever.save_index()
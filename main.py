# -*- coding: utf-8 -*-
import requests
import json
import PyPDF2
import re
from urllib.parse import urlparse, urlunparse
import numpy as np
import faiss
from tqdm import tqdm
import os
import pickle


class IPaper:
    def __init__(self, host, model, meta_dir):
        self.model = model
        self.embeddings_url = "http://{}/api/embeddings".format(host)
        self.generate_url = "http://{}/api/generate".format(host)
        self.index_path = os.path.join(meta_dir, "faiss_index.index")
        self.chunks_path = os.path.join(meta_dir, "chunks.pkl")

    # 1. 提取PDF文本
    def __extract_text_from_pdf(pdf_path):
        """
        从PDF文件中提取文本内容。
        """
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ""  # 如果某一页没有文本，避免None导致错误
        return text

    # 2. 将文本分割成块
    def __split_text_into_chunks(self, text, chunk_size=300):
        chunks = []
        sentences = re.split(r'(?<=[。！？])', text)  # 按句子结束符分割
        current_chunk = ""
        for sentence in sentences:
            if not sentence.strip():  # 跳过空句子
                continue
            if len(current_chunk) + len(sentence) > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    # 3. 获取嵌入向量
    def __get_embedding(self, text):
        """调用Ollama API获取文本嵌入向量。"""
        data = {"model": self.model, "prompt": text}
        try:
            response = requests.post(self.embeddings_url, json=data)
            response.raise_for_status()  # 检查HTTP响应是否成功
            embedding = response.json().get("embedding")
            if embedding is None:
                raise ValueError(f"API返回的嵌入向量为空: {response.json()}")
            return np.array(embedding, dtype=np.float32)
        except requests.exceptions.RequestException as e:
            print(f"连接到Ollama服务失败: {e}")
            raise

    def __build_chunk_faiss_index(self, chunks, index_path, chunks_path):
        """
        构建FAISS索引，并将索引和文本块保存到磁盘。
        """
        embeddings = []
        for chunk in tqdm(chunks, desc="生成嵌入向量", unit="块"):
            embedding = self.__get_embedding(chunk)
            if embedding.size == 0:
                print(f"警告：文本块 '{chunk}' 的嵌入向量为空，跳过此块。")
                continue
            embeddings.append(embedding)

            if not embeddings:
                raise ValueError("没有有效的嵌入向量，无法构建索引。")

        embeddings = np.vstack(embeddings)
        index = faiss.IndexFlatL2(embeddings.shape[1])  # 创建一个L2距离的索引
        index.add(embeddings)

        # 保存索引和文本块
        faiss.write_index(index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

    # 5. 加载FAISS索引
    def __load_faiss_index(self, index_path, chunks_path):
        """
        从磁盘加载FAISS索引和文本块。
        """
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            print("索引或文本块文件不存在，请先运行构建索引的程序。")
            return None, None

        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    # 6. 搜索最相似的文本块
    def __search_faiss_index(self, index, chunks, query, top_k=5):
        """
        在FAISS索引中搜索最相似的文本块。
        """
        query_embedding = self.__get_embedding(query).reshape(1, -1)
        D, I = index.search(query_embedding, top_k)
        return [(I[0][i], D[0][i], chunks[I[0][i]]) for i in range(top_k)]

    # 7. 生成回答
    def __generate_response(self, query, context_chunks):
        """
        使用DeepSeek模型基于上下文生成回答。
        """
        context = "\n".join(context_chunks)
        prompt = f"根据以下内容回答问题：\n{context}\n\n问题：{query}\n回答："
        data = {"model": self.model, "prompt": prompt}
        try:
            response = requests.post(self.generate_url, json=data, stream=True)
            response.raise_for_status()

            full_response = ""
            print("正在生成回答...")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    try:
                        json_data = json.loads(decoded_line)
                        if "response" in json_data:
                            chunk = json_data["response"]
                            full_response += chunk
                            print(chunk, end="", flush=True)
                    except json.JSONDecodeError as e:
                        print(f"\n解析错误: {e}")
                        print(f"无法解析的内容: {decoded_line}")

            return full_response, []

        except requests.exceptions.RequestException as e:
            print(f"生成回答失败: {e}")
            return "无法生成回答，请检查模型或上下文。", []

    def build_faiss_index(self, directory):
        pdf_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                pdf_files.append(file_path)

        pdf_files = sorted(pdf_files)
        for pdf_path in pdf_files:
            pdf_text = self.__extract_text_from_pdf(pdf_path)
            chunks = self.__split_text_into_chunks(pdf_text)
            self.__build_chunk_faiss_index(chunks, self.index_path,
                                           self.chunks_path)

    def start_chat(self):
        # 加载索引和文本块
        index, chunks = self.__load_faiss_index(self.index_path,
                                                self.chunks_path)
        if index is None or chunks is None:
            print("加载索引失败，请检查索引文件是否存在。")
            exit()

        # 多次对话循环
        while True:
            # 查询知识库
            query = input("\n请输入你的问题（输入 'exit' 结束对话）：").strip()
            if query.lower() == 'exit':
                print("对话结束。")
                break

            results = self.__search_faiss_index(index, chunks, query, top_k=5)

            # 提取相关块内容
            _, _, relevant_chunks = zip(*results)
            relevant_chunks = list(relevant_chunks)  # 将相关块内容转换为列表

            # 使用DeepSeek模型生成回答并展示思考过程
            response, _ = self.__generate_response(query, relevant_chunks)
            print("\n查询结果：")
            print(response)


if __name__ == "__main__":
    pdf_path = "./pdfs"

    # OLLMA_HOST = "localhost:11434"
    # MODEL = "qwen3:32b"
    # MODEL = "deepseek-r1:7b"
    instance = IPaper(host="localhost:11434", model="deepseek-r1:7b",
                      meta_dir="indexdb")

    #instance.build_faiss_index(pdf_path, index_path, chunks_path)
    instance.start_chat(index_path, chunks_path)

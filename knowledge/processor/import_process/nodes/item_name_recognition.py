import os
from typing import Tuple, List, Dict, Any
from pymilvus import MilvusClient, DataType
import json
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    """
        商品名称识别
        利用3个chunk的内容上下文，小于2500字符。
        通过提示词模板让LLM识别出商品名称和型号。
        将商品名称信息Bge-m3向量化存Milvus，供后续匹配使用
    """
    name = "item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1.参数校验，获取file_title, chunks, item_name_chunk_k, item_name_chunk_size
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = self._validate_state(state)

        #2.构建识别上下文（前3块切片，限制字符数）
        item_name_recognition_context = self._prepare_context(chunks, item_name_chunk_k, item_name_chunk_size)

        #3.LLM识别商品名    降级file_title
        item_name: str = self._recognition_name(file_title, item_name_recognition_context)

        #4.BGE-M3向量化
        dense_vector: List[float]
        sparse_vector: dict[int, float]
        dense_vector, sparse_vector= self._embedding_item_name(item_name)

        #5.存储到Milvus    StorageClients获取客户端，创建集合，数据保存;  存储失败跳过，并记录日志。
        #标量字段：pk, file_title ,item_name
        #向量字段：dense_vector,sparse_vector
        self.insert_milvus(file_title, item_name, dense_vector, sparse_vector)

        #6.回填item_name到state和chunks,同时回填state对象中
        self._fill_item_name(item_name, state, chunks)

        # 7.备份，给下个节点准备下测试数据。
        self._backup_chunks(state, chunks)

        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[str, List[dict[str, Any]], int, int]:
        """
        1.参数校验，获取file_title, chunks, item_name_chunk_k, item_name_chunk_size
        :param state: ImportGraphState
：      :return:  file_title, chunks, item_name_chunk_k, item_name_chunk_size
        """
        file_title = state["file_title"]
        chunks = state["chunks"]
        item_name_chunk_k = self.config.item_name_chunk_k
        item_name_chunk_size = self.config.item_name_chunk_size

        if not file_title:
            self.logger.error(f"file_title不存在")
            raise StateFieldError(self.name, "file_title", expected_type=str)

        if not chunks:
            self.logger.error(f"chunks不存在")
            raise StateFieldError(self.name, "chunks", expected_type=list)

        if not item_name_chunk_k or item_name_chunk_k <= 0:
            self.logger.error(f"item_name_chunk_k不存在或小于等于0")
            raise ValidationError(f"item_name_chunk_k不存在或小于等于0", self.name)

        if not item_name_chunk_size or item_name_chunk_size <= 0:
            self.logger.error(f"item_name_chunk_size不存在或小于等于0")
            raise ValidationError(f"item_name_chunk_size不存在或小于等于0", self.name)

        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    def _prepare_context(self, chunks: List[dict[str, Any]], item_name_chunk_k: int, item_name_chunk_size: int) -> str:
        """
        2.构建识别上下文（前3块切片，限制字符数）
        :param chunks:  List[dict[str, Any]]切片列表
        :param item_name_chunk_k:提取上下文切片限制
        :param item_name_chunk_size:LLM提示词模板参数长度限制
        :return:用于提取商品名称的上下文参考内容
        """
        #取前3个chunk
        sub_chunks = chunks[0:item_name_chunk_k]
        final_context = ""
        total = 0
        for index, chunk in enumerate(sub_chunks):

            if not chunk or not isinstance(chunk, dict):
                continue

            chunk_content = chunk.get("content", "")

            content = f"【切片-{index + 1}】\n" + chunk_content
            total += len(content)
            if total < item_name_chunk_size:
                final_context += content

        return "\n\n".join(final_context)

    def _recognition_name(self, file_title: str, item_name_recognition_context: str) -> str:
        """
        3.LLM识别商品名    降级file_title
        :param file_title: 文件名称
        :param item_name_recognition_context: 识别商品名称的上下文内容
        :return: 商品名称
        """

        #获取LLM模型客户端对象
        llm_client = AIClients.get_llm_openai(False)

        #用户提示词模板
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title,
                                                            context=item_name_recognition_context)

        #调用LLM模型
        llm_response = llm_client.invoke([
            SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        #处理LLM模型返回结果失败，降级处理
        if not llm_response or llm_response.content == "UNKNOWN":
            self.logger.error(f"LLM模型返回结果失败，降级处理，返回文件名称作为商品名称")
            return file_title


        # 获取响应内容
        #item_name = llm_response.content.strip()  # {"item_name": "RS-12 数字万用表"}
        response_dict = json.loads(llm_response.content.strip())
        item_name = response_dict.get("item_name")  # "RS-12 数字万用表"

        return item_name


    def _embedding_item_name(self, item_name: str) -> Tuple[List[float], Dict[int, float]]:
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"获取BGE-M3客户端初始化失败", e)
            return None, None  #降级处理

        try:
            embeddings = bge_m3_client.encode_documents([item_name])

            dense_vector = embeddings["dense"][0].tolist()  #List[float],长度1024
            sparse_matrix = embeddings["sparse"]        #CSR稀疏矩阵
            start_idx = sparse_matrix.indptr[0]   #indptr = [0,5]
            end_idx = sparse_matrix.indptr[1]
            token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()# indices = [11,22,33,44,55]
            weights = sparse_matrix.data[start_idx:end_idx].tolist() # indices = [111,222,333,444,555]
            sparse_vector = dict(zip(token_ids, weights))  # Dict[int, float]

            # 返回 稠密向量，稀疏向量
            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"BGE-M3向量化商品名称失败", e)
            return None, None

    def insert_milvus(self, file_title, item_name, dense_vector, sparse_vector):
        # 5.存储到Milvus   StorageClients获取客户端，创建集合，数据保存;  存储失败跳过，并记录日志。
        # 标量字段：pk, file_title ,item_name
        # 向量字段：dense_vector,sparse_vector

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"获取Milvus客户端初始化失败", e)
            return

        try:
            item_name_collection = self.config.item_name_collection
            if not milvus_client.has_collection(item_name_collection):
                self.logger.info(f"{item_name_collection}不存在，正在创建...")
                self._create_item_name_collection(item_name_collection, milvus_client)
                self.logger.info(f"{item_name_collection}集合创建成功")
        except Exception as e:
            self.logger.error(f"{item_name_collection} 集合创建失败", e)

        try:
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            self.logger.info(f"正在保存数据到Milvus...={data}")
            milvus_client.insert(item_name_collection, [data])
            self.logger.info(f"数据保存成功")
        except Exception as e:
            self.logger.error(f"数据保存失败", e)

    def _create_item_name_collection(self, collection_name, milvus_client):
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True,
                         max_length=100)  # 主键
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)  # 标量字段
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)  # 标量字段
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)  # 稠密向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector", index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")  # 模糊近似查询   余弦相似度  只考虑方向，夹角，不考虑长度
        index_param.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX",
                              metric_type="IP")  # IP内积  方向和长度都考虑 (查询更精准)    如果归一化后，与COSINE相似

        milvus_client.create_collection(collection_name=collection_name,
                                        schema=schema, index_params=index_param)
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")

    def _fill_item_name(self, item_name: str, state: ImportGraphState, chunks: List[dict[str, Any]]) -> None:
        for chunk in chunks:
            chunk["item_name"] = item_name

        state["item_name"] = item_name
        state["chunks"] = chunks
        """
        :return:
                             chunks = [{
                                        "title": "## 安全手册",
                                        "parent_title": "# 手册",
                                        "file_title": "万用表手册",                                      
                                        "content": "title" + "body",
                                        "item_name": "item_name"
                                    },
                                    {
                                        "title": "## 安全手册",
                                        "parent_title": "# 手册",
                                        "file_title": "万用表手册",
                                        "content": "title" + "body",
                                        "item_name": "item_name"
                                    }
                                ]
        """

    def _backup_chunks(self, state: ImportGraphState, chunks: List[dict[str, Any]]):
        """
        将切分结果备份到json文件
        :param state: ImportGraphState
        :param chunks: List[dict[str, Any]]
        :return:
        """

        local_dir = state.get("file_dir", "")
        if not local_dir:
            # 建议加上这行日志，确认是否因为目录为空而跳过
            self.logger.warning("file_dir 为空，跳过备份。请检查 state 内容。")
            return

        os.makedirs(local_dir, exist_ok=True) #exist_ok=True: 如果目录已存在，不会抛出异常，直接跳过
        output_path = os.path.join(local_dir, "chunks_item_name.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
            self.logger.info(f"切分结果已备份到 {output_path}")
        except Exception as e:
            self.logger.warning(f"切分结果备份失败", e)

if __name__ == '__main__':
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"D:\Code\shopkeeper_brain_file\tmp_dir\万用表RS-12的使用\auto\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "万用表RS-12的使用_new",
        "chunks": chunk_content,
        "file_dir": r"D:\Code\shopkeeper_brain_file\tmp_dir\万用表RS-12的使用\auto"
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    processed_state = node(state)

    # 5. 输出结果
    print(f"商品名: {processed_state.get('item_name')}")   #商品名: '{"item_name": "RS-12 数字万用表"}'
    print(type(processed_state.get('item_name')))
    print(f"chunks数量: {len(processed_state.get('chunks', []))}")    #25
    print(f"首5个chunk是否含item_name: {'item_name' in processed_state['chunks'][0]}")   #首个chunk是否含item_name: True

    """  这个方法设计
        llm_client = AIClients.get_llm_openai(False)
        商品名: {"item_name": "RS-12 数字万用表"}
        <class 'str'>
        chunks数量: 25
        首个chunk是否含item_name: True

        llm_client = AIClients.get_llm_openai(True)
        商品名: {"item_name": "RS-12 数字万用表"}
        <class 'str'>
        chunks数量: 25
        首个chunk是否含item_name: True

        response_dict = json.loads(llm_response.content.strip())
        item_name = response_dict.get("item_name")
    """
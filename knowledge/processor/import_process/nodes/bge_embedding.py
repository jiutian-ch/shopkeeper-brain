import json
from pathlib import Path
from typing import Tuple, List, Any

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.utils.client.ai_clients import AIClients


class BgeEmbeddingChunksNode(BaseNode):
    """
        BgeEmbeddingChunksNode 主要职责：

        1. 获取所有的 chunks 拼接要向量的内容
        2. 批量嵌入 chunk 的（embedding_content: item_name + chunk.get('content')）
        3. 将所有 chunk 嵌入后的向量值，存储到列表中，在返回给下一个节点用
    """
    name = "bge_embedding_node"
    def process(self,state:ImportGraphState)->ImportGraphState:
        #1.参数校验
        validated_chunks, config = self._validate_get_inputs(state)
        #2.获取批量嵌入的阈值
        embedding_batch_chunk_size = getattr(config, "embedding_batch_size", 16)

        #3.准备分批嵌入
        total_length = len(validated_chunks)
        final_chunks = []

        for i in range(0, total_length, embedding_batch_chunk_size):
            batch: List[dict[str, Any]] = validated_chunks[i: i + embedding_batch_chunk_size]
            #凭借要嵌入的内容，向量嵌入，把嵌入的向量注入到chunk中
            batch_chunks = self._process_batch_chunks(batch, i, total_length)
            final_chunks.extend(batch_chunks)

        #4.更新&返回state
        state["chunks"] = final_chunks

        return state

    def _validate_get_inputs(self, state: ImportGraphState)-> Tuple[List[dict[str, Any]], Any]:
        """验证输入参数"""
        self.log_step("step1","参数校验")

        config = get_config()

        chunks = state.get("chunks")

        if not chunks or not isinstance(chunks, list):
            raise ValidationError(f"chunks为空或者无效", self.name)

        self.logger.info(f"嵌入的块数: {len(chunks)}")
        return chunks, config

    def _process_batch_chunks(self, batch: List[dict[str, Any]], star_index: int, total_length: int):
        """处理一个批次的切片"""
        self.log_step("step2",f"开始批量处理chunk嵌入:批次{star_index + 1}-{star_index + len(batch)}")
        #1.循环处理所有chunk的要嵌入的内容拼接
        embedding_contents = []
        for chunk in batch:
            content = chunk.get("content")
            item_name = chunk.get("item_name")
            embedding_content = f"{item_name}\n{content}"
            embedding_contents.append(embedding_content)

        #2.批量嵌入
        try:
            bge_m3_model = AIClients.get_bge_m3_client()
            embedding_result = bge_m3_model.encode_documents(documents = embedding_contents)

            if not embedding_result:
                self.logger.warning(f"向量化结果不存在")
                return batch
        except Exception as e:
            self.logger.warning(f"向量化失败...{str(e)}")
            return batch

        #3.循环处理所有chunk的向量以及主要到每一个chunk中
        for index, chunk in enumerate(batch):
            # 3.1 获取稠密向量
            dense_vector = embedding_result["dense"][index].tolist()

            # 3.2解构csr矩阵 & 获取稀疏向量
            csr_array = embedding_result["sparse"]
            ind_ptr = csr_array.indptr
            start_ind_ptr = ind_ptr[index]
            end_ind_ptr = ind_ptr[index + 1]
            token_id = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()
            weight = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()
            sparse_vector = dict(zip(token_id, weight))

            #3.3注入
            chunk["dense_vector"] = dense_vector
            chunk["sparse_vector"] = sparse_vector

        self.logger.info(f"开始批量处理chunk嵌入：批次{star_index + 1}-{star_index + len(batch)}/{total_length}")
        return batch


# ================================================================== #
#                        测试代码                                     #
# ================================================================== #

if __name__ == "__main__":
    setup_logging()

    base_temp_dir = Path(r"D:\Code\shopkeeper_brain_file\tmp_dir\万用表RS-12的使用\auto")

    input_path = base_temp_dir / "chunks_item_name.json"     #上个节点识别商品名称回填chunks临时存储
    output_path = base_temp_dir / "chunks_item_name_vector.json"     #转换完向量后进行临时存储

    #1.读取上游状态
    if not input_path.exists():
        print(f"输入文件不存在: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    #2.构建虚拟的图状态
    state = {
        "chunks": content
    }

    #3.触发节点执行
    node_bge_embedding = BgeEmbeddingChunksNode()
    proceed_result = node_bge_embedding(state)

    #4.结果落盘
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f"向量生成测完成！结果已成功备份至：\n{output_path}")
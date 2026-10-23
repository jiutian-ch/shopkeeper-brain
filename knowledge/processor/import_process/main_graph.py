import json

from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph


from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.ducment_split import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode
from knowledge.processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.md_img import MarkDownImageNode
from knowledge.processor.import_process.nodes.pdf_to_md import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState


def router(state: ImportGraphState) -> str:
    if state.get("is_pdf_read_enabled"):
        return "pdf"
    if state.get("is_md_read_enabled"):
        return "md"
    return "END"


def create_import_graph() -> CompiledStateGraph:
    """
    创建导入流程图

    Returns:
        编译后的 StateGraph 实例

    流程结构:
        entry_node
              │
              ├── (PDF) ──> pdf_to_md_node ──┐
              │                              │
              └── (MD) ─────────────────────>├──> md_img_node
                                              │
                                              v
                                      document_split_node
                                              │
                                              v
                                      item_name_rec_node
                                              │
                                              v
                                        bge_embedding_node
                                              │
                                              v
                                        import_milvus_node
                                              │
                                              v
                                             END
    """


    #1.创建图
    graph = StateGraph(ImportGraphState)

    #2.设置开启节点
    graph.set_entry_point("entry_node")

    #3.增加节点
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_recognition": ItemNameRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode()
    }

    for name, node in nodes.items():
        graph.add_node(name, node)

    #4.定义边
     #4.1条件边
    graph.add_conditional_edges(
        "entry_node",
        router,
        {
            "pdf":"pdf_to_md_node",
            "md":"md_img_node",
            "END":END
        }

    )

     #4.2增加顺序边
    graph.add_edge("pdf_to_md_node","md_img_node")
    graph.add_edge("md_img_node","document_split_node")
    graph.add_edge("document_split_node","item_name_recognition")
    graph.add_edge("item_name_recognition","bge_embedding_node")
    graph.add_edge("bge_embedding_node","import_milvus_node")
    graph.add_edge("import_milvus_node",END)


    #编译图
    return graph.compile()

kb_import_graph = create_import_graph()

def run_import_graph(import_file_path: str, file_dir: str) ->dict:
    """运行导入流程"""
    state = {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path":import_file_path,
        "file_dir":file_dir
    }

    final_state = None
    for event in kb_import_graph.stream(state):
        for name_node, state in event.items():
            print(name_node)
            final_state = state
    return final_state

# ================================================================== #
#                        测试                                        #
# ================================================================== #

if __name__ == "__main__":
    import_file_path = "D:\\Code\\shopkeeper_brain_file\\doc\\万用表RS-12的使用.pdf"
    file_dir = "D:\\Code\\shopkeeper_brain_file\\tmp_dir"

    #运行导入流程
    final_state = run_import_graph(import_file_path, file_dir)

    print(json.dumps(final_state, indent=4, ensure_ascii=False))
    kb_import_graph.get_graph().print_ascii()


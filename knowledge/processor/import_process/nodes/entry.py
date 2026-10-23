import json

from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    """
    入口节点
        .pdf -> pdf_to_md_path -> md_img_node
        .md -> md_img_node

    """
    name = "entry_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1","获取文件和目录")
        # 1.获取文件路径
        import_file_path = state.get("import_file_path","")
        file_dir = state.get("file_dir","")

        self.log_step("step2","检测文件是否为空")

        # 2.检测文件路径是否为空
        if not import_file_path:
            raise ValidationError(f"import_file_path文件路径为空",self.name)

        # 3.获取文件后缀且判断文件是否为空并改为小写
        path = Path(import_file_path)
        if not path.exists():
            raise ValidationError(f"文件不存在:{import_file_path}", self.name)

        suffix = path.suffix.lower()


        # 4.不同文件类型做不同状态修改
        self.log_step("step3","判断文件类型")
        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
        elif suffix == ".md":
            state["is_md_read_enabled"] = True
        else:
            raise ValidationError(f"不支持的文件类型：{import_file_path} -> 类型： {suffix}",self.name)

        # 5.获取文件标题   不带扩展名
        file_name = path.stem

        # 6.更新状态，并返回状态
        state["file_title"] = file_name

        return state



if __name__ == "__main__":
    setup_logging()
    import_file_path = "D:\\Code\\shopkeeper_brain_file\\doc\\万用表RS-12的使用.pdf"
    file_dir = "D:\\Code\\shopkeeper_brain_file\\tmp_dir"

    state = {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    entry = EntryNode()

    process_state = entry(state)
    print(json.dumps(process_state, indent=4, ensure_ascii=False))
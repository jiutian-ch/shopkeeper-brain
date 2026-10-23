import subprocess
from pathlib import Path
from typing import Tuple

import json

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"
    def process(self,state: ImportGraphState)->ImportGraphState:
        #1.参数校验
        import_file_path_obj, file_dir_obj = self._validate_state_inputs_path(state)
        #2.执行MinerU
        processed_code = self._execute_mineru(import_file_path_obj, file_dir_obj)
        #3.获取MD路径
        md_path = self._get_md_paths(import_file_path_obj,file_dir_obj)

        #4.修改状态并返回
        state["md_path"] = md_path

        return state

    def _validate_state_inputs_path(self,state: ImportGraphState) -> Tuple[Path,Path]:
        import_file_path = state.get("import_file_path","")
        file_dir = state.get("file_dir","")

        if not import_file_path:
            raise ValidationError("输入文件路径为空",self.name)

        import_file_path_obj = Path(import_file_path)

        if not import_file_path_obj.exists():
            raise FileProcessingError("输入文件路径不存在",self.name)

        if not file_dir:
            file_dir = import_file_path_obj.parent

        file_dir_obj = Path(file_dir)

        return import_file_path_obj,file_dir_obj

    def _execute_mineru(self, import_file_path_obj, file_dir_obj) ->int:
        proc = subprocess.Popen(
            args=["mineru", "-p", import_file_path_obj, "-o",file_dir_obj, "--source", "local", "--backend", "pipeline"],
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.STDOUT,  # 合并错误到标准输出
            text=True,
            encoding="utf-8",
            errors="replace",  # 遇到乱码时替换
            bufsize=1  # 行缓冲，实时输出
        )

        for line in proc.stdout:
            print(line.rstrip())

        processed_code = proc.wait()

        return processed_code

    def _get_md_paths(self, import_file_path_obj: Path, file_dir_obj: Path) -> str :

        file_name = import_file_path_obj.stem
        md_path =  import_file_path_obj / file_name / "auto" / f"{file_name}.md"

        return str(md_path)


# ================================================================== #
#                        测试                                        #
# ================================================================== #

if __name__ == "__main__":

    import_file_path = "D:\\Code\\shopkeeper_brain_file\\doc\\万用表RS-12的使用.pdf"
    file_dir = "D:\\Code\\shopkeeper_brain_file\\tmp_dir"

    state = {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    #运行转化流程
    pdfToMdNode = PdfToMdNode()
    processed_state = pdfToMdNode(state)

    print(json.dumps(processed_state, indent=4, ensure_ascii=False))

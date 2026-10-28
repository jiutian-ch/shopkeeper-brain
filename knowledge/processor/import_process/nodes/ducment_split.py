import os
import re
from typing import List, Any

import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rdkit.Chem.FunctionalGroups import hierarchy

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):

    name = "document_split_node"

    def process(self,state:ImportGraphState)->ImportGraphState:
        """
                文档切分处理:
                    1.获取输入参数并且校验：  md_content,file_title,max_content_length,min_content_length

                    2.按照标题切分：按\n获取文档行的集合。遍历，判断是否存在代码围挡，正则表达式识别标题；构建section
                        sections = [{
                                "title": "## 安全手册",
                                "parent_title": "# 手册",
                                "file_title": "万用表手册",
                                "body": "为了您的安全，请在使用本仪表之前仔细阅读该手册。。。。。。。"
                            },
                            {
                                "title": "## 安全手册",
                                "parent_title": "# 手册",
                                "file_title": "万用表手册",
                                "body": "为了您的安全，请在使用本仪表之前仔细阅读该手册。。。。。。。"
                            }
                        ]
                    3.长切短合
                        1.长文档切分：
                            如果文档长度大于max_content_length，则进行二次切分


                        2.短文档合：如果文档长度小于min_content_length，则与前一个文档合并
                            贪婪算法

                    4.组装最终chunk
                        chunks = [{
                                "title": "## 安全手册",
                                "parent_title": "# 手册",
                                "file_title": "万用表手册",
                                "content": "title" + "body"
                            },
                            {
                                "title": "## 安全手册",
                                "parent_title": "# 手册",
                                "file_title": "万用表手册",
                                "content": "title" + "body"
                            }
                        ]

                    5.日志统计

                    6.备份chunks （非必须。为了我们自己查看。也为了方便下一个节点进行单元测试。）


                :param state: ImportGraphState
                :return: ImportGraphState
                """

        # 1.获取输入参数并且校验：md_content,file_title,max_content_length,min_content_length
        md_content, file_title, max_content_length, min_content_length = self._validate_get_input(state)

        # 2.按照标题切分
        sections: List[dict[str, Any]] = self._split_by_title(md_content, file_title)
        # 3.长切短合
        final_sections: List[dict[str, Any]] = self._split_long_merge_short(sections, max_content_length, min_content_length)
        # 4.组装切片
        final_chunks: List[dict[str, Any]] = self._assemble_chunk(final_sections)
        # 5.日志记录
        self._log_summary(md_content, final_chunks,max_content_length)
        # 6.数据备份
        self._backup_chunks(state, final_chunks)

        #状态值设置
        state["chunks"] = final_chunks

        return state

    # ==================================================================================
    #          1.获取并校验参数
    # ==================================================================================
    def _validate_get_input(self, state):
        self.log_step("step1","获取并校验参数")
        # 获取输入参数并且校验：md_content,file_title,max_content_length,min_content_length
        md_content = state.get("md_content")    #整个md文档内容
        file_title = state.get("file_title")    #文件名称
        max_content_length = self.config.max_content_length #最大切片阈值     大于这个阈值需要二次切分
        min_content_length = self.config.min_content_length #最小切片阈值     小于这个阈值需要合并

        if not md_content:
            raise ValidationError(f"切分的文档内容不存在！", self.name)

        if not file_title:
            raise ValidationError(f"切分的文档名称不存在！", self.name)

        if max_content_length < 0:
            raise ValidationError(f"参数错误：max_content_length 不能小于 0",self.name)

        if min_content_length < 0:
            raise ValidationError(f"参数错误：min_content_length 不能小于 0",self.name)

        if max_content_length < min_content_length:
            raise ValidationError(f"参数错误：max_content_length 不能小于 min_content_length",self.name)

        return md_content, file_title, max_content_length, min_content_length

    # ==================================================================================
    #          2.按照标题切分
    # ==================================================================================
    def _split_by_title(self, md_content: str, file_title: str)-> List[dict[str,Any]]:
        """
                   按照标题切分：按\n获取文档行的集合。遍历，判断是否存在代码围挡，正则表达式识别标题；构建section
                       sections = [{
                               "title": "## 安全手册",
                               "parent_title": "# 手册",
                               "file_title": "万用表手册",
                               "body": "为了您的安全，请在使用本仪表之前仔细阅读该手册。。。。。。。"
                           },
                           {
                               "title": "## 安全手册",
                               "parent_title": "# 手册",
                               "file_title": "万用表手册",
                               "body": "为了您的安全，请在使用本仪表之前仔细阅读该手册。。。。。。。"
                           }
                       ]
               :param md_content:
               :param file_title:
               :return:
        """
        self.log_step("step2","按照标题切分")
        # 变量声明区域
        sections: List[dict[str,Any]] = []
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)$")
        body_lines = []
        title = ""
        parent_title = ""
        #存放层级标题
        # ["","","","","","",""]
        # ["","# 标题","## 标题","### 标题","#### 标题","##### 标题","###### 标题"]
        #不明白什么意思
        hierarchy = [""] * 7    # 标题等级： 7个长度，第一个（索引0位置）不用  等级、层级
        current_level = 0
        is_fence = False    # 是否是代码围挡

        # 1.获取所有行集合
        lines: List[str] = md_content.split("\n")

        def _flush():
            """专业封装section"""
            body = "\n".join(body_lines)
            if body:

                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    parent_title = hierarchy[i] if hierarchy[i] else ""
                    if parent_title:
                        break

                # 降级处理：  当前标题没有父标题时，用当前标题作为父标题。比如文件名作为父标题好一些（信息更精准写）。
                if not parent_title:
                    parent_title = title if title else file_title

                #封装section
                section = {
                    "title": title if title else file_title,
                    "parent_title": parent_title,
                    "file_title": file_title,
                    "body": body
                }
                sections.append(section)

        # 2.遍历
        for line in lines:

            # 判断#是否在代码围挡中
            if line.strip().startswith("~~~") or line.strip().startswith("```"):  #strip() 去除行首行尾空格，startswith()判断是否以某个字符串开始
                is_fence = not is_fence

            match = heading_re.match(line)

            if match and not is_fence:
                #封装scetion
                _flush()
                title = line
                level = len(match.group(1)) # 获取#数量
                current_level = level
                # hierarchy = [""] * 7  不能这么处理，否则存在多个同级标题时，后面标题可能无法获取父标题。
                hierarchy[level] = line
                for i in range(level + 1, 7, 1):
                    hierarchy[i] = "" #清楚上个章节遗留标题
                body_lines = []     # 清除缓存，删除上个章节body内容。接下来存我们这个章节的内容。
            else:   #不是标题，装箱
                body_lines.append(line)

        _flush()
        return sections     # 按照标题进行初切结果列表

    # ==================================================================================
    #          3.长切短合
    # ==================================================================================
    def _split_long_merge_short(self, sections: List[dict[str, Any]], max_content_length: int = 1000, min_content_length: int = 200)-> List[dict[str,Any]]:
        self.log_step("step3","长切短合")
        #1.长的超过阈值max_content_length，需要二次切分
        current_sections = []
        for section in sections:
            #section不大于阈值，放在列表中返回
            #section大于阈值，切分完成后放列表返回。
            sub_sections: List[dict[str,Any]] = self._split_long_section(section, max_content_length)
            current_sections.extend(sub_sections)   # 把你的列表元素都放在我的列表里。

        #2.短的小于阈值min_content_length，进行合并
        final_sections = self._merge_short_section(current_sections, min_content_length)
        return final_sections

    # ==================================================================================
    #          3.1 长切
    # ==================================================================================
    def _split_long_section(self, section: dict[str, Any], max_content_length: int = 1000)-> List[dict[str, Any]]:
        """
                长章节再次切分：
                    如果文档长度大于max_content_length，则进行二次切分
                    RecursiveCharacterTextSplitter：递归字符文本切分器
                :param section:  章节信息
                :param max_content_length: 最大切片阈值
                :return: 切与不切的列表
        """
        self.log_step("step3.1","长切")
        #1.获取section对象属性
        title = section.get("title","")
        parent_title = section.get("parent_title", "")
        file_title = section.get("file_title", "")
        body = section.get("body","")

        #2.判断表格
        #利用工具类，将表格降维处理
        if "<table>" in body:
            self.logger.info(f"将表格进行降维处理")
            body = MarkdownTableLinearizer.process(body)
            section['body'] = body

        #3.对标题校验长度，超过50截断
        if len(title) > 50:
            title = title[:50]

        #4.拼接标题前缀
        title_prefix = f"{title}\n\n"

        #5.计算标题前缀长度+内容长度
        content_length = len(title_prefix) + len(body)

        #6.判断是否需要切分
        if content_length <= max_content_length:
            return [section]

        #7.计算body可用长度，判断长度是否小于等于0
        body_available_length = max_content_length - len(title_prefix)
        if body_available_length <=0:
            return [section]

        #8.需要继续切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_content_length,
            chunk_overlap=0,
            keep_separator=False,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
        )
        #判断切分数量结果，要么返回唯一原元素列表，要么返回多个新元素列表
        texts: List[str] = splitter.split_text(body)

        sub_section = []

        for index, text in enumerate(texts):
            sub_section.append(
                {
                    "title": f"{title}",
                    "parent_title": parent_title,
                    "file_title": file_title,
                    "body": text,
                    "part": index + 1
                }
            )

        #9.返回
        return sub_section

    # ==================================================================================
    #          3.2 短合
    # ==================================================================================
    def _merge_short_section(self, current_sections: List[dict[str, Any]], min_content_length: int = 200)-> List[dict[str, Any]]:
        """
        贪心累加算法
            1.累加过程可能超过阈值，可接受
            2.孤儿小块，别忘记处理

        :param current_sections: 当前章节列表，有小的块就需要被合并
        :param min_content_length: 最小内容长度阈值，小于它就需要合并下一个内容到当前内容中。
        :return:
        """
        self.log_step("step3.2","短切")
        final_sections = []     #合并后的章节列表
        current_section = current_sections[0]   #当前章节section对象

        for next_section in current_sections[1:]:

            # 同一个大章节下，两个小章节，可以合并
            same_title = current_section.get("parent_title") == next_section.get("parent_title")
            if same_title and len(current_section.get("body")) < min_content_length:
                current_section["body"] = current_section["body"].rstrip() + "\n\n" + next_section["body"].lstrip()
                current_section["title"] = next_section["parent_title"]
            else:
                final_sections.append(current_section)  #遇到不同父标题，把之前同一个标题合并内容封箱。
                current_section = next_section  #重置

        #孤儿小块
        final_sections.append(current_section)

        return final_sections

    # ==================================================================================
    #          4.组装切片
    # ==================================================================================
    def _assemble_chunk(self, final_sections: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """
                    组装切片
                :param final_sections:
                    sections = [{
                                "title": "## 安全手册",
                                "parent_title": "# 手册",
                                "file_title": "万用表手册",
                                "body": "为了您的安全，请在使用本仪表之前仔细阅读该手册。。。。。。。"
                            },
                            {
                                "title": "## 安全手册",
                                "parent_title": "# 手册",
                                "file_title": "万用表手册",
                                "body": "为了您的安全，请在使用本仪表之前仔细阅读该手册。。。。。。。"
                            }
                        ]
                :return:
                     chunks = [{
                                "title": "## 安全手册",
                                "parent_title": "# 手册",
                                "file_title": "万用表手册",
                                "content": "title" + "body"
                            },
                            {
                                "title": "## 安全手册",
                                "parent_title": "# 手册",
                                "file_title": "万用表手册",
                                "content": "title" + "body"
                            }
                        ]
                """
        self.log_step("step4","组装切片")
        final_chunks: List[dict[str, Any]] = []  # 组装后的切片列表
        for section in final_sections:
            chunk = {
                "title": section["title"],
                "parent_title": section["parent_title"],
                "file_title": section["file_title"],
                "content":section["title"] + "\n\n" + section["body"]
               }
            #3.判断part是否存在
            if "part" in section:
                chunk["part"] = section.get("part")

            final_chunks.append(chunk)
        return final_chunks

    # ------------------------------------------------------------------ #
    #                       日志 & 备份
    # ------------------------------------------------------------------ #
    def _log_summary(self, md_content: str, final_chunks: List[dict[str, Any]], max_content_length: int):
        self.log_step("step5","日志 & 备份")
        lines_count = md_content.count("\n") + 1
        self.logger.info(f"原文档行数:  {lines_count}")
        self.logger.info(f"最终切分章节数: {len(final_chunks)}")
        self.logger.info(f"最大切分长度: {max_content_length}")

        if final_chunks:
            self.logger.info("章节预览：")
            for i, sec in enumerate(final_chunks[:5]):
                title = sec.get("title","")[:30]
                self.logger.info(f"  {i + 1}.{title}...")
            if len(final_chunks) > 5:
                self.logger.info(f" ...还有{len(final_chunks) - 5} 个章节")


    def _backup_chunks(self, state, final_chunks):
        """将切分结果备份到json文件"""
        self.log_step("step6","备份切分结果")
        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置file_dir,跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)    # json.dump() 具有写操作能力    注意： 不是json.dumps()
            self.logger.info(f"已备份到: {output_path}")

        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()

    node = DocumentSplitNode()
    file_path = r"D:\Code\shopkeeper_brain_file\tmp_dir\万用表RS-12的使用\auto\万用表RS-12的使用_new.md"   #被切分文档
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    state = {
        "file_title": "万用表RS-12的使用_new",
        "md_content": content,
        "file_dir": r"D:\Code\shopkeeper_brain_file\tmp_dir\万用表RS-12的使用\auto"
    }

    node.process(state)


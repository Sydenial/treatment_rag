"""
数据准备模块
"""

import logging
import hashlib
from typing import List, Dict, Any
import json
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import uuid
import re
from typing import Optional

logger = logging.getLogger(__name__)

class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""
    # 统一维护的分类与难度配置，供外部复用，避免关键词重复定义
    CATEGORY_MAPPING = {
        'fracture': '骨折',
        'hemangioma': '血管瘤',
        'infection': '感染',
        'intervertebral': '椎间盘问题',
        'malignant_tumor': '恶性肿瘤',
        'others': '其他类型疾病',
    }
    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    
    def __init__(self, data_path: str):
        """
        初始化数据准备模块
        
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档（完整食谱）
        self.chunks: List[Document] = []     # 子文档（按标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射

    def clean_academic_markdown(self, text: str) -> str:
        """
        专门针对学术论文/病例报告 Markdown 的清洗函数
        1. 去除 References 及其之后的内容
        2. 去除 HTML 图片/表格标签
        3. 去除无用的元数据
        """
        if not text:
            return ""

        # --- 1. 截断 References (参考文献) ---
        # 匹配模式：行首 (#) + 空格 + Reference/Bibliography (忽略大小写)
        # 你的样本中是 "## References"
        ref_pattern = r'(?i)^#+\s*(references|bibliography|literature cited|works cited).*'

        # re.split 会将文本分为 [正文, 参考文献标题, 参考文献内容...]
        # 我们只需要 split 后的第一部分（正文）
        parts = re.split(ref_pattern, text, flags=re.MULTILINE)
        cleaned_text = parts[0]

        # --- 2. 去除 HTML 标签和图片 ---
        # 你的样本中包含很多 <div style="..."> 和 <img ...>
        # 提取纯文本任务不需要图片占位符
        html_tag_pattern = r'<[^>]+>'
        cleaned_text = re.sub(html_tag_pattern, '', cleaned_text)

        # --- 3. 去除 Markdown 图片链接 (可选) ---
        # 如果有 ![]() 格式
        md_image_pattern = r'!\[.*?\]\(.*?\)'
        cleaned_text = re.sub(md_image_pattern, '', cleaned_text)

        # --- 4. 去除多余空行 ---
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        return cleaned_text.strip()
    
    def load_documents(self) -> List[Document]:
        """
        加载文档数据
        
        Returns:
            加载的文档列表
        """
        logger.info(f"正在从 {self.data_path} 加载文档...")
        
        # 直接读取Markdown文件以保持原始格式
        documents = []
        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.rglob("*.md"):
            try:
                # 直接读取文件内容，保持Markdown格式
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                cleaned_content = self.clean_academic_markdown(content)

                # 为每个父文档分配确定性的唯一ID（基于数据根目录的相对路径）
                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
                except Exception:
                    relative_path = Path(md_file).as_posix()
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                # 创建Document对象
                doc = Document(
                    page_content=cleaned_content,
                    metadata={
                        "source": str(md_file),
                        "parent_id": parent_id,
                        "doc_type": "parent"  # 标记为父文档
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")
        
        # 增强文档元数据
        for doc in documents:
            self._enhance_metadata(doc)
        
        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def _enhance_metadata(self, doc: Document):
        """
        增强文档元数据
        
        Args:
            doc: 需要增强元数据的文档
        """
        file_path = Path(doc.metadata.get('source', '')).parent
        path_parts = file_path.parts
        
        # 提取菜品分类
        doc.metadata['category'] = '其他'
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata['category'] = value
                break

        # 提取菜品名称
        doc.metadata['case_report_id'] = file_path.stem


    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """对外提供支持的分类标签列表"""
        return cls.CATEGORY_LABELS


    def chunk_documents(self) -> List[Document]:
        """
        Markdown结构感知分块

        Returns:
            分块后的文档列表
        """
        logger.info("正在进行Markdown结构感知分块...")

        if not self.documents:
            raise ValueError("请先加载文档")

        # 使用Markdown标题分割器
        chunks = self._markdown_header_split()

        # 为每个chunk添加基础元数据
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                # 如果没有chunk_id（比如分割失败的情况），则生成一个
                chunk.metadata['chunk_id'] = str(uuid.uuid4())
            chunk.metadata['batch_index'] = i  # 在当前批次中的索引
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"Markdown分块完成，共生成 {len(chunks)} 个chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器进行结构化分割

        Returns:
            按标题结构分割的文档列表
        """
        # 定义要分割的标题层级
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]

        # 创建Markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题，便于理解上下文
        )

        all_chunks = []

        for doc in self.documents:
            try:
                # 检查文档内容是否包含Markdown标题
                content_preview = doc.page_content[:200]
                has_headers = any(line.strip().startswith('#') for line in content_preview.split('\n'))

                if not has_headers:
                    logger.warning(f"文档 {doc.metadata.get('case_report_id', '未知')} 内容中没有发现Markdown标题")
                    logger.debug(f"内容预览: {content_preview}")

                # 对每个文档进行Markdown分割
                md_chunks = markdown_splitter.split_text(doc.page_content)

                logger.debug(f"文档 {doc.metadata.get('case_report_id', '未知')} 分割成 {len(md_chunks)} 个chunk")

                # 如果没有分割成功，说明文档可能没有标题结构
                if len(md_chunks) <= 1:
                    logger.warning(f"文档 {doc.metadata.get('case_report_id', '未知')} 未能按标题分割，可能缺少标题结构")

                # 为每个子块建立与父文档的关系
                parent_id = doc.metadata["parent_id"]

                for i, chunk in enumerate(md_chunks):
                    # 为子块分配唯一ID
                    child_id = str(uuid.uuid4())

                    # 合并原文档元数据和新的标题元数据
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",  # 标记为子文档
                        "chunk_index": i      # 在父文档中的位置
                    })

                    # 建立父子映射关系
                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)

            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 如果Markdown分割失败，将整个文档作为一个chunk
                all_chunks.append(doc)

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

    def filter_documents_by_category(self, category: str) -> List[Document]:
        """
        按分类过滤文档
        
        Args:
            category: 疾病种类
            
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('category') == category]

    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        if not self.documents:
            return {}

        categories = {}

        for doc in self.documents:
            # 统计分类
            category = doc.metadata.get('category', '未知')
            categories[category] = categories.get(category, 0) + 1


        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': categories,
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }
    
    def export_metadata(self, output_path: str):
        """
        导出元数据到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        import json
        
        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'case_report_id': doc.metadata.get('case_report_id'),
                'category': doc.metadata.get('category'),
                'content_length': len(doc.page_content)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子块获取对应的父文档（智能去重）

        Args:
            child_chunks: 检索到的子块列表

        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {}
        parent_docs_map = {}

        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 缓存父文档（避免重复查找）
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按相关性排序（匹配次数多的排在前面）
        sorted_parent_ids = sorted(parent_relevance.keys(),
                                 key=lambda x: parent_relevance[x],
                                 reverse=True)

        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            case_report_id = doc.metadata.get('case_report_id', '未知疾病')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{case_report_id}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs


class GuidelineDataPreparationModule:
    """指南数据准备模块 - 负责层级化数据的加载、清洗和切分"""

    def __init__(self, data_path: str):
        """
        初始化
        Args:
            data_path: 数据根目录路径
        """
        self.data_path = Path(data_path)
        self.documents: List[Document] = []  # 父文档 (对应物理文件 partX.md)
        self.chunks: List[Document] = []  # 子文档 (切分后的知识点)
        self.parent_child_map: Dict[str, str] = {}  # 子ID -> 父ID

    def clean_guideline_markdown(self, text: str) -> str:
        """
        针对指南Markdown的清洗
        """
        if not text:
            return ""

        # 1. 截断参考文献 (指南中参考文献通常在最后)
        ref_pattern = r'(?i)^#+\s*(references|bibliography|literature cited).*'
        parts = re.split(ref_pattern, text, flags=re.MULTILINE)
        cleaned_text = parts[0]

        # 2. 去除 HTML 标签 (保留换行)
        cleaned_text = re.sub(r'<br\s*/?>', '\n', cleaned_text)  # br转换行
        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)  # 去除其他tag

        # 3. 去除 Markdown 图片链接 (保留图片说明文字作为上下文可能有用，这里选择去除链接)
        cleaned_text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', cleaned_text)

        # 4. 规范化空白字符
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        return cleaned_text.strip()

    def _camel_case_split(self, identifier: str) -> str:
        """
        将 PascalCase/CamelCase 转换为自然语言
        例: AdultIsthmicSpondylolisthesis -> Adult Isthmic Spondylolisthesis
        """
        # 在大写字母前加空格，但忽略字符串开头
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return " ".join([m.group(0) for m in matches])

    def _parse_path_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        核心逻辑：从文件路径解析层级元数据
        假设结构: Root / BookName / Index_ChapterName / partX.md
        """
        try:
            # 获取相对于数据根目录的路径部分
            rel_path = file_path.relative_to(self.data_path)
            parts = rel_path.parts  # ('AdultIs...', 'B_Diagnosis...', 'part1.md')

            if len(parts) < 3:
                # 处理异常层级，至少要有 书/章/文件
                return {"is_valid": False}

            # 1. 解析书名 (Book)
            raw_book_name = parts[0]
            book_name = self._camel_case_split(raw_book_name)

            # 2. 解析章节 (Chapter)
            raw_chapter_folder = parts[1]  # e.g., "B_Diagnosis_Imaging"
            if '_' in raw_chapter_folder:
                chapter_index, chapter_name_raw = raw_chapter_folder.split('_', 1)
                chapter_name = chapter_name_raw.replace('_', ' ')
            else:
                chapter_index = "0"
                chapter_name = raw_chapter_folder

            # 3. 解析分卷 (Part)
            file_name = parts[-1]  # e.g., "part1.md"
            part_index = "1"
            part_match = re.search(r'part(\d+)', file_name, re.IGNORECASE)
            if part_match:
                part_index = part_match.group(1)

            return {
                "is_valid": True,
                "book_name": book_name,  # 书名
                "chapter_index": chapter_index,  # 章节序号 (B)
                "chapter_name": chapter_name,  # 章节名 (Diagnosis Imaging)
                "part_index": int(part_index),  # 分卷号 (1)
                "source_file": file_name,
                "hierarchy_string": f"{book_name} > {chapter_index}. {chapter_name} > Part {part_index}"  # 方便LLM理解来源
            }
        except Exception as e:
            logger.warning(f"解析路径元数据失败 {file_path}: {e}")
            return {"is_valid": False}

    def load_documents(self) -> List[Document]:
        """加载并解析指南文档"""
        logger.info(f"正在从 {self.data_path} 加载指南数据...")

        documents = []

        # 遍历所有 md 文件
        for md_file in self.data_path.rglob("*.md"):
            try:
                # 1. 解析路径元数据
                meta = self._parse_path_metadata(md_file)
                if not meta["is_valid"]:
                    continue

                # 2. 读取并清洗内容
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                cleaned_content = self.clean_guideline_markdown(content)

                # 3. 生成父文档ID (基于相对路径的哈希，确保幂等性)
                rel_path_str = md_file.relative_to(self.data_path).as_posix()
                parent_id = hashlib.md5(rel_path_str.encode("utf-8")).hexdigest()

                # 4. 构建文档对象
                doc_metadata = {
                    "source": str(md_file),
                    "parent_id": parent_id,
                    "doc_type": "parent",
                    # 注入解析出的层级信息
                    "book_name": meta["book_name"],
                    "chapter_index": meta["chapter_index"],
                    "chapter_name": meta["chapter_name"],
                    "part_index": meta["part_index"],
                    "citation_source": meta["hierarchy_string"]  # 供RAG引用的字段
                }

                doc = Document(
                    page_content=cleaned_content,
                    metadata=doc_metadata
                )
                documents.append(doc)

            except Exception as e:
                logger.error(f"处理文件 {md_file} 时发生错误: {e}")

        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个指南文件 (Parts)")
        return documents

    def chunk_documents(self) -> List[Document]:
        """
        结构化分块 (Markdown Header Split)
        """
        logger.info("正在进行指南结构化分块...")

        if not self.documents:
            raise ValueError("请先调用 load_documents()")

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题在正文中，对理解上下文很有帮助
        )

        all_chunks = []

        for doc in self.documents:
            parent_id = doc.metadata["parent_id"]

            # 执行切分
            # 注意：HeaderSplitter 会把标题放入 metadata，我们需要合并回去
            md_chunks = markdown_splitter.split_text(doc.page_content)

            for i, chunk in enumerate(md_chunks):
                # 生成子块ID
                child_id = str(uuid.uuid4())

                # 1. 继承父文档的所有元数据 (Book, Chapter, etc.)
                chunk.metadata.update(doc.metadata)

                # 2. 添加子块特有元数据
                chunk.metadata.update({
                    "chunk_id": child_id,
                    "doc_type": "child",
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content)
                })

                # 3. 处理 Header 元数据 (将 Header 1/2/3 组合成一个 context 字段)
                headers = []
                if "Header 1" in chunk.metadata: headers.append(chunk.metadata["Header 1"])
                if "Header 2" in chunk.metadata: headers.append(chunk.metadata["Header 2"])
                if "Header 3" in chunk.metadata: headers.append(chunk.metadata["Header 3"])

                # 构建一个语义更丰富的 context 字符串，存入 metadata 供 向量检索 增强使用
                # 格式: Adult Isthmic Spondylolisthesis > Diagnosis > Header 1 > Header 2
                full_context = f"{doc.metadata['book_name']} > {doc.metadata['chapter_name']} > {' > '.join(headers)}"
                chunk.metadata["semantic_context"] = full_context

                # 记录映射
                self.parent_child_map[child_id] = parent_id
                all_chunks.append(chunk)

        self.chunks = all_chunks
        logger.info(f"分块完成，共生成 {len(all_chunks)} 个知识切片")
        return all_chunks

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据检索到的子块，反查完整的 partX.md 文档 (父文档)
        用于 RAG 的 "Parent Document Retriever" 模式
        """
        parent_ids = set()
        retrieved_parents = []

        # 建立父文档索引以便快速查找
        parent_map = {doc.metadata["parent_id"]: doc for doc in self.documents}

        for chunk in child_chunks:
            p_id = chunk.metadata.get("parent_id")
            if p_id and p_id not in parent_ids:
                if p_id in parent_map:
                    parent_doc = parent_map[p_id]
                    parent_ids.add(p_id)
                    retrieved_parents.append(parent_doc)

        logger.info(f"溯源: 从 {len(child_chunks)} 个切片还原出 {len(retrieved_parents)} 个父文档上下文")
        return retrieved_parents

    def export_metadata_report(self, output_path: str):
        """导出知识库结构报告"""
        report = []
        for doc in self.documents:
            report.append({
                "book": doc.metadata.get("book_name"),
                "chapter": f"{doc.metadata.get('chapter_index')}. {doc.metadata.get('chapter_name')}",
                "file": doc.metadata.get("source_file"),
                "chunks_count": len([c for c in self.chunks if c.metadata["parent_id"] == doc.metadata["parent_id"]])
            })

        # 按章节排序导出
        report.sort(key=lambda x: (x['book'], x['chapter'], x['file']))

        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(report, f, ensure_ascii=False, indent=2)
        # logger.info(f"结构报告已导出至 {output_path}")

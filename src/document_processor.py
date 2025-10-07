# document_processor.py
from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    CodeSplitter,
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
    SemanticSplitterNodeParser
)
from config_manager import get_config_manager
from llama_index.readers.web import SimpleWebPageReader


class DocumentProcessor:
    """文档处理类"""

    def __init__(self, logger, debug_monitor=None):
        self.logger = logger
        self.debug_monitor = debug_monitor

        # 从配置文件读取配置
        config_manager = get_config_manager()

        file_path = config_manager.get("Document", "file_path")
        self.chunk_size = config_manager.get_int("Document", "chunk_size", 500)
        self.chunk_overlap = config_manager.get_int("Document", "chunk_overlap", 20)

        # 获取文档加载方式配置
        self.load_mode = config_manager.get("Document", "load_mode", "file")

        # 获取分割器类型配置
        self.splitter_type = config_manager.get("Document", "splitter_type", "sentence")

        # 获取网络URL配置
        web_urls = config_manager.get("Document", "web_urls")
        if web_urls:
            self.web_urls = [url.strip() for url in web_urls.split(",") if url.strip()]
            self.logger.log_message(f"解析到 {len(self.web_urls)} 个URL")
        else:
            self.web_urls = []

        self.logger.log_message(
            f"文档处理器配置 - 加载模式: {self.load_mode}, 分割器类型: {self.splitter_type}, 分块大小: {self.chunk_size}, 重叠: {self.chunk_overlap}")

        # 设置文件路径
        if file_path:
            self.file_paths = [path.strip() for path in file_path.split(",") if path.strip()]

    def load_documents(self):
        """根据配置加载文档"""
        try:
            # 添加日志输出当前配置状态
            self.logger.log_message(f"当前加载模式: {self.load_mode}")
            self.logger.log_message(f"Web URLs: {self.web_urls}")
            if self.load_mode == "web":
                return self.load_documents_from_web(self.web_urls)
            elif self.load_mode == "url":
                return self.load_documents_from_url(self.web_urls)
            else:  # 默认file模式
                self.logger.log_message("开始加载本地文档...")
                reader = SimpleDirectoryReader(input_files=self.file_paths)
                documents = reader.load_data()

                if not documents:
                    self.logger.log_message("警告: 未加载到任何文档", "WARNING")
                    raise Exception("未加载到任何文档")

                self.logger.log_message(f"成功加载 {len(documents)} 个文档")

                # 输出文档内容预览
                self.logger.log_message("文档内容预览:")
                for i, doc in enumerate(documents[:3]):  # 只显示前3个文档
                    self.logger.log_message(f"文档 {i + 1}: {doc.text[:200]}...")  # 显示前200个字符

                return documents
        except Exception as e:
            self.logger.log_message(f"加载文档失败: {e}", "ERROR")
            raise

    def load_documents_from_url(self, urls: List[str]):
        """
        从网络URL加载文档

        Args:
            urls: 文档URL列表

        Returns:
            list: 加载的文档列表
        """
        try:
            self.logger.log_message("开始从网络加载文档...")
            reader = SimpleDirectoryReader(input_files=urls)
            documents = reader.load_data()

            if not documents:
                self.logger.log_message("警告: 未从网络加载到任何文档", "WARNING")
                raise Exception("未从网络加载到任何文档")

            self.logger.log_message(f"成功从网络加载 {len(documents)} 个文档")

            # 输出文档内容预览
            self.logger.log_message("网络文档内容预览:")
            for i, doc in enumerate(documents[:3]):  # 只显示前3个文档
                self.logger.log_message(f"文档 {i + 1}: {doc.text[:200]}...")  # 显示前200个字符

            return documents
        except Exception as e:
            self.logger.log_message(f"从网络加载文档失败: {e}", "ERROR")
            raise

    def load_documents_from_web(self, urls: List[str]):
        """
        从网页加载数据

        Args:
            urls: 网页URL列表

        Returns:
            list: 加载的文档列表
        """
        try:
            self.logger.log_message("开始从网页加载数据...")
            reader = SimpleWebPageReader()
            documents = reader.load_data(urls)

            if not documents:
                self.logger.log_message("警告: 未从网页加载到任何数据", "WARNING")
                raise Exception("未从网页加载到任何数据")

            self.logger.log_message(f"成功从网页加载 {len(documents)} 个文档")

            # 输出文档内容预览
            self.logger.log_message("网页内容预览:")
            for i, doc in enumerate(documents[:3]):  # 只显示前3个文档
                self.logger.log_message(f"文档 {i + 1}: {doc.text[:200]}...")  # 显示前200个字符

            return documents
        except Exception as e:
            self.logger.log_message(f"从网页加载数据失败: {e}", "ERROR")
            raise

    def create_node_parser(self):
        """
        根据配置创建相应的节点分割器

        Returns:
            BaseNodeParser: 节点分割器实例
        """
        if self.splitter_type == "sentence":
            # SentenceSplitter: 基于句子的分割器，适用于一般文本
            self.logger.log_message("使用 SentenceSplitter 进行文档分割")
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        elif self.splitter_type == "token":
            # TokenTextSplitter: 基于token的分割器，适用于控制token数量的场景
            self.logger.log_message("使用 TokenTextSplitter 进行文档分割")
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        elif self.splitter_type == "code":
            # CodeSplitter: 专门针对代码的分割器，保持代码结构完整性
            self.logger.log_message("使用 CodeSplitter 进行代码文档分割")
            # 注意：CodeSplitter通常需要指定语言
            language = "python"  # 可从配置获取
            return CodeSplitter(
                language=language
            )

        elif self.splitter_type == "sentence_window":
            # SentenceWindowNodeParser: 创建句子窗口节点，适用于需要上下文的场景
            self.logger.log_message("使用 SentenceWindowNodeParser 进行文档分割")
            return SentenceWindowNodeParser(
                window_size=3,  # 可从配置获取
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            )

        elif self.splitter_type == "hierarchical":
            # HierarchicalNodeParser: 分层节点解析器，适用于构建层次化索引
            self.logger.log_message("使用 HierarchicalNodeParser 进行文档分割")
            return HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128]  # 可从配置获取
            )

        elif self.splitter_type == "semantic":
            # SemanticSplitterNodeParser: 语义分割器，基于嵌入相似度进行分割
            self.logger.log_message("使用 SemanticSplitterNodeParser 进行文档分割")
            # 需要嵌入模型，这里简化处理
            return SemanticSplitterNodeParser(
                buffer_size=1,  # 可从配置获取
                breakpoint_percentile_threshold=95  # 可从配置获取
            )

        else:
            # 默认使用SentenceSplitter
            self.logger.log_message("使用默认的 SentenceSplitter 进行文档分割")
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

    def split_documents(self, documents):
        """分割文档"""
        try:
            self.logger.log_message("开始分割文档...")
            if not documents:
                raise Exception("没有文档可供分割")

            # 根据配置创建相应的节点分割器
            node_parser = self.create_node_parser()
            nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

            if not nodes:
                self.logger.log_message("警告: 文档分割后未生成任何节点", "WARNING")
                raise Exception("文档分割后未生成任何节点")

            self.logger.log_message(f"文档已分割为 {len(nodes)} 个节点")

            # 输出分割后的内容
            self.logger.log_message("文档分割结果:")
            for i, node in enumerate(nodes):
                self.logger.log_message(f"节点 {i + 1}: {node.text[:200]}...")  # 显示前200个字符
                if i >= 9:  # 只显示前10个节点以避免输出过多
                    self.logger.log_message(f"...还有 {len(nodes) - 10} 个节点")
                    break

            return nodes
        except Exception as e:
            self.logger.log_message(f"分割文档失败: {e}", "ERROR")
            raise

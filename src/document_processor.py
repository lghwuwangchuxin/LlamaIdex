# document_processor.py
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter


class DocumentProcessor:
    """文档处理类"""

    def __init__(self, logger, debug_monitor=None):
        self.logger = logger
        self.debug_monitor = debug_monitor
        self.file_paths = [
            "/Users/liuguanghu/PythonPorject/LlamaIdex/data/xiaomaiUltra.txt",
            "/Users/liuguanghu/PythonPorject/LlamaIdex/data/yiyan.txt",
            "/Users/liuguanghu/PythonPorject/LlamaIdex/data/Qwen1.5-110B.pdf",
        ]

    def load_documents(self):
        """加载与读取文档"""
        try:
            self.logger.log_message("开始加载文档...")
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

    def split_documents(self, documents):
        """分割文档"""
        try:
            self.logger.log_message("开始分割文档...")
            if not documents:
                raise Exception("没有文档可供分割")

            node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
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

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from pymilvus import connections, utility, Collection
import logging
from datetime import datetime
import traceback


class Logger:
    """日志管理类"""

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.timestamp_format = "%Y-%m-%d %H:%M:%S"

    def log_message(self, message, level="INFO"):
        """统一日志输出格式"""
        timestamp = datetime.now().strftime(self.timestamp_format)
        print(f"[{timestamp}] [{level}] {message}")


class ModelManager:
    """模型管理类"""

    def __init__(self, logger):
        self.logger = logger

    def setup_models(self):
        """设置Ollama模型"""
        try:
            self.logger.log_message("开始配置 Ollama 模型...")

            # 配置语言模型
            self.logger.log_message("配置语言模型...")
            Settings.llm = Ollama(
                model="llama3",
                base_url="http://127.0.0.1:11434",
                request_timeout=120.0
            )
            self.logger.log_message("语言模型配置完成")

            # 配置嵌入模型
            self.logger.log_message("配置嵌入模型...")
            Settings.embed_model = OllamaEmbedding(
                model_name="nomic-embed-text",
                base_url="http://127.0.0.1:11434"
            )
            self.logger.log_message("嵌入模型配置完成")

            self.logger.log_message("Ollama模型配置成功")
            return True
        except Exception as e:
            self.logger.log_message(f"Ollama模型配置失败: {e}", "ERROR")
            return False


class MilvusManager:
    """Milvus数据库管理类"""

    def __init__(self, logger):
        self.logger = logger
        self.host = "127.0.0.1"
        self.port = "19530"
        self.collection_name = "ragdb"
        self.db_name = "rag_database"

    def connect(self):
        """连接到Milvus服务"""
        try:
            self.logger.log_message("开始连接 Milvus 服务...")
            connections.connect("default", host=self.host, port=self.port)
            self.logger.log_message("成功连接到 Milvus")
            return True
        except Exception as e:
            self.logger.log_message(f"连接 Milvus 失败: {e}", "ERROR")
            return False

    def check_database_support(self):
        """检查是否支持多数据库功能"""
        self.logger.log_message("检查 Milvus 数据库支持功能...")
        has_list = hasattr(utility, 'list_database')
        has_create = hasattr(utility, 'create_database')
        support = has_list and has_create
        self.logger.log_message(
            f"数据库支持检查结果: list_database={has_list}, create_database={has_create}, 支持多数据库={support}")
        return support

    def create_database(self):
        """创建数据库"""
        self.logger.log_message(f"开始创建数据库: {self.db_name}")
        if not self.check_database_support():
            self.logger.log_message("当前版本不支持多数据库功能，使用默认数据库", "WARNING")
            return False

        try:
            existing_databases = utility.list_database()
            self.logger.log_message(f"现有数据库列表: {existing_databases}")

            if self.db_name not in existing_databases:
                utility.create_database(db_name=self.db_name)
                self.logger.log_message(f"数据库 '{self.db_name}' 创建成功")
                return True
            else:
                self.logger.log_message(f"数据库 '{self.db_name}' 已存在")
                return True
        except Exception as e:
            self.logger.log_message(f"创建数据库失败: {e}", "ERROR")
            raise

    def setup_vector_store(self, dim):
        """设置向量存储"""
        try:
            self.logger.log_message("开始准备 Milvus 向量存储...")

            # 检查并创建数据库（如果支持）
            if self.check_database_support():
                self.create_database()
                # 切换到指定数据库
                self.logger.log_message("断开当前连接...")
                connections.disconnect("default")
                self.logger.log_message("重新连接到指定数据库...")
                connections.connect("default", host=self.host, port=self.port, db_name=self.db_name)
                self.logger.log_message(f"成功切换到数据库 '{self.db_name}'")

            # 检查集合是否存在，如果存在则删除
            self.logger.log_message("检查并清理现有集合...")
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                self.logger.log_message(f"已删除现有的集合 {self.collection_name}")
            else:
                self.logger.log_message(f"集合 {self.collection_name} 不存在，无需删除")

            # 创建 Milvus 向量存储
            self.logger.log_message("创建 Milvus 向量存储...")
            vector_store = MilvusVectorStore(
                uri=f"http://{self.host}:{self.port}",
                collection_name=self.collection_name,
                dim=dim,
                overwrite=True
            )
            self.logger.log_message(f"成功创建 Milvus 集合 {self.collection_name}，维度: {dim}")
            return vector_store
        except Exception as e:
            self.logger.log_message(f"创建 Milvus 向量存储失败: {e}", "ERROR")
            raise


class DocumentProcessor:
    """文档处理类"""

    def __init__(self, logger):
        self.logger = logger
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


class RAGSystem:
    """RAG系统主类"""

    def __init__(self):
        self.logger = Logger()
        self.model_manager = ModelManager(self.logger)
        self.milvus_manager = MilvusManager(self.logger)
        self.document_processor = DocumentProcessor(self.logger)
        self.vector_store = None
        self.index = None
        self.query_engine = None

    def initialize_system(self):
        """初始化RAG系统"""
        try:
            # 1. 设置模型
            if not self.model_manager.setup_models():
                return False

            # 2. 连接Milvus
            if not self.milvus_manager.connect():
                return False

            # 3. 加载和处理文档
            documents = self.document_processor.load_documents()
            nodes = self.document_processor.split_documents(documents)

            # 4. 获取嵌入模型维度
            self.logger.log_message("正在测试嵌入模型维度...")
            try:
                test_embedding = Settings.embed_model.get_text_embedding(
                    "This is a test sentence to determine embedding dimension.")
                actual_dim = len(test_embedding)
                self.logger.log_message(f"嵌入模型 'nomic-embed-text' 实际维度: {actual_dim}")
            except Exception as dim_error:
                self.logger.log_message(f"获取嵌入模型维度失败: {dim_error}", "ERROR")
                actual_dim = 768  # 默认维度
                self.logger.log_message(f"使用默认维度: {actual_dim}")

            # 5. 设置向量存储
            self.vector_store = self.milvus_manager.setup_vector_store(actual_dim)

            # 6. 创建索引
            self.logger.log_message("开始创建向量存储索引...")
            if not nodes:
                raise Exception("没有节点数据可用于创建索引")

            self.logger.log_message(f"准备插入 {len(nodes)} 个节点到向量数据库")

            # 检查第一个节点的内容和嵌入
            if nodes:
                first_node = nodes[0]
                self.logger.log_message(f"第一个节点文本预览: {first_node.text[:100]}...")

                # 测试嵌入生成
                try:
                    embedding = Settings.embed_model.get_text_embedding(first_node.text[:100])
                    self.logger.log_message(f"成功生成嵌入，维度: {len(embedding)}")
                except Exception as emb_error:
                    self.logger.log_message(f"嵌入生成测试失败: {emb_error}", "ERROR")

            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.logger.log_message("存储上下文创建成功")

            self.index = VectorStoreIndex(nodes, storage_context=storage_context)
            self.logger.log_message("索引创建成功")

            # 创建索引后立即检查数据
            try:
                collection = Collection(name=self.milvus_manager.collection_name)
                collection.load()
                post_creation_count = collection.num_entities
                self.logger.log_message(f"索引创建后集合实体数: {post_creation_count}")
            except Exception as check_error:
                self.logger.log_message(f"索引创建后检查失败: {check_error}", "ERROR")

            # 7. 创建查询引擎
            self.logger.log_message("开始构造查询引擎...")
            self.query_engine = self.index.as_query_engine()
            self.logger.log_message("查询引擎已准备就绪")

            return True

        except Exception as e:
            self.logger.log_message(f"系统初始化失败: {e}", "ERROR")
            self.logger.log_message(f"详细错误信息: {traceback.format_exc()}", "ERROR")
            return False

    def run_interactive_qa(self):
        """运行交互式问答"""
        self.logger.log_message("欢迎使用AI助手！输入 'exit' 退出程序。")
        while True:
            try:
                user_input = input("\n问题：").strip()
                if user_input.lower() == "exit":
                    self.logger.log_message("用户退出程序")
                    print("再见！")
                    break

                if not user_input:
                    print("请输入有效问题")
                    continue

                self.logger.log_message(f"正在处理用户问题: {user_input}")
                response = self.query_engine.query(user_input)
                print("AI助手：", response.response)

                # 显示来源信息
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    print("\n来源信息：")
                    for i, node in enumerate(response.source_nodes[:3]):  # 显示前3个来源
                        print(f"{i + 1}. {node.text[:200]}...")  # 显示前200个字符
                print("-" * 50)

            except KeyboardInterrupt:
                self.logger.log_message("程序被用户中断")
                print("\n程序被用户中断")
                break
            except Exception as e:
                self.logger.log_message(f"查询过程中出错: {e}", "ERROR")
                print(f"查询出错：{e}")

    def cleanup(self):
        """清理资源"""
        try:
            self.logger.log_message("正在断开 Milvus 连接...")
            connections.disconnect("default")
            self.logger.log_message("已断开 Milvus 连接")
        except Exception as e:
            self.logger.log_message(f"断开连接时发生错误: {e}", "ERROR")

        self.logger.log_message("程序执行完成")


def main():
    """主函数"""
    rag_system = RAGSystem()

    try:
        # 初始化系统
        if rag_system.initialize_system():
            # 运行交互式问答
            rag_system.run_interactive_qa()
        else:
            print("系统初始化失败，无法启动问答功能")
    finally:
        # 清理资源
        rag_system.cleanup()


if __name__ == "__main__":
    main()

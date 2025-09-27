# milvus_manager.py
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import connections, utility, Collection


class MilvusManager:
    """Milvus数据库管理类"""

    def __init__(self, logger, debug_monitor=None):
        self.logger = logger
        self.debug_monitor = debug_monitor
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

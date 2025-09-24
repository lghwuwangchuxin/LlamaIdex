from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from pymilvus import Collection
import traceback


class RAGSystem:
    """RAG系统主类"""

    def __init__(self, logger, model_manager, milvus_manager, document_processor):
        self.logger = logger
        self.model_manager = model_manager
        self.milvus_manager = milvus_manager
        self.document_processor = document_processor
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
            from pymilvus import connections
            connections.disconnect("default")
            self.logger.log_message("已断开 Milvus 连接")
        except Exception as e:
            self.logger.log_message(f"断开连接时发生错误: {e}", "ERROR")

        self.logger.log_message("程序执行完成")

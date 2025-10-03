from typing import Any, List
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from pymilvus import Collection
import traceback
class RAGSystem:
    """RAG系统主类 - 实现完整的检索增强生成系统"""

    def __init__(self, logger, model_manager, milvus_manager, document_processor, monitoring_manager):
        """
        初始化RAG系统

        Args:
            logger: 日志管理器
            model_manager: 模型管理器
            milvus_manager: Milvus数据库管理器
            document_processor: 文档处理器
            monitoring_manager: 监控管理器
        """
        self.logger = logger
        self.model_manager = model_manager
        self.milvus_manager = milvus_manager
        self.document_processor = document_processor
        self.monitoring_manager = monitoring_manager
        self.vector_store = None
        self.index = None
        self.query_engine = None

    def initialize_system(self):
        """
        初始化RAG系统 - 完整的系统设置流程
        """
        # 根据监控模式执行相应初始化
        if self.monitoring_manager.is_langfuse_mode():
            return self._initialize_system_with_langfuse()
        else:
            return self._initialize_system_with_debug()

    def _initialize_system_with_langfuse(self):
        """使用Langfuse监控初始化系统"""
        trace = self.monitoring_manager.langfuse_create_trace(
            name="rag-system-initialization",
            metadata={
                "system": "RAGSystem",
                "phase": "initialization"
            }
        )

        try:
            # 1. 设置模型
            model_span = self.monitoring_manager.langfuse_create_span(trace, name="model-setup")
            if not self.model_manager.setup_models():
                model_span.end()
                return False
            # 在调用 end() 方法前添加空值检查
            if model_span is not None:
                model_span.end()

            # 2. 连接Milvus
            milvus_span = self.monitoring_manager.langfuse_create_span(trace, name="milvus-connection")
            if not self.milvus_manager.connect():
                milvus_span.end()
                return False
            # 在调用 end() 方法前添加空值检查
            if milvus_span is not None:
                milvus_span.end()

            # 3. 加载和处理文档
            document_span = self.monitoring_manager.langfuse_create_span(trace, name="document-processing")
            documents = self.document_processor.load_documents()
            nodes = self.document_processor.split_documents(documents)
            # 在调用 end() 方法前添加空值检查
            if document_span is not None:
                document_span.end()


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
            vector_span = self.monitoring_manager.langfuse_create_span(trace, name="vector-store-setup")
            self.vector_store = self.milvus_manager.setup_vector_store(actual_dim)
            if vector_span is not None:
                vector_span.end()

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

            index_span = self.monitoring_manager.langfuse_create_span(trace, name="index-creation")
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.logger.log_message("存储上下文创建成功")

            self.index = VectorStoreIndex(nodes, storage_context=storage_context)
            self.logger.log_message("索引创建成功")
            if index_span is not None:
                index_span.end()

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
            query_span = self.monitoring_manager.langfuse_create_span(trace, name="query-engine-setup")
            self.query_engine = self.index.as_query_engine()
            self.logger.log_message("查询引擎已准备就绪")
            if query_span is not None:
                query_span.end()

            self.monitoring_manager.langfuse_update_trace(
                trace,
                output="System initialized successfully"
            )

            return True

        except Exception as e:
            self.logger.log_message(f"系统初始化失败: {e}", "ERROR")
            self.logger.log_message(f"详细错误信息: {traceback.format_exc()}", "ERROR")
            self.monitoring_manager.langfuse_update_trace(
                trace,
                status_message=f"Initialization failed: {str(e)}",
                level="ERROR"
            )
            return False

    def _initialize_system_with_debug(self):
        """使用Debug监控初始化系统"""
        try:
            # 1. 设置模型
            if not self.model_manager.setup_models():
                return False

            # 2. 连接Milvus
            if not self.milvus_manager.connect():
                return False

            # 3. 加载和处理文档
            documents = self.document_processor.load_documents()

            # 监控文本分割事件
            nodes = self.document_processor.split_documents(documents)
            self.monitoring_manager.debug_monitor_text_split_event(
                [doc.text for doc in documents] if documents else []
            )
            self.monitoring_manager.debug_monitor_node_parse_event(nodes)

            # 4. 获取嵌入模型维度
            self.logger.log_message("正在测试嵌入模型维度...")
            try:
                test_embedding = Settings.embed_model.get_text_embedding(
                    "This is a test sentence to determine embedding dimension.")
                actual_dim = len(test_embedding)
                self.logger.log_message(f"嵌入模型 'nomic-embed-text' 实际维度: {actual_dim}")

                # 监控文本嵌入事件
                self.monitoring_manager.debug_monitor_text_embedding_event({
                    'dimension': actual_dim,
                    'text_count': 1
                })
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

                    # 监控文本嵌入事件
                    self.monitoring_manager.debug_monitor_text_embedding_event({
                        'dimension': len(embedding),
                        'text_count': 1
                    })
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

    def _safe_extract_response(self, response: Any) -> str:
        """
        安全地提取响应内容，处理各种响应类型

        Args:
            response: 可能是 Response、ChatResponse、str 等类型的响应对象

        Returns:
            str: 提取的响应文本
        """
        try:
            # 如果是字符串，直接返回
            if isinstance(response, str):
                return response

            # 如果是 Response 对象，且有 response 属性
            if hasattr(response, 'response'):
                response_content = response.response

                # 如果 response 属性是字符串
                if isinstance(response_content, str):
                    return response_content

                # 如果是 ChatResponse 或类似对象
                if hasattr(response_content, 'message') and hasattr(response_content.message, 'content'):
                    return response_content.message.content

                # 其他情况转换为字符串
                return str(response_content)

            # 如果是 ChatResponse 对象（直接就是，而不是在 response 属性中）
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                return response.message.content

            # 最后尝试转换为字符串
            return str(response)

        except Exception as e:
            self.logger.log_message(f"响应内容提取失败: {e}", "WARNING")
            return "无法获取响应内容"

    def _safe_extract_source_nodes(self, response: Any) -> List[Any]:
        """
        安全地提取源节点信息

        Args:
            response: 响应对象

        Returns:
            List: 源节点列表
        """
        try:
            # 优先尝试 source_nodes 属性
            if hasattr(response, 'source_nodes') and response.source_nodes:
                return response.source_nodes

            # 尝试 metadata 中的 source_nodes
            if hasattr(response, 'metadata') and isinstance(response.metadata, dict):
                nodes = response.metadata.get('source_nodes', [])
                if nodes:
                    return nodes

            # 尝试从 response 属性中查找
            if hasattr(response, 'response'):
                return self._safe_extract_source_nodes(response.response)

            return []

        except Exception as e:
            self.logger.log_message(f"源节点提取失败: {e}", "WARNING")
            return []

    def run_interactive_qa(self):
        """
        运行交互式问答 - 用户交互主循环
        """
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

                # 根据监控模式执行相应查询
                if self.monitoring_manager.is_langfuse_mode():
                    self._run_query_with_langfuse(user_input)
                else:
                    self._run_query_with_debug(user_input)

            except KeyboardInterrupt:
                self.logger.log_message("程序被用户中断")
                print("\n程序被用户中断")
                break
            except Exception as e:
                self.logger.log_message(f"查询过程中出错: {e}", "ERROR")
                print(f"查询出错：{e}")

    def _run_query_with_langfuse(self, user_input: str):
        """使用Langfuse监控执行查询"""
        trace = self.monitoring_manager.langfuse_create_trace(
            name="user-query",
            input=user_input,
            metadata={
                "user": "interactive_user",
                "session": "qa_session"
            }
        )

        try:
            response = self.query_engine.query(user_input)

            self.monitoring_manager.langfuse_update_trace(
                trace,
                output=response.response if hasattr(response, 'response') else str(response)
            )

            print("AI助手：", response.response)

            # 显示来源信息
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print("\n来源信息：")
                MAX_SOURCE_NODES = 3
                MAX_TEXT_LENGTH = 200
                for i, node in enumerate(response.source_nodes[:MAX_SOURCE_NODES]):
                    try:
                        if hasattr(node, 'text') and isinstance(node.text, str):
                            # 安全截取文本，避免在多字节字符中间截断
                            if len(node.text) > MAX_TEXT_LENGTH:
                                display_text = node.text[:MAX_TEXT_LENGTH] + "..."
                            else:
                                display_text = node.text
                            print(f"{i + 1}. {display_text}")
                        else:
                            print(f"{i + 1}. [无法获取有效文本内容]")
                    except Exception as e:
                        print(f"{i + 1}. [文本处理出错: {str(e)}]")

            print("-" * 50)
        except Exception as e:
            self.monitoring_manager.langfuse_update_trace(
                trace,
                status_message=f"Query failed: {str(e)}",
                level="ERROR"
            )
            raise e

    def _run_query_with_debug(self, user_input: str):
        """使用Debug监控执行查询"""
        # 监控查询引擎事件开始
        self.monitoring_manager.debug_monitor_query_engine_event(user_input)

        response = self.query_engine.query(user_input)

        # 安全地提取响应内容
        response_content = self._safe_extract_response(response)
        source_nodes = self._safe_extract_source_nodes(response)
        # 记录成功查询事件
        self.logger.log_message(f"查询成功: 问题长度={len(user_input)}, 响应长度={len(response_content)}")

        # 监控查询引擎事件完成
        self.monitoring_manager.debug_monitor_query_engine_event(user_input)
        print("AI助手：", response.response)

        # 显示来源信息
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print("\n来源信息：")
            MAX_SOURCE_NODES = 3
            MAX_TEXT_LENGTH = 200
            for i, node in enumerate(response.source_nodes[:MAX_SOURCE_NODES]):
                try:
                    if hasattr(node, 'text') and isinstance(node.text, str):
                        # 安全截取文本，避免在多字节字符中间截断
                        if len(node.text) > MAX_TEXT_LENGTH:
                            display_text = node.text[:MAX_TEXT_LENGTH] + "..."
                        else:
                            display_text = node.text
                        print(f"{i + 1}. {display_text}")
                    else:
                        print(f"{i + 1}. [无法获取有效文本内容]")
                except Exception as e:
                    print(f"{i + 1}. [文本处理出错: {str(e)}]")

            # 监控语义检索事件
            self.monitoring_manager.debug_monitor_semantic_retrieval_event(
                response.source_nodes,
                user_input
            )

        print("-" * 50)

    def cleanup(self):
        """
        清理资源 - 系统关闭时的清理工作
        """
        try:
            self.logger.log_message("正在断开 Milvus 连接...")
            from pymilvus import connections
            connections.disconnect("default")
            self.logger.log_message("已断开 Milvus 连接")
        except Exception as e:
            self.logger.log_message(f"断开连接时发生错误: {e}", "ERROR")

        self.logger.log_message("程序执行完成")

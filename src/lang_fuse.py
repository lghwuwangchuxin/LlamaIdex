import os
import logging
from typing import Optional
from langfuse import Langfuse
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from config_manager import get_config_manager
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangfuseManager:
    """Langfuse客户端管理器"""

    def __init__(self):
        self.langfuse = None
        self.is_authenticated = False

    def initialize(self) -> bool:
        """
        初始化Langfuse客户端和LlamaIndex插桩

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 1. 初始化Langfuse客户端
            if not self._init_langfuse():
                return False

            # 2. 验证连接
            if not self._verify_connection():
                return False

            # 3. 初始化LlamaIndex插桩
            self._init_instrumentation()

            logger.info("Langfuse client initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {str(e)}")
            return False

    def _init_langfuse(self) -> bool:
        """初始化Langfuse客户端"""
        try:
            # 从配置文件读取配置
            config = self._get_config_from_config_file()

            self.langfuse = Langfuse(**config)
            logger.info("Langfuse client created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create Langfuse client: {str(e)}")
            return False

    def _get_config_from_config_file(self) -> dict:
        """从配置文件获取配置"""
        config_manager = get_config_manager()

        config = {
            "public_key": config_manager.get("Langfuse", "public_key"),
            "secret_key": config_manager.get("Langfuse", "secret_key"),
            "host": config_manager.get("Langfuse", "host", "https://cloud.langfuse.com"),
        }

        # 移除None值的配置项
        config = {k: v for k, v in config.items() if v is not None}

        # 添加可选配置
        environment = config_manager.get("Langfuse", "environment")
        if environment:
            config["environment"] = environment

        return config

    def _verify_connection(self) -> bool:
        """验证Langfuse连接"""
        try:
            auth_result = self.langfuse.auth_check()
            if not auth_result:
                logger.error("Langfuse authentication failed")
                return False

            self.is_authenticated = True
            logger.info("Langfuse authentication successful")
            return True

        except Exception as e:
            logger.error(f"Connection verification failed: {str(e)}")
            return False

    def _init_instrumentation(self):
        """初始化LlamaIndex插桩"""
        try:
            # 检查是否已经插桩
            if not hasattr(self, '_instrumented'):
                LlamaIndexInstrumentor().instrument()
                self._instrumented = True
                logger.info("LlamaIndex instrumentation completed")
            else:
                logger.info("LlamaIndex already instrumented")

        except Exception as e:
            logger.error(f"Failed to instrument LlamaIndex: {str(e)}")
            raise

    def flush(self):
        """强制刷新所有待发送的数据"""
        try:
            if self.langfuse:
                self.langfuse.flush()
                logger.debug("Data flushed successfully")
        except Exception as e:
            logger.warning(f"Failed to flush data: {str(e)}")

    def shutdown(self):
        """关闭客户端并清理资源"""
        try:
            if self.langfuse:
                self.flush()
                # Langfuse客户端没有显式的关闭方法，但我们可以删除引用
                self.langfuse = None
                self.is_authenticated = False
                logger.info("Langfuse client shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")


# 全局实例
_langfuse_manager: Optional[LangfuseManager] = None


def get_langfuse_manager() -> LangfuseManager:
    """获取全局Langfuse管理器实例"""
    global _langfuse_manager
    if _langfuse_manager is None:
        _langfuse_manager = LangfuseManager()
    return _langfuse_manager


def initialize_langfuse() -> bool:
    """
    初始化Langfuse系统

    Returns:
        bool: 初始化是否成功
    """
    manager = get_langfuse_manager()
    return manager.initialize()


def get_langfuse_client() -> Optional[Langfuse]:
    """获取Langfuse客户端实例"""
    manager = get_langfuse_manager()
    return manager.langfuse if manager.is_authenticated else None
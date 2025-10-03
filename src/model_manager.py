# model_manager.py (修改后)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from config_manager import get_config_manager  # 添加导入

class ModelManager:
    """模型管理类"""

    def __init__(self, logger, debug_monitor=None):
        self.logger = logger
        self.debug_monitor = debug_monitor

    def setup_models(self):
        """设置Ollama模型"""
        try:
            # 从配置文件读取配置
            config_manager = get_config_manager()
            llm_model = config_manager.get("Model", "llm_model", "llama3")
            embedding_model = config_manager.get("Model", "embedding_model", "nomic-embed-text")
            ollama_base_url = config_manager.get("Model", "ollama_base_url", "http://127.0.0.1:11434")
            request_timeout = config_manager.get_float("Model", "request_timeout", 120.0)

            self.logger.log_message("开始配置 Ollama 模型...")

            # 配置语言模型
            self.logger.log_message("配置语言模型...")
            Settings.llm = Ollama(
                model=llm_model,
                base_url=ollama_base_url,
                request_timeout=request_timeout
            )
            self.logger.log_message("语言模型配置完成")

            # 配置嵌入模型
            self.logger.log_message("配置嵌入模型...")
            Settings.embed_model = OllamaEmbedding(
                model_name=embedding_model,
                base_url=ollama_base_url
            )
            self.logger.log_message("嵌入模型配置完成")

            self.logger.log_message("Ollama模型配置成功")
            return True
        except Exception as e:
            self.logger.log_message(f"Ollama模型配置失败: {e}", "ERROR")
            return False

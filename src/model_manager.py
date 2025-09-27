# model_manager.py
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings


class ModelManager:
    """模型管理类"""

    def __init__(self, logger, debug_monitor=None):
        self.logger = logger
        self.debug_monitor = debug_monitor

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

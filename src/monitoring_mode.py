# monitoring_mode.py
from enum import Enum
from typing import Optional, Any, List, Dict
from AdvancedLlamaDebugMonitor import AdvancedLlamaDebugMonitor
from lang_fuse import get_langfuse_manager, initialize_langfuse, get_langfuse_client
from langfuse import Langfuse

class MonitoringMode(Enum):
    """监控模式枚举"""
    DEBUG = "debug"  # 仅使用Debug监控
    LANGFUSE = "langfuse"  # 仅使用Langfuse监控


class MonitoringManager:
    """监控模式管理器"""

    def __init__(self):
        self.current_mode = MonitoringMode.DEBUG
        self.debug_monitor: Optional[AdvancedLlamaDebugMonitor] = None
        self.langfuse_manager = None
        self.is_langfuse_initialized = False

    def set_mode(self, mode: MonitoringMode, debug_monitor=None, langfuse_client=None):
        """设置监控模式"""
        self.current_mode = mode
        self.debug_monitor = debug_monitor

        # 使用 lang_fuse.py 中的逻辑初始化 Langfuse
        if mode == MonitoringMode.LANGFUSE:
            if initialize_langfuse():
                self.langfuse_manager = get_langfuse_manager()
                self.is_langfuse_initialized = True
                print("✅ Langfuse初始化成功")
            else:
                print("❌ Langfuse初始化失败")
                self.is_langfuse_initialized = False
        else:
            self.is_langfuse_initialized = False

    # Debug监控方法
    def debug_monitor_text_split_event(self, text_chunks: List[str] = None):
        """Debug模式下的文本分割事件监控"""
        if self.current_mode == MonitoringMode.DEBUG and self.debug_monitor:
            return self.debug_monitor.monitor_text_split_event(text_chunks)
        return None

    def debug_monitor_node_parse_event(self, nodes: List[Any] = None):
        """Debug模式下的节点解析事件监控"""
        if self.current_mode == MonitoringMode.DEBUG and self.debug_monitor:
            return self.debug_monitor.monitor_node_parse_event(nodes)
        return None

    def debug_monitor_text_embedding_event(self, embeddings_info: Dict[str, Any] = None):
        """Debug模式下的文本嵌入事件监控"""
        if self.current_mode == MonitoringMode.DEBUG and self.debug_monitor:
            return self.debug_monitor.monitor_text_embedding_event(embeddings_info)
        return None

    def debug_monitor_query_engine_event(self, query: str, response: Any = None):
        """Debug模式下的查询引擎事件监控"""
        if self.current_mode == MonitoringMode.DEBUG and self.debug_monitor:
            return self.debug_monitor.monitor_query_engine_event(query, response)
        return None

    def debug_monitor_semantic_retrieval_event(self, retrieved_nodes: List[Any] = None, query: str = None):
        """Debug模式下的语义检索事件监控"""
        if self.current_mode == MonitoringMode.DEBUG and self.debug_monitor:
            return self.debug_monitor.monitor_semantic_retrieval_event(retrieved_nodes, query)
        return None

    def langfuse_create_trace(self, name: str, **kwargs):
        """Langfuse模式下创建trace"""
        if self.current_mode == MonitoringMode.LANGFUSE and self.is_langfuse_initialized and self.langfuse_manager:
            try:
                langfuse_client: Langfuse = get_langfuse_client()
                if langfuse_client:
                    # 正确调用 Langfuse 实例的 trace 方法
                    return langfuse_client.trace(name=name, **kwargs)
            except Exception as e:
                # 如果创建trace失败，返回None而不是抛出异常
                return None
        return None

    def langfuse_create_span(self, trace, name: str, **kwargs):
        """Langfuse模式下创建span"""
        if self.current_mode == MonitoringMode.LANGFUSE and trace:
            try:
                return trace.span(name=name, **kwargs)
            except Exception as e:
                # 如果创建span失败，返回None而不是抛出异常
                return None
        return None

    def langfuse_update_trace(self, trace, **kwargs):
        """Langfuse模式下更新trace"""
        if self.current_mode == MonitoringMode.LANGFUSE and trace:
            try:
                trace.update(**kwargs)
            except Exception as e:
                # 静默处理更新失败
                pass

    def langfuse_flush(self):
        """Langfuse模式下刷新数据"""
        if self.current_mode == MonitoringMode.LANGFUSE and self.is_langfuse_initialized and self.langfuse_manager:
            try:
                self.langfuse_manager.flush()
            except Exception as e:
                # 静默处理刷新失败
                pass

    def langfuse_shutdown(self):
        """Langfuse模式下关闭客户端"""
        if self.current_mode == MonitoringMode.LANGFUSE and self.is_langfuse_initialized and self.langfuse_manager:
            try:
                self.langfuse_manager.shutdown()
            except Exception as e:
                # 静默处理关闭失败
                pass

    # 模式检查方法
    def is_debug_mode(self) -> bool:
        """检查是否为Debug模式"""
        return self.current_mode == MonitoringMode.DEBUG

    def is_langfuse_mode(self) -> bool:
        """检查是否为Langfuse模式"""
        return self.current_mode == MonitoringMode.LANGFUSE


# 全局监控管理器实例
_monitoring_manager = MonitoringManager()


def get_monitoring_manager() -> MonitoringManager:
    """获取全局监控管理器实例"""
    return _monitoring_manager

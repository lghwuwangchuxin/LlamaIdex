# main.py
from logger import Logger
from model_manager import ModelManager
from milvus_manager import MilvusManager
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from AdvancedLlamaDebugMonitor import AdvancedLlamaDebugMonitor


def main():
    """主函数"""
    # 创建各个组件
    logger = Logger()
    debug_monitor = AdvancedLlamaDebugMonitor(print_trace_on_end=True, enable_detail_log=True)

    model_manager = ModelManager(logger)
    milvus_manager = MilvusManager(logger)
    document_processor = DocumentProcessor(logger)

    # 创建RAG系统
    rag_system = RAGSystem(logger, model_manager, milvus_manager, document_processor, debug_monitor)

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
        # 打印调试摘要
        debug_monitor.print_summary()


if __name__ == "__main__":
    main()

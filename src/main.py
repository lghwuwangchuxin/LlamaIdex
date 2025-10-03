from logger import Logger
from model_manager import ModelManager
from milvus_manager import MilvusManager
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from AdvancedLlamaDebugMonitor import AdvancedLlamaDebugMonitor
from lang_fuse import initialize_langfuse, get_langfuse_client, get_langfuse_manager
from monitoring_mode import MonitoringMode, get_monitoring_manager
import argparse
from config_manager import get_config_manager  # 添加导入

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RAG System with Monitoring')
    parser.add_argument('--monitor-mode',
                        choices=['debug', 'langfuse'],
                        default='debug',
                        help='监控模式: debug, langfuse')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    monitor_mode = MonitoringMode(args.monitor_mode)

    # 初始化监控管理器
    monitoring_manager = get_monitoring_manager()

    # 根据模式初始化相应监控组件
    debug_monitor = None
    langfuse_client = None

    if monitor_mode == MonitoringMode.DEBUG:
        debug_monitor = AdvancedLlamaDebugMonitor(print_trace_on_end=True, enable_detail_log=True)
    elif monitor_mode == MonitoringMode.LANGFUSE:
        if initialize_langfuse():
            print("✅ Langfuse初始化成功")
            langfuse_client = get_langfuse_client()
        else:
            print("❌ Langfuse初始化失败，降级到Debug模式")
            monitor_mode = MonitoringMode.DEBUG
            debug_monitor = AdvancedLlamaDebugMonitor(print_trace_on_end=True, enable_detail_log=True)

    # 设置监控模式
    monitoring_manager.set_mode(monitor_mode, debug_monitor, langfuse_client)

    # 创建各个组件
    logger = Logger()
    model_manager = ModelManager(logger)
    milvus_manager = MilvusManager(logger)
    document_processor = DocumentProcessor(logger)

    # 创建RAG系统
    rag_system = RAGSystem(logger, model_manager, milvus_manager, document_processor, monitoring_manager)

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

        # 根据模式执行相应的清理操作
        if monitor_mode == MonitoringMode.DEBUG and debug_monitor:
            debug_monitor.print_summary()
        elif monitor_mode == MonitoringMode.LANGFUSE and langfuse_client:
            langfuse_manager = get_langfuse_manager()
            if langfuse_manager:
                langfuse_manager.shutdown()


if __name__ == "__main__":
    main()

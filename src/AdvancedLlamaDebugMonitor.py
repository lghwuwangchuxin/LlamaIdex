# AdvancedLlamaDebugMonitor.py
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEvent, CBEventType
from llama_index.core import Settings
from typing import List, Dict, Any, Optional
import time
import json


class AdvancedLlamaDebugMonitor:
    """
    高级 LlamaIndex 调试监控类
    封装了 LlamaDebugHandler 并提供特定事件的监控方法
    """

    def __init__(self, print_trace_on_end: bool = True, enable_detail_log: bool = False):
        """
        初始化调试监控器

        Args:
            print_trace_on_end: 是否在结束时打印跟踪信息
            enable_detail_log: 是否启用详细日志记录
        """
        self.llama_debug = LlamaDebugHandler(print_trace_on_end=print_trace_on_end)
        self.callback_manager = CallbackManager([self.llama_debug])
        self.enable_detail_log = enable_detail_log
        self.query_start_time = None

        # 设置全局回调管理器
        Settings.callback_manager = self.callback_manager

        # 事件统计
        self.event_stats = {
            'text_split': 0,
            'node_parse': 0,
            'text_embedding': 0,
            'llm_call': 0,
            'query_engine': 0,
            'semantic_retrieval': 0,
            'prompt_synthesis': 0,
            'text_summarization': 0,
            'sub_question': 0,
            'function_call': 0,
            'reranking': 0,
            'exception': 0,
            'agent_step': 0
        }

        #基础事件类型
        #CHUNKING: 文本分块事件 - 记录文档分割前后的文本块信息
        #NODE_PARSING: 节点解析事件 - 记录文档解析为节点的过程
        #EMBEDDING: 文本嵌入事件 - 记录文本向量化嵌入的处理过程
        #LLM: 大语言模型调用事件 - 记录LLM的提示词模板和响应调用
        #QUERY: 查询事件 - 跟踪每个查询的开始和结束过程
        #RETRIEVE: 检索事件 - 记录查询时节点检索的相关信息
        #SYNTHESIZE: 合成事件 - 记录结果合成调用的日志
        #SUB_QUESTION: 子问题事件 - 记录生成的子问题及答案
        #高级事件类型
        #TEMPLATING: 模板事件 - 记录提示词模板的组装和处理
        #FUNCTION_CALL: 函数调用事件 - 记录LLM函数调用的过程
        #RERANKING: 重排序事件 - 记录检索结果重排序的处理
        #EXCEPTION: 异常事件 - 记录处理过程中发生的异常
        #AGENT_STEP: 智能体步骤事件 - 记录Agent执行的各个步骤
        #叶子事件类型
        #LEAF_EVENTS: 叶子事件集合 - 包含
        #CHUNKING、LLM、EMBEDDING，这些事件不会产生子事件
        self.event_type_mapping = {
            'text_split': CBEventType.CHUNKING,
            'node_parse': CBEventType.NODE_PARSING,
            'text_embedding': CBEventType.EMBEDDING,
            'llm_call': CBEventType.LLM,
            'query_engine': CBEventType.QUERY,
            'semantic_retrieval': CBEventType.RETRIEVE,
            'prompt_synthesis': CBEventType.TEMPLATING,
            'text_summarization': CBEventType.SYNTHESIZE,
            'sub_question': CBEventType.SUB_QUESTION,
            'function_call': CBEventType.FUNCTION_CALL,
            'reranking': CBEventType.RERANKING,
            'exception': CBEventType.EXCEPTION,
            'agent_step': CBEventType.AGENT_STEP
        }

    def _create_cb_event(self, event_name: str, payload: Dict[str, Any] = None) -> CBEvent:
        """
        创建 CBEvent 实例

        Args:
            event_name: 事件名称
            payload: 事件负载数据

        Returns:
            CBEvent: 创建的事件对象
        """
        event_type = self.event_type_mapping.get(event_name, CBEventType.LLM)
        return CBEvent(event_type=event_type, payload=payload or {})

    def _record_cb_event(self, event_name: str, payload: Dict[str, Any] = None):
        """
        记录 CBEvent 到回调管理器

        Args:
            event_name: 事件名称
            payload: 事件负载数据
        """
        event = self._create_cb_event(event_name, payload)
        # 在实际使用中，这里会将事件发送到回调管理器进行处理
        if self.enable_detail_log:
            print(f"记录 CBEvent: {event.event_type.value}, 负载: {payload}")

    def log_event(self, event_name: str, details: Dict[str, Any] = None):
        """记录事件日志"""
        if self.enable_detail_log:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {event_name}"
            if details:
                log_entry += f" - {json.dumps(details, indent=2, ensure_ascii=False)}"
            print(log_entry)

    # 文本分割事件监控
    def monitor_text_split_event(self, text_chunks: List[str] = None):
        """
        监控文本分割事件

        Args:
            text_chunks: 分割后的文本块列表
        """
        self.event_stats['text_split'] += 1
        details = {
            'chunk_count': len(text_chunks) if text_chunks else 0,
            'sample_chunk': text_chunks[0][:100] + "..." if text_chunks and len(text_chunks) > 0 else None
        }
        self.log_event("TEXT_SPLIT_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "chunk_count": len(text_chunks) if text_chunks else 0,
            "event_type": "text_split"
        }
        self._record_cb_event('text_split', payload)

        # 获取相关事件信息
        events = self.llama_debug.get_events()
        text_split_events = [e for e in events if hasattr(e, 'event_type') and e.event_type == CBEventType.CHUNKING]

        return {
            'event_type': 'text_split',
            'chunk_count': len(text_chunks) if text_chunks else 0,
            'related_events': [e.event_type.value for e in text_split_events],
            'timestamp': time.time()
        }

    # Node 解析事件监控
    def monitor_node_parse_event(self, nodes: List[Any] = None):
        """
        监控 Node 解析事件

        Args:
            nodes: 解析后的节点列表
        """
        self.event_stats['node_parse'] += 1
        details = {
            'node_count': len(nodes) if nodes else 0,
            'node_types': list(set([type(node).__name__ for node in nodes])) if nodes else []
        }
        self.log_event("NODE_PARSE_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "node_count": len(nodes) if nodes else 0,
            "event_type": "node_parse"
        }
        self._record_cb_event('node_parse', payload)

        events = self.llama_debug.get_events()
        node_events = [e for e in events if hasattr(e, 'event_type') and e.event_type == CBEventType.NODE_PARSING]
        return {
            'event_type': 'node_parse',
            'node_count': len(nodes) if nodes else 0,
            'parsing_events': [e.event_type.value for e in node_events],
            'duration': sum(getattr(e, 'duration', 0) for e in node_events)
        }

    # 文本嵌入事件监控
    def monitor_text_embedding_event(self, embeddings_info: Dict[str, Any] = None):
        """
        监控文本嵌入事件

        Args:
            embeddings_info: 嵌入相关信息
        """
        self.event_stats['text_embedding'] += 1
        details = {
            'embedding_dim': embeddings_info.get('dimension') if embeddings_info else None,
            'text_count': embeddings_info.get('text_count') if embeddings_info else 0
        }
        self.log_event("TEXT_EMBEDDING_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "embedding_dim": embeddings_info.get('dimension') if embeddings_info else None,
            "text_count": embeddings_info.get('text_count') if embeddings_info else 0,
            "event_type": "text_embedding"
        }
        self._record_cb_event('text_embedding', payload)

        events = self.llama_debug.get_events()
        embedding_events = [e for e in events if hasattr(e, 'event_type') and e.event_type == CBEventType.EMBEDDING]
        embedding_stats = []
        for event in embedding_events:
            embedding_stats.append({
                'event_name': event.event_type.value,
                'duration': getattr(event, 'duration', 0),
                'timestamp': getattr(event, 'time', '')
            })

        return {
            'event_type': 'text_embedding',
            'total_embedding_events': len(embedding_events),
            'embedding_stats': embedding_stats,
            'total_duration': sum(getattr(e, 'duration', 0) for e in embedding_events)
        }

    # 调用大模型事件监控
    def monitor_llm_call_event(self, query: str = None, response: str = None):
        """
        监控大模型调用事件

        Args:
            query: 查询文本
            response: 模型响应
        """
        self.event_stats['llm_call'] += 1
        details = {
            'query_length': len(query) if query else 0,
            'response_length': len(response) if response else 0,
            'query_preview': query[:100] + "..." if query and len(query) > 100 else query
        }
        self.log_event("LLM_CALL_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "query": query,
            "response": response,
            "query_length": len(query) if query else 0,
            "response_length": len(response) if response else 0,
            "event_type": "llm_call"
        }
        self._record_cb_event('llm_call', payload)

        # 获取 LLM 输入输出对
        event_pairs = self.llama_debug.get_llm_inputs_outputs()

        llm_events = []
        for i, (input_event, output_event) in enumerate(event_pairs):
            llm_events.append({
                'call_index': i,
                'prompt_template': input_event.payload.get("formatted_prompt", "")[
                                       :200] + "..." if input_event.payload.get("formatted_prompt") else None,
                'response_preview': output_event.payload.get("response", "")[:200] + "..." if output_event.payload.get(
                    "response") else None,
                'duration': getattr(output_event, 'duration', 0)
            })

        return {
            'event_type': 'llm_call',
            'total_calls': len(event_pairs),
            'llm_events': llm_events,
            'latest_query': query,
            'latest_response': response
        }

    # Query 引擎调用事件监控
    def monitor_query_engine_event(self, query: str, response: Any = None):
        """
        监控 Query 引擎调用事件

        Args:
            query: 查询文本
            response: 查询响应
        """
        self.event_stats['query_engine'] += 1
        self.query_start_time = time.time()

        details = {
            'query': query,
            'query_length': len(query)
        }
        self.log_event("QUERY_ENGINE_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "query": query,
            "query_length": len(query),
            "event_type": "query_engine"
        }
        self._record_cb_event('query_engine', payload)

        return {
            'event_type': 'query_engine_start',
            'query': query,
            'start_time': self.query_start_time
        }

    def complete_query_engine_event(self, response: Any, query: str = None):
        """
        完成 Query 引擎事件监控

        Args:
            response: 查询响应
            query: 查询文本（可选）
        """
        end_time = time.time()
        duration = end_time - self.query_start_time if self.query_start_time else 0

        details = {
            'response_length': len(str(response)) if response else 0,
            'duration': duration,
            'response_preview': str(response)[:200] + "..." if response else None
        }
        self.log_event("QUERY_ENGINE_COMPLETE", details)

        # 创建并记录 CBEvent
        payload = {
            "response": str(response) if response else None,
            "duration": duration,
            "event_type": "query_engine_complete"
        }
        self._record_cb_event('query_engine', payload)

        return {
            'event_type': 'query_engine_complete',
            'duration': duration,
            'response_preview': str(response)[:200] + "..." if response else None,
            'end_time': end_time
        }

    # 语义检索事件监控
    def monitor_semantic_retrieval_event(self, retrieved_nodes: List[Any] = None, query: str = None):
        """
        监控语义检索事件

        Args:
            retrieved_nodes: 检索到的节点列表
            query: 查询文本
        """
        self.event_stats['semantic_retrieval'] += 1

        details = {
            'retrieved_count': len(retrieved_nodes) if retrieved_nodes else 0,
            'query': query,
            'node_sources': list(set([getattr(node, 'node_id', '') for node in retrieved_nodes])) if retrieved_nodes else []
        }
        self.log_event("SEMANTIC_RETRIEVAL_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "retrieved_count": len(retrieved_nodes) if retrieved_nodes else 0,
            "query": query,
            "event_type": "semantic_retrieval"
        }
        self._record_cb_event('semantic_retrieval', payload)

        events = self.llama_debug.get_events()
        retrieval_events = [e for e in events if hasattr(e, 'event_type') and e.event_type == CBEventType.RETRIEVE]

        return {
            'event_type': 'semantic_retrieval',
            'retrieved_nodes_count': len(retrieved_nodes) if retrieved_nodes else 0,
            'retrieval_events': [e.event_type.value for e in retrieval_events],
            'retrieval_duration': sum(getattr(e, 'duration', 0) for e in retrieval_events)
        }

    # Prompt 组装和生成事件监控
    def monitor_prompt_synthesis_event(self, prompt_template: str = None, final_prompt: str = None):
        """
        监控 Prompt 组装和生成事件

        Args:
            prompt_template: 提示词模板
            final_prompt: 最终生成的提示词
        """
        self.event_stats['prompt_synthesis'] += 1

        details = {
            'template_preview': prompt_template[:100] + "..." if prompt_template else None,
            'final_prompt_preview': final_prompt[:100] + "..." if final_prompt else None,
            'prompt_length': len(final_prompt) if final_prompt else 0
        }
        self.log_event("PROMPT_SYNTHESIS_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "template_preview": prompt_template[:100] + "..." if prompt_template else None,
            "final_prompt_preview": final_prompt[:100] + "..." if final_prompt else None,
            "prompt_length": len(final_prompt) if final_prompt else 0,
            "event_type": "prompt_synthesis"
        }
        self._record_cb_event('prompt_synthesis', payload)

        events = self.llama_debug.get_events()
        synthesis_events = [e for e in events if hasattr(e, 'event_type') and e.event_type == CBEventType.TEMPLATING]

        return {
            'event_type': 'prompt_synthesis',
            'synthesis_events': [e.event_type.value for e in synthesis_events],
            'prompt_length': len(final_prompt) if final_prompt else 0,
            'synthesis_duration': sum(getattr(e, 'duration', 0) for e in synthesis_events)
        }

    # 文本摘要事件监控
    def monitor_text_summarization_event(self, original_text: str = None, summary: str = None):
        """
        监控文本摘要生成事件

        Args:
            original_text: 原始文本
            summary: 生成的摘要
        """
        self.event_stats['text_summarization'] += 1

        details = {
            'original_length': len(original_text) if original_text else 0,
            'summary_length': len(summary) if summary else 0,
            'compression_ratio': len(summary) / len(original_text) if original_text and summary and len(
                original_text) > 0 else 0,
            'summary_preview': summary[:100] + "..." if summary else None
        }
        self.log_event("TEXT_SUMMARIZATION_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "original_length": len(original_text) if original_text else 0,
            "summary_length": len(summary) if summary else 0,
            "compression_ratio": len(summary) / len(original_text) if original_text and summary and len(
                original_text) > 0 else 0,
            "summary_preview": summary[:100] + "..." if summary else None,
            "event_type": "text_summarization"
        }
        self._record_cb_event('text_summarization', payload)

        return {
            'event_type': 'text_summarization',
            'original_length': len(original_text) if original_text else 0,
            'summary_length': len(summary) if summary else 0,
            'compression_ratio': len(summary) / len(original_text) if original_text and summary and len(
                original_text) > 0 else 0
        }

    # 子问题生成事件监控
    def monitor_sub_question_event(self, main_question: str = None, sub_questions: List[str] = None):
        """
        监控子问题生成事件

        Args:
            main_question: 主问题
            sub_questions: 生成的子问题列表
        """
        self.event_stats['sub_question'] += 1

        details = {
            'main_question': main_question,
            'sub_question_count': len(sub_questions) if sub_questions else 0,
            'sub_questions': sub_questions
        }
        self.log_event("SUB_QUESTION_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "main_question": main_question,
            "sub_question_count": len(sub_questions) if sub_questions else 0,
            "sub_questions": sub_questions,
            "event_type": "sub_question"
        }
        self._record_cb_event('sub_question', payload)

        return {
            'event_type': 'sub_question_generation',
            'main_question': main_question,
            'sub_question_count': len(sub_questions) if sub_questions else 0,
            'sub_questions': sub_questions
        }

    # 函数调用事件监控
    def monitor_function_call_event(self, function_name: str, arguments: Dict[str, Any], response: Any = None):
        """
        监控函数调用事件

        Args:
            function_name: 函数名称
            arguments: 函数参数
            response: 函数响应
        """
        self.event_stats['function_call'] += 1
        details = {
            'function_name': function_name,
            'arguments': arguments,
            'response': str(response)[:100] + "..." if response else None
        }
        self.log_event("FUNCTION_CALL_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "function_name": function_name,
            "arguments": arguments,
            "response": response,
            "event_type": "function_call"
        }
        self._record_cb_event('function_call', payload)

        return {
            'event_type': 'function_call',
            'function_name': function_name,
            'arguments': arguments
        }

    # 重排序事件监控
    def monitor_reranking_event(self, query: str, nodes: List[Any], top_k: int):
        """
        监控重排序事件

        Args:
            query: 查询文本
            nodes: 节点列表
            top_k: 返回的top k结果
        """
        self.event_stats['reranking'] += 1
        details = {
            'query': query,
            'nodes_count': len(nodes),
            'top_k': top_k
        }
        self.log_event("RERANKING_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "query": query,
            "nodes_count": len(nodes),
            "top_k": top_k,
            "event_type": "reranking"
        }
        self._record_cb_event('reranking', payload)

        return {
            'event_type': 'reranking',
            'query': query,
            'nodes_count': len(nodes),
            'top_k': top_k
        }

    # 异常事件监控
    def monitor_exception_event(self, exception: Exception, context: str = ""):
        """
        监控异常事件

        Args:
            exception: 异常对象
            context: 异常上下文信息
        """
        self.event_stats['exception'] += 1
        details = {
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'context': context
        }
        self.log_event("EXCEPTION_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "context": context,
            "event_type": "exception"
        }
        self._record_cb_event('exception', payload)

        return {
            'event_type': 'exception',
            'exception_type': type(exception).__name__,
            'exception_message': str(exception)
        }

    # Agent步骤事件监控
    def monitor_agent_step_event(self, step_name: str, input_data: Any, output_data: Any):
        """
        监控Agent步骤事件

        Args:
            step_name: 步骤名称
            input_data: 输入数据
            output_data: 输出数据
        """
        self.event_stats['agent_step'] += 1
        details = {
            'step_name': step_name,
            'input_preview': str(input_data)[:100] + "..." if input_data else None,
            'output_preview': str(output_data)[:100] + "..." if output_data else None
        }
        self.log_event("AGENT_STEP_EVENT", details)

        # 创建并记录 CBEvent
        payload = {
            "step_name": step_name,
            "input_data": input_data,
            "output_data": output_data,
            "event_type": "agent_step"
        }
        self._record_cb_event('agent_step', payload)

        return {
            'event_type': 'agent_step',
            'step_name': step_name,
            'input_preview': str(input_data)[:100] + "..." if input_data else None
        }

    # 获取完整的事件统计报告
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """生成完整的调试报告"""
        events = self.llama_debug.get_events()
        total_duration = sum(getattr(event, 'duration', 0) for event in events)

        # 按事件类型分组
        event_groups = {}
        for event in events:
            event_type = getattr(event, 'event_type', CBEventType.LLM).value
            if event_type not in event_groups:
                event_groups[event_type] = []
            event_groups[event_type].append(event)

        report = {
            'summary': {
                'total_events': len(events),
                'total_duration': total_duration,
                'event_types_count': len(event_groups),
                'custom_event_stats': self.event_stats
            },
            'event_breakdown': {
                event_type: {
                    'count': len(events_list),
                    'total_duration': sum(getattr(e, 'duration', 0) for e in events_list),
                    'avg_duration': sum(getattr(e, 'duration', 0) for e in events_list) / len(events_list) if events_list else 0
                }
                for event_type, events_list in event_groups.items()
            },
            'performance_analysis': {
                'slowest_events': sorted(events, key=lambda x: getattr(x, 'duration', 0), reverse=True)[:5],
                'most_frequent_events': sorted(event_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]
            }
        }

        return report

    # 打印简要统计信息
    def print_summary(self):
        """打印调试摘要信息"""
        report = self.get_comprehensive_report()

        print("=" * 60)
        print("LLAMAINDEX 调试监控摘要")
        print("=" * 60)
        print(f"总事件数: {report['summary']['total_events']}")
        print(f"总耗时: {report['summary']['total_duration']:.2f}秒")
        print(f"事件类型数: {report['summary']['event_types_count']}")
        print("\n自定义事件统计:")
        for event_type, count in report['summary']['custom_event_stats'].items():
            print(f"  {event_type}: {count}次")

        print("\n性能分析 - 最耗时的5个事件:")
        for i, event in enumerate(report['performance_analysis']['slowest_events'][:5]):
            event_type = getattr(event, 'event_type', CBEventType.LLM).value
            duration = getattr(event, 'duration', 0)
            print(f"  {i + 1}. {event_type}: {duration:.2f}秒")

        print("=" * 60)

from typing import List, Optional, Any, Union
from llama_index.core.schema import Document, BaseNode, TransformComponent
import hashlib
from datetime import datetime


class DocumentCleaner(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        cleaned_nodes = []

        for node in nodes:
            if not hasattr(node, 'text') or node.text is None:
                continue  # 或记录警告

            cleaned_text = node.text

            if cleaned_text:
                cleaned_text = ' '.join(cleaned_text.split())
                lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
                cleaned_text = '\n'.join(lines)

            # 创建适当类型的新节点
            if isinstance(node, Document):
                cleaned_node = Document(
                    text=cleaned_text.strip(),
                    **{k: v for k, v in node.to_dict().items()
                       if k not in ['text', 'metadata']}
                )
            else:
                # 对于其他BaseNode子类
                node.text = cleaned_text.strip()
                cleaned_node = node

            cleaned_nodes.append(cleaned_node)

        return cleaned_nodes


class MetadataEnricher(TransformComponent):
    def _compute_hash(self, content: str) -> str:
        # 考虑使用更安全的哈希算法
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        for node in nodes:
            if not hasattr(node, 'text') or node.text is None:
                continue

            metadata = node.metadata.copy() if node.metadata else {}

            if self.add_hash:
                metadata['content_hash'] = self._compute_hash(node.text)

            if self.add_timestamp:
                metadata['processed_at'] = datetime.now().isoformat()

            metadata.update(self.custom_metadata)
            node.metadata = metadata

        return nodes


class ContentFilter(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        filtered_nodes = []

        for node in nodes:
            filtered_nodes.append(node)

        return filtered_nodes


class DocumentDeduplicator(TransformComponent):
    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        unique_nodes = []
        seen_hashes = set()

        for node in nodes:
            if not hasattr(node, 'text') or node.text is None:
                unique_nodes.append(node)  # 没有文本的节点保留
                continue

            if self.use_content_hash:
                content_hash = self._compute_hash(node.text)
            else:
                metadata_str = str(sorted((node.metadata or {}).items()))
                content_hash = self._compute_hash(metadata_str)

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_nodes.append(node)

        return unique_nodes
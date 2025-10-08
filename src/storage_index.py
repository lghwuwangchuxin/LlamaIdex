from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import json
import pickle


class BaseIndex(ABC):
    """
    索引存储的基类，定义所有索引类型的基本接口
    """

    def __init__(self, index_name: str, config: Optional[Dict] = None):
        """
        初始化索引

        Args:
            index_name: 索引名称
            config: 配置参数
        """
        self.index_name = index_name
        self.config = config or {}
        self.is_built = False

    @abstractmethod
    def build(self, data: Any) -> None:
        """
        构建索引

        Args:
            data: 用于构建索引的数据
        """
        pass

    @abstractmethod
    def query(self, query_data: Any, **kwargs) -> Any:
        """
        查询索引

        Args:
            query_data: 查询数据
            **kwargs: 其他查询参数

        Returns:
            查询结果
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存索引到文件

        Args:
            path: 保存路径
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        从文件加载索引

        Args:
            path: 加载路径
        """
        pass


class VectorStoreIndex(BaseIndex):
    """
    向量存储索引
    用于存储和检索向量嵌入表示
    """

    def __init__(self, index_name: str, config: Optional[Dict] = None):
        super().__init__(index_name, config)
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.dimension = config.get('dimension') if config else None

    def build(self, data: List[Dict[str, Any]]) -> None:
        """
        构建向量索引

        Args:
            data: 包含向量和元数据的列表
        """
        self.vectors = [item['vector'] for item in data]
        self.metadata = [item.get('metadata', {}) for item in data]
        if self.vectors and not self.dimension:
            self.dimension = len(self.vectors[0])
        self.is_built = True

    def query(self, query_vector: np.ndarray, top_k: int = 5,
              similarity_metric: str = 'cosine') -> List[Dict]:
        """
        向量相似性查询

        Args:
            query_vector: 查询向量
            top_k: 返回最相似的k个结果
            similarity_metric: 相似度计算方法 ('cosine', 'euclidean')

        Returns:
            相似结果列表
        """
        if not self.is_built:
            raise RuntimeError("索引尚未构建")

        similarities = []
        for i, vector in enumerate(self.vectors):
            if similarity_metric == 'cosine':
                # 余弦相似度
                sim = np.dot(query_vector, vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(vector)
                )
            elif similarity_metric == 'euclidean':
                # 欧氏距离(转换为相似度)
                sim = 1 / (1 + np.linalg.norm(query_vector - vector))
            else:
                raise ValueError(f"不支持的相似度计算方法: {similarity_metric}")

            similarities.append((sim, i))

        # 按相似度排序并返回top_k结果
        similarities.sort(reverse=True)
        results = []
        for sim, idx in similarities[:top_k]:
            results.append({
                'similarity': sim,
                'metadata': self.metadata[idx],
                'vector': self.vectors[idx]
            })

        return results

    def save(self, path: str) -> None:
        """保存向量索引到文件"""
        data = {
            'vectors': [vec.tolist() for vec in self.vectors],  # 转换为列表以便序列化
            'metadata': self.metadata,
            'dimension': self.dimension,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """从文件加载向量索引"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectors = [np.array(vec) for vec in data['vectors']]
        self.metadata = data['metadata']
        self.dimension = data['dimension']
        self.config = data['config']
        self.is_built = True


class SummaryIndex(BaseIndex):
    """
    文档摘要索引
    用于存储和检索文档摘要信息
    """

    def __init__(self, index_name: str, config: Optional[Dict] = None):
        super().__init__(index_name, config)
        self.summaries: Dict[str, str] = {}
        self.documents: Dict[str, str] = {}

    def build(self, data: Dict[str, Dict[str, str]]) -> None:
        """
        构建摘要索引

        Args:
            data: 包含文档ID、原文和摘要的字典
        """
        for doc_id, content in data.items():
            self.documents[doc_id] = content.get('document', '')
            self.summaries[doc_id] = content.get('summary', '')
        self.is_built = True

    def query(self, query_text: str, use_summary: bool = True) -> List[Dict]:
        """
        基于关键字的摘要查询

        Args:
            query_text: 查询文本
            use_summary: 是否使用摘要进行查询

        Returns:
            匹配的文档列表
        """
        if not self.is_built:
            raise RuntimeError("索引尚未构建")

        query_words = set(query_text.lower().split())
        results = []

        target_dict = self.summaries if use_summary else self.documents
        for doc_id, text in target_dict.items():
            text_words = set(text.lower().split())
            # 计算交集大小作为匹配度
            match_score = len(query_words.intersection(text_words))
            if match_score > 0:
                results.append({
                    'doc_id': doc_id,
                    'score': match_score,
                    'summary': self.summaries[doc_id],
                    'document': self.documents[doc_id]
                })

        # 按匹配度排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def save(self, path: str) -> None:
        """保存摘要索引到文件"""
        data = {
            'summaries': self.summaries,
            'documents': self.documents,
            'config': self.config
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从文件加载摘要索引"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.summaries = data['summaries']
        self.documents = data['documents']
        self.config = data['config']
        self.is_built = True


class ObjectIndex(BaseIndex):
    """
    对象索引
    用于存储和检索结构化对象
    """

    def __init__(self, index_name: str, config: Optional[Dict] = None):
        super().__init__(index_name, config)
        self.objects: List[Dict] = []
        self.indexed_fields: List[str] = config.get('indexed_fields', []) if config else []

    def build(self, data: List[Dict]) -> None:
        """
        构建对象索引

        Args:
            data: 对象列表
        """
        self.objects = data
        self.is_built = True

    def query(self, filters: Dict[str, Any]) -> List[Dict]:
        """
        对象查询

        Args:
            filters: 查询条件字典

        Returns:
            匹配的对象列表
        """
        if not self.is_built:
            raise RuntimeError("索引尚未构建")

        results = []
        for obj in self.objects:
            match = True
            for field, value in filters.items():
                if field not in obj or obj[field] != value:
                    match = False
                    break
            if match:
                results.append(obj)

        return results

    def save(self, path: str) -> None:
        """保存对象索引到文件"""
        data = {
            'objects': self.objects,
            'indexed_fields': self.indexed_fields,
            'config': self.config
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从文件加载对象索引"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.objects = data['objects']
        self.indexed_fields = data['indexed_fields']
        self.config = data['config']
        self.is_built = True


class KnowledgeGraphIndex(BaseIndex):
    """
    知识图谱索引
    用于存储和检索实体关系图
    """

    def __init__(self, index_name: str, config: Optional[Dict] = None):
        super().__init__(index_name, config)
        self.entities: Dict[str, Dict] = {}
        self.relations: List[Dict] = []

    def build(self, data: Dict[str, Any]) -> None:
        """
        构建知识图谱索引

        Args:
            data: 包含实体和关系的数据
        """
        self.entities = data.get('entities', {})
        self.relations = data.get('relations', [])
        self.is_built = True

    def query(self, entity: str = None, relation: str = None,
              target_entity: str = None) -> List[Dict]:
        """
        知识图谱查询

        Args:
            entity: 起始实体
            relation: 关系类型
            target_entity: 目标实体

        Returns:
            匹配的关系列表
        """
        if not self.is_built:
            raise RuntimeError("索引尚未构建")

        results = []
        for rel in self.relations:
            match = True
            if entity and rel.get('source') != entity:
                match = False
            if relation and rel.get('relation') != relation:
                match = False
            if target_entity and rel.get('target') != target_entity:
                match = False

            if match:
                results.append(rel)

        return results

    def get_entity_info(self, entity_id: str) -> Optional[Dict]:
        """
        获取实体详细信息

        Args:
            entity_id: 实体ID

        Returns:
            实体信息
        """
        return self.entities.get(entity_id)

    def save(self, path: str) -> None:
        """保存知识图谱索引到文件"""
        data = {
            'entities': self.entities,
            'relations': self.relations,
            'config': self.config
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从文件加载知识图谱索引"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.entities = data['entities']
        self.relations = data['relations']
        self.config = data['config']
        self.is_built = True


class TreeIndex(BaseIndex):
    """
    树索引
    用于存储和检索层次化树形结构数据
    """

    def __init__(self, index_name: str, config: Optional[Dict] = None):
        super().__init__(index_name, config)
        self.tree_data: Dict = {}
        self.node_index: Dict[str, Dict] = {}

    def build(self, data: Dict) -> None:
        """
        构建树索引

        Args:
            data: 树形结构数据
        """
        self.tree_data = data
        self._build_node_index(data)
        self.is_built = True

    def _build_node_index(self, node: Dict, path: str = "") -> None:
        """
        递归构建节点索引

        Args:
            node: 当前节点
            path: 节点路径
        """
        node_id = node.get('id')
        if node_id:
            current_path = f"{path}/{node_id}" if path else node_id
            self.node_index[node_id] = {
                'data': node,
                'path': current_path
            }

            # 递归处理子节点
            for child in node.get('children', []):
                self._build_node_index(child, current_path)

    def query(self, node_id: str = None, path: str = None) -> Union[Dict, List[Dict]]:
        """
        树节点查询

        Args:
            node_id: 节点ID
            path: 节点路径

        Returns:
            匹配的节点或节点列表
        """
        if not self.is_built:
            raise RuntimeError("索引尚未构建")

        if node_id:
            return self.node_index.get(node_id)
        elif path:
            results = []
            for node_info in self.node_index.values():
                if node_info['path'].startswith(path):
                    results.append(node_info)
            return results
        else:
            return list(self.node_index.values())

    def get_children(self, node_id: str) -> List[Dict]:
        """
        获取节点的子节点

        Args:
            node_id: 节点ID

        Returns:
            子节点列表
        """
        node_info = self.node_index.get(node_id)
        if not node_info:
            return []
        return node_info['data'].get('children', [])

    def save(self, path: str) -> None:
        """保存树索引到文件"""
        data = {
            'tree_data': self.tree_data,
            'config': self.config
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从文件加载树索引"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.tree_data = data['tree_data']
        self.config = data['config']
        self.node_index = {}
        self._build_node_index(self.tree_data)
        self.is_built = True


class KeywordTableIndex(BaseIndex):
    """
    关键词表索引
    用于基于关键词的快速检索
    """

    def __init__(self, index_name: str, config: Optional[Dict] = None):
        super().__init__(index_name, config)
        self.keyword_index: Dict[str, List[str]] = {}  # keyword -> doc_ids
        self.doc_keywords: Dict[str, List[str]] = {}  # doc_id -> keywords
        self.doc_contents: Dict[str, str] = {}  # doc_id -> content

    def build(self, data: Dict[str, Dict]) -> None:
        """
        构建关键词索引

        Args:
            data: 包含文档ID和关键词列表的字典
        """
        for doc_id, doc_info in data.items():
            keywords = doc_info.get('keywords', [])
            content = doc_info.get('content', '')

            self.doc_keywords[doc_id] = keywords
            self.doc_contents[doc_id] = content

            # 构建倒排索引
            for keyword in keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(doc_id)

        self.is_built = True

    def query(self, keywords: List[str], operator: str = 'AND') -> List[Dict]:
        """
        关键词查询

        Args:
            keywords: 查询关键词列表
            operator: 操作符 ('AND' 或 'OR')

        Returns:
            匹配的文档列表
        """
        if not self.is_built:
            raise RuntimeError("索引尚未构建")

        if not keywords:
            return []

        result_doc_ids = set()
        if operator.upper() == 'AND':
            # 所有关键词都必须匹配
            first_keyword = keywords[0]
            result_doc_ids = set(self.keyword_index.get(first_keyword, []))
            for keyword in keywords[1:]:
                keyword_docs = set(self.keyword_index.get(keyword, []))
                result_doc_ids = result_doc_ids.intersection(keyword_docs)
        elif operator.upper() == 'OR':
            # 任一关键词匹配即可
            for keyword in keywords:
                keyword_docs = set(self.keyword_index.get(keyword, []))
                result_doc_ids = result_doc_ids.union(keyword_docs)

        # 构建结果
        results = []
        for doc_id in result_doc_ids:
            results.append({
                'doc_id': doc_id,
                'content': self.doc_contents.get(doc_id, ''),
                'keywords': self.doc_keywords.get(doc_id, [])
            })

        return results

    def get_keyword_documents(self, keyword: str) -> List[str]:
        """
        获取包含特定关键词的文档列表

        Args:
            keyword: 关键词

        Returns:
            文档ID列表
        """
        return self.keyword_index.get(keyword, [])

    def save(self, path: str) -> None:
        """保存关键词索引到文件"""
        data = {
            'keyword_index': self.keyword_index,
            'doc_keywords': self.doc_keywords,
            'doc_contents': self.doc_contents,
            'config': self.config
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从文件加载关键词索引"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.keyword_index = data['keyword_index']
        self.doc_keywords = data['doc_keywords']
        self.doc_contents = data['doc_contents']
        self.config = data['config']
        self.is_built = True


# 索引工厂类，用于创建不同类型的索引
class IndexFactory:
    """
    索引工厂类，用于创建不同类型的索引实例
    """

    _index_types = {
        'vector': VectorStoreIndex,
        'summary': SummaryIndex,
        'object': ObjectIndex,
        'knowledge_graph': KnowledgeGraphIndex,
        'tree': TreeIndex,
        'keyword_table': KeywordTableIndex
    }

    @classmethod
    def create_index(cls, index_type: str, index_name: str,
                     config: Optional[Dict] = None) -> BaseIndex:
        """
        创建指定类型的索引实例

        Args:
            index_type: 索引类型
            index_name: 索引名称
            config: 配置参数

        Returns:
            索引实例
        """
        if index_type not in cls._index_types:
            raise ValueError(f"不支持的索引类型: {index_type}")

        index_class = cls._index_types[index_type]
        return index_class(index_name, config)

    @classmethod
    def register_index_type(cls, index_type: str, index_class: type) -> None:
        """
        注册新的索引类型

        Args:
            index_type: 索引类型名称
            index_class: 索引类
        """
        if not issubclass(index_class, BaseIndex):
            raise ValueError("索引类必须继承自BaseIndex")
        cls._index_types[index_type] = index_class


# 多索引管理器
class MultiIndexManager:
    """
    多索引管理器，用于同时管理多种类型的索引
    """

    def __init__(self):
        self.indexes: Dict[str, BaseIndex] = {}

    def add_index(self, index_name: str, index: BaseIndex) -> None:
        """
        添加索引到管理器

        Args:
            index_name: 索引名称
            index: 索引实例
        """
        self.indexes[index_name] = index

    def remove_index(self, index_name: str) -> None:
        """
        从管理器中移除索引

        Args:
            index_name: 索引名称
        """
        if index_name in self.indexes:
            del self.indexes[index_name]

    def get_index(self, index_name: str) -> Optional[BaseIndex]:
        """
        获取指定名称的索引

        Args:
            index_name: 索引名称

        Returns:
            索引实例或None
        """
        return self.indexes.get(index_name)

    def query_all_indexes(self, query_data: Any, **kwargs) -> Dict[str, Any]:
        """
        在所有索引上执行查询

        Args:
            query_data: 查询数据
            **kwargs: 查询参数

        Returns:
            各索引的查询结果字典
        """
        results = {}
        for name, index in self.indexes.items():
            try:
                results[name] = index.query(query_data, **kwargs)
            except Exception as e:
                results[name] = {"error": str(e)}
        return results

    def save_all_indexes(self, base_path: str) -> None:
        """
        保存所有索引到文件

        Args:
            base_path: 基础保存路径
        """
        import os
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        for name, index in self.indexes.items():
            save_path = os.path.join(base_path, f"{name}_index.pkl")
            index.save(save_path)

    def load_all_indexes(self, base_path: str) -> None:
        """
        从文件加载所有索引

        Args:
            base_path: 基础加载路径
        """
        import os
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"路径不存在: {base_path}")

        for filename in os.listdir(base_path):
            if filename.endswith("_index.pkl"):
                index_name = filename.replace("_index.pkl", "")
                load_path = os.path.join(base_path, filename)

                # 这里需要知道索引类型才能正确加载
                # 实际应用中可能需要在保存时记录索引类型信息
                # 此处仅为示例
                try:
                    # 假设我们已经知道索引类型
                    index = IndexFactory.create_index('vector', index_name)  # 示例
                    index.load(load_path)
                    self.indexes[index_name] = index
                except Exception as e:
                    print(f"加载索引 {index_name} 失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 创建不同类型的索引示例
    vector_index = IndexFactory.create_index('vector', 'test_vector_index')
    summary_index = IndexFactory.create_index('summary', 'test_summary_index')
    object_index = IndexFactory.create_index('object', 'test_object_index')
    kg_index = IndexFactory.create_index('knowledge_graph', 'test_kg_index')
    tree_index = IndexFactory.create_index('tree', 'test_tree_index')
    keyword_index = IndexFactory.create_index('keyword_table', 'test_keyword_index')

    print("所有索引类型创建成功")

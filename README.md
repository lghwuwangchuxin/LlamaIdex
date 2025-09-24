
# LlamaIndex RAG 系统

基于 LlamaIndex 和 Milvus 构建的检索增强生成(RAG)问答系统，使用 Ollama 提供的本地大语言模型。

## 📋 项目概述

本项目实现了一个完整的 RAG 系统，能够：
- 从本地文档中提取和分割文本内容
- 使用嵌入模型将文本转换为向量表示
- 将向量存储在 Milvus 向量数据库中
- 通过语义搜索检索相关文档片段
- 使用大语言模型生成基于检索内容的回答

## 🛠 运行环境要求

### 系统要求
- Python 3.8 或更高版本
- macOS/Linux/Windows (推荐 macOS 或 Linux)
- 至少 8GB RAM (推荐 16GB 或更多)
- 至少 20GB 可用磁盘空间

### 依赖服务
1. **Ollama** - 本地大语言模型服务
2. **Milvus** - 向量数据库

### Python 依赖包
- llama-index-core>=0.10.0
- llama-index-llms-ollama>=0.1.0
- llama-index-embeddings-ollama>=0.1.0
- llama-index-vector-stores-milvus>=0.1.0
- pymilvus>=2.4.0
- chromadb (可选)

## 🚀 安装与配置

### 1. 安装 Ollama
brew install ollama
Linux
curl -fsSL https://ollama.com/install.sh | sh
Windows
访问 https://ollama.com/download/OllamaSetup.exe 下载安
### 2. 拉取所需模型
拉取语言模型
ollama pull llama3
拉取嵌入模型
ollama pull nomic-embed-text
### 3. 安装 Milvus
推荐使用 Docker 方式安装:
使用 Docker Compose 启动 Milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.4.9/milvus-standalone-docker-compose.yml -O docker-compose.yml
### 4. 安装 Python 依赖
pip install llama-index-core
pip install llama-index-llms-ollama
pip install llama-index-embeddings-ollama
pip install llama-index-vector-stores-milvus
pip install pymilvus
### 5. 准备数据文件
在项目目录下创建 `data` 文件夹，并放入要处理的文本文件:
## 📁 项目结构
LlamaIdex/
├── main.py # 程序入口文件
├── logger.py # 日志管理模块
├── model_manager.py # 模型管理模块
├── milvus_manager.py # Milvus 数据库管理模块
├── document_processor.py # 文档处理模块
├── rag_system.py # RAG 系统主控制模块
├── data/ # 文档数据目录
│ └── yiyan.txt # 示例文档文件
└── README.md # 项目说明文档
## ▶️ 运行程序

### 启动依赖服务
启动 Ollama 服务
ollama serve
启动 Milvus (如果使用 Docker)
docker-compose up -d
### 运行主程序
python main.py
## 🎮 使用方法

程序启动后会显示:
[时间] [INFO] 欢迎使用AI助手！输入 'exit' 退出程序。 问题
输入你的问题，系统会:
1. 在向量数据库中搜索相关内容
2. 使用大语言模型生成基于检索内容的回答
3. 显示回答和来源信息

示例:
问题：什么是人工智能？
AI助手：人工智能是计算机科学的一个分支，它企图了解智能的实质...
输入 `exit` 退出程序。

## ⚙️ 配置说明

### 模型配置
在 [model_manager.py](file:///Users/liuguanghu/PythonPorject/LlamaIdex/src/model_manager.py) 中可以修改:
- LLM 模型: `model="llama3"`
- 嵌入模型: `model_name="nomic-embed-text"`
- 模型服务地址: `base_url="http://127.0.0.1:11434"`

### 文档配置
在 [document_processor.py](file:///Users/liuguanghu/PythonPorject/LlamaIdex/src/document_processor.py) 中可以修改:
- 文档路径: `self.file_path`
- 分割参数: `chunk_size` 和 `chunk_overlap`

### Milvus 配置
在 [milvus_manager.py](file:///Users/liuguanghu/PythonPorject/LlamaIdex/src/milvus_manager.py) 中可以修改:
- 数据库地址: `self.host` 和 `self.port`
- 集合名称: `self.collection_name`
- 数据库名称: `self.db_name`

## 🔧 故障排除

### 常见问题

1. **Ollama 模型未找到**
解决方法: ollama pull llama3
2. **无法连接到 Milvus**
检查: docker-compose ps
启动: docker-compose up -d
3. **嵌入维度不匹配**
解决方法: 检查嵌入模型维度并在 Milvus 集合中设置相同维度
4. **文档文件不存在**
解决方法: 确保 data/yiyan.txt 文件存在或修改文件路径
### 日志查看
程序会输出详细的执行日志，帮助诊断问题:
- `[INFO]` - 正常信息
- `[WARNING]` - 警告信息
- `[ERROR]` - 错误信息

## 📈 系统架构
mermaid
graph TD
A[用户问题] --> B[RAG系统]
B --> C[文档处理器] 
C --> D[文档分割]
D --> E[Milvus向量存储] 
B --> F[Ollama模型] 
F --> G[LLM推理]
E --> H[向量检索]
H --> G
G --> I[生成回答]



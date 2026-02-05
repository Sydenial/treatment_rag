"""
RAG系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    case_report_data_path: str = "/home/syd/remote/rag/case_report_data_cleaned"
    guidelines_data_path = "/home/syd/remote/rag/English_guidelines_data"
    index_save_path: str = "/home/syd/remote/rag/case_report_vector_index"
    guidelines_index_save_path: str = "/home/syd/remote/rag/guidelines_vector_index"

    # 模型配置
    embedding_model: str = "/data/24T/Sunyuandong/Qwen3-Embedding-8B"
    llm_model: str = "deepseek-chat"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        """初始化后的处理"""
        pass
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()

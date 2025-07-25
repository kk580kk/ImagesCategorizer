"""
混合向量生成器
使用multimodal-embedding-v1处理图像，text-embedding-v4处理文本
不使用降级机制，API调用失败时直接报告
"""
import numpy as np
import logging
import json
import requests
from typing import List, Dict, Any, Optional, Union
import base64
import io
from PIL import Image
import dashscope
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridEmbeddingGenerator:
    """混合向量生成器类"""
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        初始化混合向量生成器
        
        Args:
            api_key: 阿里云API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.embedding_dim = 1024
        
        # 设置dashscope API密钥
        dashscope.api_key = api_key
        
        logger.info("初始化混合向量生成器 - multimodal-embedding-v1 + text-embedding-v4")
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        将图像文件编码为base64字符串
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: base64编码的图像数据
        """
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
                return base64_string
        except Exception as e:
            logger.error(f"图像编码失败: {e}")
            raise
    
    def generate_multimodal_embedding(self, image_path: str, text: str = "") -> np.ndarray:
        """
        使用multimodal-embedding-v1生成多模态向量
        
        Args:
            image_path: 图像文件路径
            text: 可选的文本描述
            
        Returns:
            np.ndarray: 1024维向量
        """
        try:
            # 编码图像
            image_base64 = self._encode_image_to_base64(image_path)
            
            # 构建输入内容（使用dashscope格式）
            content = [{"image": f"data:image/png;base64,{image_base64}"}]
            
            # 如果有文本，添加到输入中
            if text and text.strip():
                content.append({"text": text.strip()})
            
            logger.info(f"调用multimodal-embedding-v1 API，图像: {image_path}")
            
            # 使用dashscope.MultiModalEmbedding.call
            resp = dashscope.MultiModalEmbedding.call(
                model="multimodal-embedding-v1",
                input=content
            )
            
            # 处理响应
            if resp.get("status_code") == 200:
                embedding = resp['output']['embeddings'][0]['embedding']
                embedding_array = np.array(embedding, dtype=np.float32)
                logger.info(f"multimodal-embedding-v1成功生成向量，维度: {len(embedding_array)}")
                return embedding_array
            else:
                error_code = resp.get('code', 'Unknown')
                error_msg = resp.get('message', 'Unknown error')
                error_msg = f"multimodal-embedding-v1 API调用失败: {error_code} - {error_msg}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
        except Exception as e:
            error_msg = f"multimodal-embedding-v1调用失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        使用text-embedding-v4生成文本向量
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 1024维向量
        """
        try:
            if not text or not text.strip():
                logger.warning("输入文本为空")
                return np.zeros(self.embedding_dim, dtype=np.float32)
            
            payload = {
                "model": "text-embedding-v4",
                "input": text.strip()
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"调用text-embedding-v4 API，文本长度: {len(text)}")
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = f"text-embedding-v4 API调用失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            
            if 'data' not in result or not result['data']:
                error_msg = f"text-embedding-v4 API返回数据格式错误: {result}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # 提取向量
            embedding = result['data'][0]['embedding']
            embedding_array = np.array(embedding, dtype=np.float32)
            
            logger.info(f"text-embedding-v4成功生成向量，维度: {len(embedding_array)}")
            return embedding_array
            
        except Exception as e:
            error_msg = f"text-embedding-v4调用失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def generate_image_embedding(self, image_path: str, description: str = "") -> np.ndarray:
        """
        为图像生成向量（使用multimodal-embedding-v1）
        
        Args:
            image_path: 图像文件路径
            description: 可选的图像描述
            
        Returns:
            np.ndarray: 1024维向量
        """
        return self.generate_multimodal_embedding(image_path, description)
    
    def text_to_embedding(self, text: str) -> np.ndarray:
        """
        将文本转换为向量（使用text-embedding-v4）
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 1024维向量
        """
        return self.generate_text_embedding(text)
    
    def image_description_to_embedding(self, description: str) -> np.ndarray:
        """
        将图像描述转换为向量（使用text-embedding-v4）
        
        Args:
            description: 图像描述文本
            
        Returns:
            np.ndarray: 1024维向量
        """
        return self.generate_text_embedding(description)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个向量之间的余弦相似度
        
        Args:
            embedding1: 第一个向量
            embedding2: 第二个向量
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            # 确保输入是numpy数组
            emb1 = np.array(embedding1, dtype=np.float32)
            emb2 = np.array(embedding2, dtype=np.float32)
            
            # 归一化向量
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            emb1_normalized = emb1 / norm1
            emb2_normalized = emb2 / norm2
            
            # 计算余弦相似度
            similarity = np.dot(emb1_normalized, emb2_normalized)
            
            # 将相似度从[-1, 1]映射到[0, 1]
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def batch_text_to_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        批量将文本转换为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            List[np.ndarray]: 向量列表
        """
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.generate_text_embedding(text)
                embeddings.append(embedding)
                logger.info(f"批量处理进度: {i+1}/{len(texts)}")
            except Exception as e:
                logger.error(f"批量处理第{i+1}个文本失败: {e}")
                # 不使用降级，直接抛出异常
                raise Exception(f"批量文本向量化失败，第{i+1}个文本: {str(e)}")
        
        logger.info(f"批量文本向量化完成，处理了{len(texts)}个文本")
        return embeddings
    
    def batch_image_to_embeddings(self, image_paths: List[str], descriptions: List[str] = None) -> List[np.ndarray]:
        """
        批量将图像转换为向量
        
        Args:
            image_paths: 图像路径列表
            descriptions: 可选的描述列表
            
        Returns:
            List[np.ndarray]: 向量列表
        """
        embeddings = []
        descriptions = descriptions or [""] * len(image_paths)
        
        for i, (image_path, description) in enumerate(zip(image_paths, descriptions)):
            try:
                embedding = self.generate_multimodal_embedding(image_path, description)
                embeddings.append(embedding)
                logger.info(f"批量处理进度: {i+1}/{len(image_paths)}")
            except Exception as e:
                logger.error(f"批量处理第{i+1}个图像失败: {e}")
                # 不使用降级，直接抛出异常
                raise Exception(f"批量图像向量化失败，第{i+1}个图像: {str(e)}")
        
        logger.info(f"批量图像向量化完成，处理了{len(image_paths)}个图像")
        return embeddings
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """
        获取向量生成器信息
        
        Returns:
            Dict[str, Any]: 信息字典
        """
        return {
            "type": "HybridEmbeddingGenerator",
            "image_model": "multimodal-embedding-v1",
            "text_model": "text-embedding-v4",
            "embedding_dim": self.embedding_dim,
            "fallback_enabled": False,  # 明确标识不使用降级
            "api_base": self.base_url
        }


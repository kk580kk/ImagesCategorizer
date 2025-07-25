"""
简化的Embedding生成器
使用字符级别的向量化，更适合中文文本
"""
import numpy as np
import logging
import hashlib
import json
from collections import Counter
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEmbeddingGenerator:
    """简化的Embedding生成器类"""
    
    def __init__(self, embedding_dim=1024):
        """
        初始化embedding生成器
        
        Args:
            embedding_dim: embedding维度
        """
        self.embedding_dim = embedding_dim
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.is_fitted = False
        logger.info(f"初始化简化Embedding生成器，维度: {embedding_dim}")
    
    def build_vocabulary(self, texts):
        """
        构建字符词汇表
        
        Args:
            texts: 文本列表
        """
        try:
            # 收集所有字符
            all_chars = set()
            for text in texts:
                if text and isinstance(text, str):
                    # 清理文本，保留中文、英文、数字和基本标点
                    cleaned_text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？；：""''（）【】]', '', text)
                    all_chars.update(cleaned_text)
            
            # 构建字符到索引的映射
            self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
            self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
            self.vocab_size = len(self.char_to_idx)
            self.is_fitted = True
            
            logger.info(f"词汇表构建完成，包含{self.vocab_size}个字符")
            return True
            
        except Exception as e:
            logger.error(f"词汇表构建失败: {e}")
            return False
    
    def text_to_char_vector(self, text):
        """
        将文本转换为字符频率向量
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 字符频率向量
        """
        try:
            if not text or not isinstance(text, str):
                return np.zeros(self.embedding_dim)
            
            # 清理文本
            cleaned_text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？；：""''（）【】]', '', text)
            
            # 计算字符频率
            char_counts = Counter(cleaned_text)
            
            # 创建向量
            vector = np.zeros(min(self.vocab_size, self.embedding_dim))
            
            for char, count in char_counts.items():
                if char in self.char_to_idx:
                    idx = self.char_to_idx[char]
                    if idx < len(vector):
                        vector[idx] = count
            
            # 归一化
            if np.sum(vector) > 0:
                vector = vector / np.sum(vector)
            
            # 如果向量维度不足，用零填充
            if len(vector) < self.embedding_dim:
                padded_vector = np.zeros(self.embedding_dim)
                padded_vector[:len(vector)] = vector
                vector = padded_vector
            elif len(vector) > self.embedding_dim:
                vector = vector[:self.embedding_dim]
            
            return vector
            
        except Exception as e:
            logger.error(f"文本向量化失败: {e}")
            return np.zeros(self.embedding_dim)
    
    def text_to_embedding(self, text):
        """
        将文本转换为embedding向量
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: embedding向量
        """
        try:
            if not self.is_fitted:
                # 如果没有训练，使用单个文本快速构建词汇表
                self.build_vocabulary([text])
            
            # 生成字符向量
            char_vector = self.text_to_char_vector(text)
            
            # 添加一些简单的语义特征
            semantic_features = self.extract_semantic_features(text)
            
            # 组合特征
            combined_vector = np.concatenate([
                char_vector[:self.embedding_dim//2],
                semantic_features[:self.embedding_dim//2]
            ])
            
            # 确保维度正确
            if len(combined_vector) < self.embedding_dim:
                padded_vector = np.zeros(self.embedding_dim)
                padded_vector[:len(combined_vector)] = combined_vector
                combined_vector = padded_vector
            elif len(combined_vector) > self.embedding_dim:
                combined_vector = combined_vector[:self.embedding_dim]
            
            # 最终归一化
            norm = np.linalg.norm(combined_vector)
            if norm > 0:
                combined_vector = combined_vector / norm
            
            return combined_vector
            
        except Exception as e:
            logger.error(f"文本转embedding失败: {e}")
            return np.zeros(self.embedding_dim)
    
    def extract_semantic_features(self, text):
        """
        提取简单的语义特征
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 语义特征向量
        """
        try:
            features = np.zeros(self.embedding_dim // 2)
            
            if not text:
                return features
            
            # 文本长度特征
            features[0] = min(len(text) / 100.0, 1.0)
            
            # 关键词特征
            keywords = {
                '动物': ['猫', '狗', '鸟', '鱼', '动物', '宠物'],
                '建筑': ['建筑', '房子', '大楼', '城市', '街道'],
                '交通': ['车', '汽车', '飞机', '火车', '交通'],
                '自然': ['花', '树', '山', '水', '天空', '阳光'],
                '人物': ['人', '男', '女', '孩子', '老人'],
                '食物': ['食物', '饭', '菜', '水果', '蛋糕'],
                '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
                '情感': ['美丽', '可爱', '快乐', '悲伤', '愤怒']
            }
            
            feature_idx = 1
            for category, words in keywords.items():
                if feature_idx >= len(features):
                    break
                
                count = sum(1 for word in words if word in text)
                features[feature_idx] = min(count / len(words), 1.0)
                feature_idx += 1
            
            return features
            
        except Exception as e:
            logger.error(f"语义特征提取失败: {e}")
            return np.zeros(self.embedding_dim // 2)
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        计算两个embedding之间的相似度
        
        Args:
            embedding1: 第一个embedding向量
            embedding2: 第二个embedding向量
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            # 确保输入是numpy数组
            emb1 = np.array(embedding1).flatten()
            emb2 = np.array(embedding2).flatten()
            
            # 计算余弦相似度
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # 将相似度从[-1, 1]映射到[0, 1]
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0


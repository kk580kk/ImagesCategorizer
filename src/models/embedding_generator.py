"""
Embedding生成器
将图像描述和文本转换为向量表示
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import logging
import json
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Embedding生成器类"""
    
    def __init__(self, embedding_dim=1024):
        """
        初始化embedding生成器
        
        Args:
            embedding_dim: embedding维度
        """
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(
            max_features=embedding_dim,
            stop_words=None,  # 保留中文停用词处理
            ngram_range=(1, 2),  # 使用1-2gram
            min_df=1,
            max_df=1.0  # 修改为1.0以支持少量文档
        )
        self.is_fitted = False
        self.text_corpus = []
        logger.info(f"初始化Embedding生成器，维度: {embedding_dim}")
    
    def add_text_to_corpus(self, text):
        """
        添加文本到语料库
        
        Args:
            text: 文本内容
        """
        if text and isinstance(text, str):
            self.text_corpus.append(text)
            logger.debug(f"添加文本到语料库: {text[:50]}...")
    
    def fit_vectorizer(self, texts=None):
        """
        训练向量化器
        
        Args:
            texts: 文本列表，如果为None则使用内部语料库
        """
        try:
            if texts is None:
                texts = self.text_corpus
            
            if not texts:
                logger.warning("没有文本数据用于训练向量化器")
                return False
            
            # 过滤空文本
            valid_texts = [text for text in texts if text and isinstance(text, str)]
            
            if not valid_texts:
                logger.warning("没有有效的文本数据")
                return False
            
            self.vectorizer.fit(valid_texts)
            self.is_fitted = True
            logger.info(f"向量化器训练完成，使用{len(valid_texts)}个文本样本")
            return True
            
        except Exception as e:
            logger.error(f"向量化器训练失败: {e}")
            return False
    
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
                logger.warning("向量化器未训练，使用单个文本进行快速训练")
                self.fit_vectorizer([text])
            
            if not text or not isinstance(text, str):
                logger.warning("输入文本无效")
                return np.zeros(self.embedding_dim)
            
            # 生成TF-IDF向量
            tfidf_vector = self.vectorizer.transform([text])
            
            # 转换为密集向量
            dense_vector = tfidf_vector.toarray()[0]
            
            # 如果维度不足，用零填充
            if len(dense_vector) < self.embedding_dim:
                padded_vector = np.zeros(self.embedding_dim)
                padded_vector[:len(dense_vector)] = dense_vector
                dense_vector = padded_vector
            elif len(dense_vector) > self.embedding_dim:
                dense_vector = dense_vector[:self.embedding_dim]
            
            # 归一化
            norm = np.linalg.norm(dense_vector)
            if norm > 0:
                dense_vector = dense_vector / norm
            
            logger.debug(f"文本转换为embedding成功，维度: {len(dense_vector)}")
            return dense_vector
            
        except Exception as e:
            logger.error(f"文本转embedding失败: {e}")
            return np.zeros(self.embedding_dim)
    
    def image_description_to_embedding(self, description):
        """
        将图像描述转换为embedding向量
        
        Args:
            description: 图像描述文本
            
        Returns:
            np.ndarray: embedding向量
        """
        return self.text_to_embedding(description)
    
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
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)
            
            # 计算余弦相似度
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            # 将相似度从[-1, 1]映射到[0, 1]
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def save_model(self, filepath):
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
        """
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'embedding_dim': self.embedding_dim,
                'is_fitted': self.is_fitted,
                'text_corpus': self.text_corpus
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"模型保存成功: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            return False
    
    def load_model(self, filepath):
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"模型文件不存在: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.embedding_dim = model_data['embedding_dim']
            self.is_fitted = model_data['is_fitted']
            self.text_corpus = model_data['text_corpus']
            
            logger.info(f"模型加载成功: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def batch_text_to_embeddings(self, texts):
        """
        批量将文本转换为embeddings
        
        Args:
            texts: 文本列表
            
        Returns:
            list: embedding向量列表
        """
        embeddings = []
        for text in texts:
            embedding = self.text_to_embedding(text)
            embeddings.append(embedding)
        
        logger.info(f"批量转换完成，处理了{len(texts)}个文本")
        return embeddings


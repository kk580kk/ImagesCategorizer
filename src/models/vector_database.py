"""
简化的向量数据库实现
用于存储和检索图像向量
"""
import os
import json
import pickle
import logging
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVectorDatabase:
    """简化的向量数据库类"""
    
    def __init__(self, db_path="vector_db"):
        """
        初始化向量数据库
        
        Args:
            db_path: 数据库存储路径
        """
        self.db_path = db_path
        self.vectors = {}  # 存储向量数据
        self.metadata = {}  # 存储元数据
        self.index_file = os.path.join(db_path, "index.json")
        self.vectors_file = os.path.join(db_path, "vectors.pkl")
        
        # 创建数据库目录
        os.makedirs(db_path, exist_ok=True)
        
        # 加载现有数据
        self.load_database()
        
        logger.info(f"向量数据库初始化完成，路径: {db_path}")
    
    def generate_id(self, data):
        """
        为数据生成唯一ID
        
        Args:
            data: 数据内容
            
        Returns:
            str: 唯一ID
        """
        if isinstance(data, str):
            content = data
        else:
            content = str(data)
        
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def insert_vector(self, vector_id, vector, metadata=None):
        """
        插入向量到数据库
        
        Args:
            vector_id: 向量ID
            vector: 向量数据
            metadata: 元数据
            
        Returns:
            bool: 是否成功
        """
        try:
            # 确保向量是numpy数组
            vector_array = np.array(vector)
            
            # 存储向量
            self.vectors[vector_id] = vector_array
            
            # 存储元数据
            if metadata is None:
                metadata = {}
            
            metadata['vector_id'] = vector_id
            metadata['dimension'] = len(vector_array)
            metadata['created_at'] = datetime.now().isoformat()
            
            self.metadata[vector_id] = metadata
            
            logger.info(f"向量插入成功: {vector_id}")
            return True
            
        except Exception as e:
            logger.error(f"向量插入失败: {vector_id}, {e}")
            return False
    
    def insert_image_vector(self, image_path, vector, description=None, category=None):
        """
        插入图像向量
        
        Args:
            image_path: 图像路径
            vector: 图像向量
            description: 图像描述
            category: 图像类别
            
        Returns:
            str: 向量ID
        """
        try:
            # 生成向量ID
            vector_id = self.generate_id(image_path)
            
            # 准备元数据
            metadata = {
                'type': 'image',
                'image_path': image_path,
                'description': description,
                'category': category,
                'filename': os.path.basename(image_path)
            }
            
            # 插入向量
            success = self.insert_vector(vector_id, vector, metadata)
            
            if success:
                return vector_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"图像向量插入失败: {image_path}, {e}")
            return None
    
    def search_similar_vectors(self, query_vector, top_k=9, threshold=0.0):
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回最相似的k个结果
            threshold: 相似度阈值
            
        Returns:
            list: 相似向量列表
        """
        try:
            if not self.vectors:
                logger.warning("数据库中没有向量数据")
                return []
            
            query_array = np.array(query_vector).reshape(1, -1)
            similarities = []
            
            # 计算与所有向量的相似度
            for vector_id, stored_vector in self.vectors.items():
                stored_array = stored_vector.reshape(1, -1)
                
                # 计算余弦相似度
                similarity = cosine_similarity(query_array, stored_array)[0][0]
                
                # 将相似度从[-1, 1]映射到[0, 1]
                similarity = (similarity + 1) / 2
                
                if similarity >= threshold:
                    similarities.append({
                        'vector_id': vector_id,
                        'similarity': float(similarity),
                        'metadata': self.metadata.get(vector_id, {})
                    })
            
            # 按相似度排序
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 返回top_k结果
            result = similarities[:top_k]
            logger.info(f"找到{len(result)}个相似向量")
            
            return result
            
        except Exception as e:
            logger.error(f"相似向量搜索失败: {e}")
            return []
    
    def search_images_by_text(self, text_vector, top_k=9):
        """
        根据文本向量搜索相似图像
        
        Args:
            text_vector: 文本向量
            top_k: 返回最相似的k个结果
            
        Returns:
            list: 相似图像列表
        """
        try:
            # 搜索所有相似向量
            similar_vectors = self.search_similar_vectors(text_vector, top_k * 2)  # 搜索更多结果
            
            # 过滤出图像类型的结果
            image_results = []
            for result in similar_vectors:
                metadata = result['metadata']
                if metadata.get('type') == 'image':
                    image_results.append({
                        'image_path': metadata.get('image_path'),
                        'description': metadata.get('description'),
                        'category': metadata.get('category'),
                        'similarity': result['similarity'],
                        'vector_id': result['vector_id']
                    })
                
                if len(image_results) >= top_k:
                    break
            
            logger.info(f"根据文本找到{len(image_results)}个相似图像")
            return image_results
            
        except Exception as e:
            logger.error(f"文本搜索图像失败: {e}")
            return []
    
    def get_vector_by_id(self, vector_id):
        """
        根据ID获取向量
        
        Args:
            vector_id: 向量ID
            
        Returns:
            dict: 向量数据和元数据
        """
        try:
            if vector_id not in self.vectors:
                return None
            
            return {
                'vector_id': vector_id,
                'vector': self.vectors[vector_id],
                'metadata': self.metadata.get(vector_id, {})
            }
            
        except Exception as e:
            logger.error(f"获取向量失败: {vector_id}, {e}")
            return None
    
    def delete_vector(self, vector_id):
        """
        删除向量
        
        Args:
            vector_id: 向量ID
            
        Returns:
            bool: 是否成功
        """
        try:
            if vector_id in self.vectors:
                del self.vectors[vector_id]
            
            if vector_id in self.metadata:
                del self.metadata[vector_id]
            
            logger.info(f"向量删除成功: {vector_id}")
            return True
            
        except Exception as e:
            logger.error(f"向量删除失败: {vector_id}, {e}")
            return False
    
    def get_database_stats(self):
        """
        获取数据库统计信息
        
        Returns:
            dict: 统计信息
        """
        try:
            total_vectors = len(self.vectors)
            
            # 统计不同类型的向量
            type_counts = {}
            category_counts = {}
            
            for metadata in self.metadata.values():
                vector_type = metadata.get('type', 'unknown')
                type_counts[vector_type] = type_counts.get(vector_type, 0) + 1
                
                if vector_type == 'image':
                    category = metadata.get('category', 'unknown')
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            return {
                'total_vectors': total_vectors,
                'type_counts': type_counts,
                'category_counts': category_counts,
                'database_path': self.db_path
            }
            
        except Exception as e:
            logger.error(f"统计信息获取失败: {e}")
            return {'total_vectors': 0, 'error': str(e)}
    
    def save_database(self):
        """
        保存数据库到文件
        
        Returns:
            bool: 是否成功
        """
        try:
            # 保存元数据索引
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            # 保存向量数据
            with open(self.vectors_file, 'wb') as f:
                pickle.dump(self.vectors, f)
            
            logger.info("数据库保存成功")
            return True
            
        except Exception as e:
            logger.error(f"数据库保存失败: {e}")
            return False
    
    def load_database(self):
        """
        从文件加载数据库
        
        Returns:
            bool: 是否成功
        """
        try:
            # 加载元数据索引
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            # 加载向量数据
            if os.path.exists(self.vectors_file):
                with open(self.vectors_file, 'rb') as f:
                    self.vectors = pickle.load(f)
            
            logger.info(f"数据库加载成功，包含{len(self.vectors)}个向量")
            return True
            
        except Exception as e:
            logger.error(f"数据库加载失败: {e}")
            return False
    
    def clear_database(self):
        """
        清空数据库
        
        Returns:
            bool: 是否成功
        """
        try:
            self.vectors = {}
            self.metadata = {}
            
            # 删除文件
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
            
            if os.path.exists(self.vectors_file):
                os.remove(self.vectors_file)
            
            logger.info("数据库清空成功")
            return True
            
        except Exception as e:
            logger.error(f"数据库清空失败: {e}")
            return False


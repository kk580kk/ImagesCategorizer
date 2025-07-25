"""
检索引擎
整合混合向量生成、零样本分类和向量数据库功能
使用multimodal-embedding-v1和text-embedding-v4，无降级机制
"""
import os
import logging
import numpy as np
from .hybrid_embedding_generator import HybridEmbeddingGenerator
from .zero_shot_classifier import ZeroShotClassifier
from .vector_database import SimpleVectorDatabase
from config import TOP_K_RESULTS, DASHSCOPE_API_KEY

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalEngine:
    """检索引擎类"""
    
    def __init__(self, db_path="vector_db"):
        """
        初始化检索引擎
        
        Args:
            db_path: 向量数据库路径
        """
        # 使用混合向量生成器
        self.embedding_generator = HybridEmbeddingGenerator(
            api_key=DASHSCOPE_API_KEY
        )
        
        self.classifier = ZeroShotClassifier()
        
        # 使用简单向量数据库（暂时替代Zilliz）
        try:
            self.vector_db = SimpleVectorDatabase(db_path)
            logger.info("使用简单向量数据库")
        except Exception as e:
            logger.error(f"向量数据库连接失败: {e}")
            raise Exception(f"向量数据库初始化失败: {str(e)}")
        
        logger.info("检索引擎初始化完成 - 混合向量架构")
    
    def add_image_to_database(self, image_path):
        """
        添加图像到数据库
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 处理结果
        """
        try:
            logger.info(f"开始处理图像: {image_path}")
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                logger.error(f"图像文件不存在: {image_path}")
                return {'success': False, 'error': '图像文件不存在'}
            
            # 使用混合向量生成器生成图像向量
            try:
                image_embedding = self.embedding_generator.generate_image_embedding(image_path)
                logger.info(f"图像向量生成成功，维度: {len(image_embedding)}")
            except Exception as e:
                error_msg = f"图像向量生成失败: {str(e)}"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}
            
            # 进行零样本分类
            classification_result = self.classifier.classify_image(image_path)
            category = None
            description = ""
            
            if classification_result:
                category = classification_result.get('category', 'unknown')
                description = classification_result.get('description', '')
            
            # 插入向量数据库
            try:
                vector_id = self.vector_db.insert_image_vector(
                    image_path=image_path,
                    vector=image_embedding,
                    description=description,
                    category=category
                )
                
                if vector_id:
                    result = {
                        'success': True,
                        'vector_id': vector_id,
                        'image_path': image_path,
                        'description': description,
                        'category': category,
                        'classification_result': classification_result
                    }
                    
                    logger.info(f"图像添加成功: {image_path}, 类别: {category}")
                    return result
                else:
                    logger.error(f"向量数据库插入失败: {image_path}")
                    return {'success': False, 'error': '向量数据库插入失败'}
            except Exception as e:
                error_msg = f"向量数据库操作失败: {str(e)}"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}
                
        except Exception as e:
            error_msg = f"图像添加失败: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def search_images_by_text(self, query_text, top_k=None):
        """
        根据文本搜索相似图像
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            list: 搜索结果
        """
        try:
            if top_k is None:
                top_k = TOP_K_RESULTS
            
            logger.info(f"开始文本搜索: {query_text}")
            
            # 使用混合向量生成器处理查询文本
            try:
                text_embedding = self.embedding_generator.generate_text_embedding(query_text)
                logger.info(f"文本向量生成成功，维度: {len(text_embedding)}")
            except Exception as e:
                error_msg = f"文本向量生成失败: {str(e)}"
                logger.error(error_msg)
                return []
            
            # 在向量数据库中搜索
            try:
                search_results = self.vector_db.search_images_by_text(
                    text_embedding, top_k
                )
                
                # 格式化结果
                formatted_results = []
                for result in search_results:
                    formatted_result = {
                        'image_path': result['image_path'],
                        'description': result['description'],
                        'category': result['category'],
                        'similarity': result['similarity'],
                        'similarity_percentage': round(result['similarity'] * 100, 1)
                    }
                    formatted_results.append(formatted_result)
                
                logger.info(f"文本搜索完成，找到{len(formatted_results)}个结果")
                return formatted_results
            except Exception as e:
                error_msg = f"向量数据库搜索失败: {str(e)}"
                logger.error(error_msg)
                return []
            
        except Exception as e:
            logger.error(f"文本搜索失败: {e}")
            return []
    
    def search_similar_images(self, image_path, top_k=None):
        """
        根据图像搜索相似图像
        
        Args:
            image_path: 查询图像路径
            top_k: 返回结果数量
            
        Returns:
            list: 搜索结果
        """
        try:
            if top_k is None:
                top_k = TOP_K_RESULTS
            
            logger.info(f"开始图像搜索: {image_path}")
            
            # 使用混合向量生成器处理查询图像
            try:
                image_embedding = self.embedding_generator.generate_image_embedding(image_path)
                logger.info(f"图像向量生成成功，维度: {len(image_embedding)}")
            except Exception as e:
                error_msg = f"图像向量生成失败: {str(e)}"
                logger.error(error_msg)
                return []
            
            # 在向量数据库中搜索
            try:
                search_results = self.vector_db.search_similar_vectors(
                    image_embedding, top_k + 1  # +1 因为可能包含自己
                )
                
                # 过滤掉自己，格式化结果
                formatted_results = []
                for result in search_results:
                    metadata = result['metadata']
                    result_image_path = metadata.get('image_path')
                    
                    # 跳过自己
                    if result_image_path == image_path:
                        continue
                    
                    if metadata.get('type') == 'image':
                        formatted_result = {
                            'image_path': result_image_path,
                            'description': metadata.get('description'),
                            'category': metadata.get('category'),
                            'similarity': result['similarity'],
                            'similarity_percentage': round(result['similarity'] * 100, 1)
                        }
                        formatted_results.append(formatted_result)
                    
                    if len(formatted_results) >= top_k:
                        break
                
                logger.info(f"图像搜索完成，找到{len(formatted_results)}个结果")
                return formatted_results
            except Exception as e:
                error_msg = f"向量数据库搜索失败: {str(e)}"
                logger.error(error_msg)
                return []
            
        except Exception as e:
            logger.error(f"图像搜索失败: {e}")
            return []
    
    def batch_add_images(self, image_paths):
        """
        批量添加图像到数据库
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            dict: 批量处理结果
        """
        try:
            results = []
            success_count = 0
            
            for image_path in image_paths:
                result = self.add_image_to_database(image_path)
                results.append(result)
                
                if result.get('success', False):
                    success_count += 1
            
            # 保存数据库
            self.vector_db.save_database()
            
            summary = {
                'total': len(image_paths),
                'success': success_count,
                'failed': len(image_paths) - success_count,
                'results': results
            }
            
            logger.info(f"批量添加完成: {success_count}/{len(image_paths)} 成功")
            return summary
            
        except Exception as e:
            logger.error(f"批量添加失败: {e}")
            return {'total': 0, 'success': 0, 'failed': 0, 'error': str(e)}
    
    def get_database_statistics(self):
        """
        获取数据库统计信息
        
        Returns:
            dict: 统计信息
        """
        try:
            # 获取向量数据库统计
            db_stats = self.vector_db.get_database_stats()
            
            # 获取分类器统计
            classifier_stats = self.classifier.get_classification_statistics()
            
            # 合并统计信息
            combined_stats = {
                'database': db_stats,
                'classification': classifier_stats,
                'engine_status': 'active'
            }
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"统计信息获取失败: {e}")
            return {'error': str(e)}
    
    def classify_image(self, image_path):
        """
        对图像进行分类
        
        Args:
            image_path: 图像路径
            
        Returns:
            dict: 分类结果
        """
        try:
            return self.classifier.classify_image(image_path)
        except Exception as e:
            logger.error(f"图像分类失败: {e}")
            return None
    
    def save_all_data(self):
        """
        保存所有数据
        
        Returns:
            bool: 是否成功
        """
        try:
            # 保存向量数据库
            db_saved = self.vector_db.save_database()
            
            # 保存分类历史
            classifier_saved = self.classifier.save_classification_history("classification_history.json")
            
            success = db_saved and classifier_saved
            
            if success:
                logger.info("所有数据保存成功")
            else:
                logger.warning("部分数据保存失败")
            
            return success
            
        except Exception as e:
            logger.error(f"数据保存失败: {e}")
            return False
    
    def load_all_data(self):
        """
        加载所有数据
        
        Returns:
            bool: 是否成功
        """
        try:
            # 加载向量数据库
            db_loaded = self.vector_db.load_database()
            
            success = db_loaded  # 至少数据库要加载成功
            
            if success:
                logger.info("数据加载成功")
            else:
                logger.warning("数据加载失败")
            
            return success
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return False
    
    def clear_all_data(self):
        """
        清空所有数据
        
        Returns:
            bool: 是否成功
        """
        try:
            # 清空向量数据库
            db_cleared = self.vector_db.clear_database()
            
            # 清空分类历史
            self.classifier.classification_history = []
            
            logger.info("所有数据清空成功")
            return True
            
        except Exception as e:
            logger.error(f"数据清空失败: {e}")
            return False


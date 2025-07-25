#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型图文检索引擎
整合多维度信息提取和增强向量存储，提供完整的检索功能
"""

import logging
from typing import List, Dict, Any, Optional
import os
from datetime import datetime

from .multi_dimensional_extractor import MultiDimensionalExtractor
from .enhanced_vector_storage import EnhancedVectorStorage
from .zero_shot_classifier import ZeroShotClassifier

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRetrievalEngine:
    """增强型图文检索引擎"""
    
    def __init__(self):
        """初始化检索引擎"""
        self.extractor = MultiDimensionalExtractor()
        self.vector_storage = EnhancedVectorStorage()
        self.classifier = ZeroShotClassifier()
        logger.info("增强型检索引擎初始化完成")
    
    def add_image_to_database(self, image_path: str, description: str = None) -> Dict[str, Any]:
        """添加图像到数据库（增强版）"""
        try:
            logger.info(f"添加图像到数据库（多维度提取）: {image_path}")
            
            # 多维度信息提取
            extraction_results = self.extractor.extract_all_dimensions(image_path)
            
            # 进行分类
            classification_result = self.classifier.classify_image(image_path)
            
            # 将分类结果添加到提取结果中
            extraction_results['classification'] = {
                'category': classification_result.get('category', 'unknown'),
                'confidence': classification_result.get('confidence', 0.0),
                'method': classification_result.get('method', 'unknown'),
                'classification_time': datetime.now().isoformat()
            }
            
            # 存储多维度数据
            image_id = self.vector_storage.store_multi_dimensional_data(image_path, extraction_results)
            
            # 准备返回的元数据
            stats = extraction_results.get('statistics', {})
            metadata = {
                'image_id': image_id,
                'image_path': image_path,
                'extraction_statistics': stats,
                'classification': classification_result,
                'total_dimensions': stats.get('total_dimensions', 0),
                'successful_dimensions': stats.get('successful_dimensions', 0),
                'total_content_length': stats.get('total_content_length', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"图像添加成功: {image_id}")
            logger.info(f"提取统计: {stats.get('successful_dimensions', 0)}/{stats.get('total_dimensions', 0)} 维度成功")
            
            return {
                'success': True,
                'image_id': image_id,
                'metadata': metadata,
                'extraction_results': extraction_results
            }
            
        except Exception as e:
            logger.error(f"添加图像失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_images_by_text(self, query: str, top_k: int = 9, 
                             dimension_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """通过文本查询搜索图像（多维度）"""
        try:
            logger.info(f"多维度文本搜索: {query}")
            
            # 多维度搜索
            results = self.vector_storage.search_multi_dimensional(
                query=query, 
                top_k=top_k,
                dimension_filter=dimension_filter
            )
            
            # 处理搜索结果
            processed_results = []
            for result in results:
                # 获取图像的所有维度信息
                image_id = result.get('image_id')
                all_dimensions = self.vector_storage.get_image_all_dimensions(image_id)
                
                processed_results.append({
                    'image_id': image_id,
                    'image_path': result.get('image_path', ''),
                    'similarity_score': result.get('final_score', 0.0),
                    'best_dimension': result.get('best_dimension', 'unknown'),
                    'best_content': result.get('best_content', ''),
                    'dimension_count': result.get('dimension_count', 0),
                    'dimension_scores': result.get('dimension_scores', {}),
                    'all_dimensions': all_dimensions
                })
            
            logger.info(f"多维度文本搜索完成，返回 {len(processed_results)} 个结果")
            return processed_results
            
        except Exception as e:
            logger.error(f"文本搜索失败: {str(e)}")
            return []
    
    def search_by_dimension(self, query: str, dimension: str, top_k: int = 9) -> List[Dict[str, Any]]:
        """按指定维度搜索"""
        try:
            logger.info(f"按维度搜索: {dimension}, 查询: {query}")
            
            results = self.vector_storage.search_by_dimension(query, dimension, top_k)
            
            processed_results = []
            for result in results:
                metadata = result.get('metadata', {})
                processed_results.append({
                    'image_id': metadata.get('image_id', ''),
                    'image_path': metadata.get('image_path', ''),
                    'dimension': dimension,
                    'content': metadata.get('content', ''),
                    'similarity_score': result.get('score', 0.0),
                    'content_length': metadata.get('content_length', 0)
                })
            
            logger.info(f"维度搜索完成，返回 {len(processed_results)} 个结果")
            return processed_results
            
        except Exception as e:
            logger.error(f"维度搜索失败: {str(e)}")
            return []
    
    def search_similar_images(self, image_path: str, top_k: int = 9) -> List[Dict[str, Any]]:
        """搜索相似图像（基于多维度分析）"""
        try:
            logger.info(f"相似图像搜索: {image_path}")
            
            # 对查询图像进行多维度分析
            extraction_results = self.extractor.extract_all_dimensions(image_path)
            
            # 获取组合描述
            combined_description = self.extractor.get_combined_description(extraction_results)
            
            # 使用组合描述进行搜索
            results = self.vector_storage.search_multi_dimensional(
                query=combined_description,
                top_k=top_k + 1  # +1 以防包含查询图像本身
            )
            
            # 处理搜索结果，排除查询图像本身
            processed_results = []
            for result in results:
                result_image_path = result.get('image_path', '')
                
                # 跳过查询图像本身
                if os.path.abspath(result_image_path) == os.path.abspath(image_path):
                    continue
                
                # 获取图像的所有维度信息
                image_id = result.get('image_id')
                all_dimensions = self.vector_storage.get_image_all_dimensions(image_id)
                
                processed_results.append({
                    'image_id': image_id,
                    'image_path': result_image_path,
                    'similarity_score': result.get('final_score', 0.0),
                    'best_dimension': result.get('best_dimension', 'unknown'),
                    'best_content': result.get('best_content', ''),
                    'dimension_count': result.get('dimension_count', 0),
                    'all_dimensions': all_dimensions
                })
                
                if len(processed_results) >= top_k:
                    break
            
            logger.info(f"相似图像搜索完成，返回 {len(processed_results)} 个结果")
            return processed_results
            
        except Exception as e:
            logger.error(f"相似图像搜索失败: {str(e)}")
            return []
    
    def get_image_detailed_info(self, image_id: str) -> Dict[str, Any]:
        """获取图像的详细信息"""
        try:
            all_dimensions = self.vector_storage.get_image_all_dimensions(image_id)
            
            if not all_dimensions:
                return {'error': '图像不存在或无维度信息'}
            
            # 整理维度信息
            dimension_info = {}
            image_path = None
            
            for dimension_name, metadata in all_dimensions.items():
                if not image_path:
                    image_path = metadata.get('image_path')
                
                dimension_info[dimension_name] = {
                    'display_name': metadata.get('dimension_display_name', dimension_name),
                    'content': metadata.get('content', ''),
                    'content_length': metadata.get('content_length', 0),
                    'weight': metadata.get('weight', 1.0),
                    'extraction_time': metadata.get('extraction_time', '')
                }
            
            return {
                'image_id': image_id,
                'image_path': image_path,
                'dimension_count': len(dimension_info),
                'dimensions': dimension_info
            }
            
        except Exception as e:
            logger.error(f"获取图像详细信息失败: {str(e)}")
            return {'error': str(e)}
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息（增强版）"""
        try:
            # 获取维度统计
            dimension_stats = self.vector_storage.get_dimension_statistics()
            
            # 获取分类统计（从向量数据库中提取）
            all_vectors = self.vector_storage.vector_db.get_all_vectors()
            classification_stats = {}
            image_count = 0
            
            processed_images = set()
            for vector_data in all_vectors:
                metadata = vector_data.get('metadata', {})
                image_id = metadata.get('image_id')
                
                if image_id and image_id not in processed_images:
                    processed_images.add(image_id)
                    image_count += 1
            
            # 简化的分类统计（由于新架构中分类信息可能不在每个向量中）
            # 这里可以根据需要进一步优化
            
            return {
                'total_images': image_count,
                'total_vectors': dimension_stats['total_vectors'],
                'unique_dimensions': dimension_stats['unique_dimensions'],
                'dimension_statistics': dimension_stats['dimension_statistics'],
                'classification_distribution': classification_stats,  # 可能为空
                'database_info': 'Enhanced Multi-dimensional Storage'
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {
                'total_images': 0,
                'total_vectors': 0,
                'error': str(e)
            }
    
    def clear_database(self) -> bool:
        """清空数据库"""
        try:
            logger.info("清空增强型向量数据库")
            self.vector_storage.vector_db.clear_database()
            logger.info("数据库清空成功")
            return True
        except Exception as e:
            logger.error(f"清空数据库失败: {str(e)}")
            return False
    
    def remove_image(self, image_id: str) -> bool:
        """从数据库中移除图像（所有维度）"""
        try:
            logger.info(f"移除图像所有维度数据: {image_id}")
            self.vector_storage.clear_image_data(image_id)
            logger.info("图像移除成功")
            return True
        except Exception as e:
            logger.error(f"移除图像失败: {str(e)}")
            return False
    
    def update_dimension_weights(self, new_weights: Dict[str, float]) -> bool:
        """更新维度权重"""
        try:
            self.vector_storage.update_dimension_weights(new_weights)
            logger.info(f"维度权重更新成功: {new_weights}")
            return True
        except Exception as e:
            logger.error(f"更新维度权重失败: {str(e)}")
            return False


if __name__ == "__main__":
    # 测试增强型检索引擎
    engine = EnhancedRetrievalEngine()
    
    # 测试添加图像
    test_image = "static/test_images/cat1.jpg"
    if os.path.exists(test_image):
        print("测试多维度图像添加...")
        result = engine.add_image_to_database(test_image)
        print(f"添加结果: {result['success']}")
        
        if result['success']:
            image_id = result['image_id']
            
            # 测试文本搜索
            print("\n测试多维度文本搜索...")
            search_results = engine.search_images_by_text("可爱的猫咪", top_k=3)
            print(f"搜索结果: {len(search_results)} 个")
            
            # 测试按维度搜索
            print("\n测试按维度搜索...")
            dim_results = engine.search_by_dimension("猫咪", "basic_visual_description", top_k=3)
            print(f"维度搜索结果: {len(dim_results)} 个")
            
            # 测试获取详细信息
            print("\n测试获取图像详细信息...")
            detailed_info = engine.get_image_detailed_info(image_id)
            print(f"维度数量: {detailed_info.get('dimension_count', 0)}")
            
            # 测试统计信息
            print("\n测试数据库统计...")
            stats = engine.get_database_statistics()
            print(f"总图像数: {stats.get('total_images', 0)}")
            print(f"总向量数: {stats.get('total_vectors', 0)}")
            print(f"维度类型: {stats.get('unique_dimensions', 0)}")
        
    else:
        print(f"测试图像不存在: {test_image}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型多维度向量存储系统
支持多维度信息的分离存储和检索
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import json
import uuid

from .simple_embedding_generator import SimpleEmbeddingGenerator
from .vector_database import SimpleVectorDatabase

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedVectorStorage:
    """增强型多维度向量存储系统"""
    
    def __init__(self):
        """初始化增强型向量存储"""
        self.text_embedder = SimpleEmbeddingGenerator()
        self.vector_db = SimpleVectorDatabase()
        
        # 维度权重配置
        self.dimension_weights = {
            'basic_visual_description': 1.0,
            'person_identification': 1.2,
            'emotion_atmosphere': 0.9,
            'scene_context': 1.0,
            'technical_artistic': 0.7,
            'semantic_tags': 1.1
        }
        
        logger.info("增强型向量存储系统初始化完成")
    
    def store_multi_dimensional_data(self, image_path: str, extraction_results: Dict[str, Any]) -> str:
        """存储多维度数据"""
        logger.info(f"开始存储多维度数据: {image_path}")
        
        # 生成唯一的图像ID
        image_id = str(uuid.uuid4())
        
        # 存储维度数据统计
        stored_dimensions = 0
        total_dimensions = 0
        
        if 'dimensions' not in extraction_results:
            logger.error("提取结果中没有维度数据")
            return image_id
        
        # 存储每个维度的数据
        for dimension_name, dimension_data in extraction_results['dimensions'].items():
            total_dimensions += 1
            
            if 'error' in dimension_data:
                logger.warning(f"跳过错误维度: {dimension_name}")
                continue
            
            content = dimension_data.get('content', '')
            if not content or not content.strip():
                logger.warning(f"跳过空内容维度: {dimension_name}")
                continue
            
            try:
                # 生成向量
                vector = self.text_embedder.generate_embedding(content)
                
                # 准备存储数据
                storage_data = {
                    'id': f"{image_id}_{dimension_name}",
                    'image_id': image_id,
                    'image_path': image_path,
                    'dimension': dimension_name,
                    'dimension_display_name': dimension_data.get('display_name', dimension_name),
                    'content': content,
                    'content_length': len(content),
                    'weight': dimension_data.get('weight', 1.0),
                    'extraction_time': dimension_data.get('extraction_time', datetime.now().isoformat()),
                    'storage_time': datetime.now().isoformat()
                }
                
                # 存储到向量数据库
                self.vector_db.add_vector(
                    vector_id=storage_data['id'],
                    vector=vector,
                    metadata=storage_data
                )
                
                stored_dimensions += 1
                logger.info(f"维度 {dimension_name} 存储成功")
                
            except Exception as e:
                logger.error(f"维度 {dimension_name} 存储失败: {str(e)}")
        
        # 存储组合描述（用于综合检索）
        try:
            combined_description = self._get_combined_description(extraction_results)
            if combined_description:
                combined_vector = self.text_embedder.generate_embedding(combined_description)
                
                combined_data = {
                    'id': f"{image_id}_combined",
                    'image_id': image_id,
                    'image_path': image_path,
                    'dimension': 'combined',
                    'dimension_display_name': '综合描述',
                    'content': combined_description,
                    'content_length': len(combined_description),
                    'weight': 1.0,
                    'storage_time': datetime.now().isoformat()
                }
                
                self.vector_db.add_vector(
                    vector_id=combined_data['id'],
                    vector=combined_vector,
                    metadata=combined_data
                )
                
                stored_dimensions += 1
                logger.info("综合描述存储成功")
        
        except Exception as e:
            logger.error(f"综合描述存储失败: {str(e)}")
        
        logger.info(f"多维度数据存储完成: {stored_dimensions}/{total_dimensions + 1} 成功")
        return image_id
    
    def search_multi_dimensional(self, query: str, top_k: int = 9, 
                                dimension_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """多维度检索"""
        logger.info(f"开始多维度检索: {query}")
        
        # 生成查询向量
        query_vector = self.text_embedder.generate_embedding(query)
        
        # 从向量数据库检索
        raw_results = self.vector_db.search(query_vector, top_k * 3)  # 获取更多候选
        
        # 按维度分组结果
        dimension_results = {}
        for result in raw_results:
            metadata = result.get('metadata', {})
            dimension = metadata.get('dimension', 'unknown')
            
            # 应用维度过滤
            if dimension_filter and dimension not in dimension_filter and dimension != 'combined':
                continue
            
            if dimension not in dimension_results:
                dimension_results[dimension] = []
            
            dimension_results[dimension].append(result)
        
        # 融合多维度结果
        final_results = self._fuse_dimension_results(dimension_results, top_k)
        
        logger.info(f"多维度检索完成，返回 {len(final_results)} 个结果")
        return final_results
    
    def search_by_dimension(self, query: str, dimension: str, top_k: int = 9) -> List[Dict[str, Any]]:
        """按指定维度检索"""
        logger.info(f"按维度检索: {dimension}, 查询: {query}")
        
        # 生成查询向量
        query_vector = self.text_embedder.generate_embedding(query)
        
        # 从向量数据库检索
        raw_results = self.vector_db.search(query_vector, top_k * 2)
        
        # 过滤指定维度的结果
        filtered_results = []
        for result in raw_results:
            metadata = result.get('metadata', {})
            if metadata.get('dimension') == dimension:
                filtered_results.append(result)
        
        # 限制返回数量
        return filtered_results[:top_k]
    
    def get_image_all_dimensions(self, image_id: str) -> Dict[str, Any]:
        """获取图像的所有维度信息"""
        logger.info(f"获取图像所有维度信息: {image_id}")
        
        # 从向量数据库获取所有相关记录
        all_vectors = self.vector_db.get_all_vectors()
        
        image_dimensions = {}
        for vector_data in all_vectors:
            metadata = vector_data.get('metadata', {})
            if metadata.get('image_id') == image_id:
                dimension = metadata.get('dimension')
                if dimension and dimension != 'combined':
                    image_dimensions[dimension] = metadata
        
        return image_dimensions
    
    def get_dimension_statistics(self) -> Dict[str, Any]:
        """获取维度统计信息"""
        all_vectors = self.vector_db.get_all_vectors()
        
        dimension_stats = {}
        total_vectors = len(all_vectors)
        
        for vector_data in all_vectors:
            metadata = vector_data.get('metadata', {})
            dimension = metadata.get('dimension', 'unknown')
            
            if dimension not in dimension_stats:
                dimension_stats[dimension] = {
                    'count': 0,
                    'total_content_length': 0,
                    'display_name': metadata.get('dimension_display_name', dimension)
                }
            
            dimension_stats[dimension]['count'] += 1
            dimension_stats[dimension]['total_content_length'] += metadata.get('content_length', 0)
        
        # 计算平均长度
        for stats in dimension_stats.values():
            if stats['count'] > 0:
                stats['average_content_length'] = stats['total_content_length'] / stats['count']
            else:
                stats['average_content_length'] = 0
        
        return {
            'total_vectors': total_vectors,
            'dimension_statistics': dimension_stats,
            'unique_dimensions': len(dimension_stats)
        }
    
    def _get_combined_description(self, extraction_results: Dict[str, Any]) -> str:
        """获取组合描述"""
        if 'dimensions' not in extraction_results:
            return ""
        
        combined_parts = []
        
        # 按权重排序维度
        sorted_dimensions = sorted(
            extraction_results['dimensions'].items(),
            key=lambda x: x[1].get('weight', 1.0),
            reverse=True
        )
        
        for dim_name, dim_data in sorted_dimensions:
            if 'error' in dim_data:
                continue
                
            content = dim_data.get('content', '')
            if content and content.strip():
                # 添加维度标识和内容
                combined_parts.append(f"[{dim_data.get('display_name', dim_name)}] {content}")
        
        return '\n\n'.join(combined_parts)
    
    def _fuse_dimension_results(self, dimension_results: Dict[str, List[Dict[str, Any]]], 
                               top_k: int) -> List[Dict[str, Any]]:
        """融合多维度检索结果"""
        # 收集所有候选结果
        all_candidates = {}
        
        for dimension, results in dimension_results.items():
            weight = self.dimension_weights.get(dimension, 1.0)
            
            for i, result in enumerate(results):
                metadata = result.get('metadata', {})
                image_id = metadata.get('image_id')
                
                if not image_id:
                    continue
                
                # 计算加权分数
                base_score = result.get('score', 0.0)
                position_penalty = i * 0.01  # 位置惩罚
                weighted_score = base_score * weight - position_penalty
                
                if image_id not in all_candidates:
                    all_candidates[image_id] = {
                        'image_id': image_id,
                        'image_path': metadata.get('image_path'),
                        'best_score': weighted_score,
                        'total_score': weighted_score,
                        'dimension_scores': {dimension: weighted_score},
                        'dimension_count': 1,
                        'best_dimension': dimension,
                        'best_content': metadata.get('content', ''),
                        'all_metadata': [metadata]
                    }
                else:
                    candidate = all_candidates[image_id]
                    candidate['total_score'] += weighted_score
                    candidate['dimension_scores'][dimension] = weighted_score
                    candidate['dimension_count'] += 1
                    candidate['all_metadata'].append(metadata)
                    
                    # 更新最佳维度
                    if weighted_score > candidate['best_score']:
                        candidate['best_score'] = weighted_score
                        candidate['best_dimension'] = dimension
                        candidate['best_content'] = metadata.get('content', '')
        
        # 计算最终分数并排序
        final_candidates = []
        for candidate in all_candidates.values():
            # 综合分数 = 总分数 + 维度数量奖励
            final_score = candidate['total_score'] + candidate['dimension_count'] * 0.1
            candidate['final_score'] = final_score
            final_candidates.append(candidate)
        
        # 按最终分数排序
        final_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 返回前top_k个结果
        return final_candidates[:top_k]
    
    def update_dimension_weights(self, new_weights: Dict[str, float]):
        """更新维度权重"""
        self.dimension_weights.update(new_weights)
        logger.info(f"维度权重已更新: {self.dimension_weights}")
    
    def clear_image_data(self, image_id: str):
        """清除指定图像的所有数据"""
        logger.info(f"清除图像数据: {image_id}")
        
        # 获取所有相关向量ID
        all_vectors = self.vector_db.get_all_vectors()
        vector_ids_to_remove = []
        
        for vector_data in all_vectors:
            metadata = vector_data.get('metadata', {})
            if metadata.get('image_id') == image_id:
                vector_ids_to_remove.append(vector_data.get('id'))
        
        # 删除向量
        for vector_id in vector_ids_to_remove:
            self.vector_db.remove_vector(vector_id)
        
        logger.info(f"已删除 {len(vector_ids_to_remove)} 个向量")


if __name__ == "__main__":
    # 测试增强型向量存储
    storage = EnhancedVectorStorage()
    
    print("增强型向量存储系统测试完成")
    print(f"维度权重配置: {storage.dimension_weights}")
    
    # 获取统计信息
    stats = storage.get_dimension_statistics()
    print(f"当前统计信息: {stats}")


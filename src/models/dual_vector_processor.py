#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向量库存储和检索系统
整合qwen-vl-plus深度分析、multimodal-embedding-v1、text-embedding-v4和Zilliz存储
"""

import logging
import os
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image
import json

# 导入自定义模块
from .qwen_vl_deep_analyzer import QwenVLDeepAnalyzer
from .zilliz_vector_database import ZillizVectorDatabase, ZILLIZ_CONFIG
from .hybrid_embedding_generator import HybridEmbeddingGenerator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualVectorProcessor:
    """双向量库处理器"""
    
    def __init__(self, api_key: str):
        """
        初始化双向量处理器
        
        Args:
            api_key: 阿里云API密钥
        """
        self.api_key = api_key
        
        # 初始化组件
        logger.info("正在初始化双向量处理器...")
        
        # 深度图像分析器
        self.deep_analyzer = QwenVLDeepAnalyzer(api_key)
        
        # Zilliz向量数据库
        self.vector_db = ZillizVectorDatabase(ZILLIZ_CONFIG)
        
        # 混合向量生成器
        self.embedding_generator = HybridEmbeddingGenerator(api_key)
        
        logger.info("双向量处理器初始化完成")
    
    def _generate_image_id(self, image_path: str) -> str:
        """
        生成图像唯一ID
        
        Args:
            image_path: 图像路径
            
        Returns:
            str: 图像ID
        """
        # 使用文件路径和修改时间生成唯一ID
        file_stat = os.stat(image_path)
        content = f"{image_path}_{file_stat.st_mtime}_{file_stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        提取图像基本信息
        
        Args:
            image_path: 图像路径
            
        Returns:
            Dict: 图像信息
        """
        try:
            # 获取文件信息
            file_stat = os.stat(image_path)
            file_name = os.path.basename(image_path)
            
            # 获取图像尺寸
            with Image.open(image_path) as img:
                width, height = img.size
            
            return {
                "file_name": file_name,
                "file_size": file_stat.st_size,
                "image_width": width,
                "image_height": height,
                "upload_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"提取图像信息失败: {str(e)}")
            return {
                "file_name": os.path.basename(image_path),
                "file_size": 0,
                "image_width": 0,
                "image_height": 0,
                "upload_time": datetime.now().isoformat()
            }
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        处理单张图像，生成双向量并存储
        
        Args:
            image_path: 图像路径
            
        Returns:
            Dict: 处理结果
        """
        logger.info(f"开始处理图像: {image_path}")
        start_time = time.time()
        
        try:
            # 1. 生成图像ID和基本信息
            image_id = self._generate_image_id(image_path)
            image_info = self._extract_image_info(image_path)
            
            logger.info(f"图像ID: {image_id}")
            
            # 2. 使用qwen-vl-plus进行深度分析
            logger.info("正在进行深度图像分析...")
            analysis_result = self.deep_analyzer.generate_comprehensive_analysis(image_path)
            
            if "error" in analysis_result:
                return {
                    "success": False,
                    "error": analysis_result["error"],
                    "image_id": image_id,
                    "image_path": image_path
                }
            
            # 3. 生成图像多模态向量
            logger.info("正在生成图像多模态向量...")
            image_vector = self.embedding_generator.generate_image_embedding(image_path)
            
            if image_vector is None:
                return {
                    "success": False,
                    "error": "图像向量生成失败",
                    "image_id": image_id,
                    "image_path": image_path
                }
            
            # 4. 存储图像向量到Zilliz
            image_data = {
                "vector": image_vector,
                "image_id": image_id,
                "image_path": image_path,
                "vector_type": "multimodal",
                **image_info
            }
            
            image_insert_success = self.vector_db.insert_image_vector(image_data)
            
            # 5. 处理文本描述并生成向量
            text_insert_count = 0
            text_total_chars = 0
            
            for desc_type, desc_content in analysis_result.items():
                if desc_type == "_metadata":
                    continue
                
                # 确保desc_content是字符串
                if isinstance(desc_content, list):
                    desc_content = " ".join(str(item) for item in desc_content)
                elif not isinstance(desc_content, str):
                    desc_content = str(desc_content)
                
                if not desc_content or len(desc_content.strip()) < 10:
                    continue
                
                logger.info(f"正在处理 {desc_type} 描述...")
                
                # 生成文本向量
                text_vector = self.embedding_generator.generate_text_embedding(desc_content)
                
                if text_vector is not None:
                    # 存储文本向量到Zilliz
                    text_data = {
                        "vector": text_vector,
                        "image_id": image_id,
                        "description_id": f"{image_id}_{desc_type}",
                        "description_text": desc_content,
                        "description_type": desc_type,
                        "text_length": len(desc_content),
                        "confidence": 0.9,  # 默认置信度
                        "generation_time": datetime.now().isoformat(),
                        "vector_type": "text"
                    }
                    
                    if self.vector_db.insert_text_vector(text_data):
                        text_insert_count += 1
                        text_total_chars += len(desc_content)
            
            # 6. 计算处理结果
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "image_id": image_id,
                "image_path": image_path,
                "processing_time": processing_time,
                "image_vector_inserted": image_insert_success,
                "text_vectors_inserted": text_insert_count,
                "total_text_characters": text_total_chars,
                "analysis_dimensions": len([k for k in analysis_result.keys() if k != "_metadata"]),
                "image_info": image_info
            }
            
            logger.info(f"图像处理完成: {image_id}, 耗时: {processing_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path,
                "processing_time": time.time() - start_time
            }
    
    def process_batch_images(self, image_paths: List[str], batch_size: int = 5) -> Dict[str, Any]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批次大小
            
        Returns:
            Dict: 批量处理结果
        """
        logger.info(f"开始批量处理 {len(image_paths)} 张图像，批次大小: {batch_size}")
        start_time = time.time()
        
        results = {
            "total_images": len(image_paths),
            "processed_images": 0,
            "successful_images": 0,
            "failed_images": 0,
            "total_text_vectors": 0,
            "total_text_characters": 0,
            "processing_time": 0,
            "results": [],
            "errors": []
        }
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"处理进度: {i}/{len(image_paths)} - {image_path}")
            
            try:
                # 处理单张图像
                result = self.process_single_image(image_path)
                results["results"].append(result)
                results["processed_images"] += 1
                
                if result["success"]:
                    results["successful_images"] += 1
                    results["total_text_vectors"] += result.get("text_vectors_inserted", 0)
                    results["total_text_characters"] += result.get("total_text_characters", 0)
                else:
                    results["failed_images"] += 1
                    results["errors"].append({
                        "image_path": image_path,
                        "error": result.get("error", "未知错误")
                    })
                
                # 批次间延迟，避免API限流
                if i % batch_size == 0 and i < len(image_paths):
                    logger.info(f"批次完成，等待3秒...")
                    time.sleep(3)
                
            except Exception as e:
                logger.error(f"处理图像 {image_path} 时发生异常: {str(e)}")
                results["processed_images"] += 1
                results["failed_images"] += 1
                results["errors"].append({
                    "image_path": image_path,
                    "error": str(e)
                })
        
        results["processing_time"] = time.time() - start_time
        
        logger.info(f"批量处理完成! 成功: {results['successful_images']}, 失败: {results['failed_images']}, 总耗时: {results['processing_time']:.2f}秒")
        
        return results
    
    def search_by_text(self, query_text: str, top_k: int = 9) -> List[Dict[str, Any]]:
        """
        基于文本查询搜索图像
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果
        """
        logger.info(f"文本搜索: '{query_text}', top_k: {top_k}")
        
        try:
            # 1. 生成查询文本向量
            query_vector = self.embedding_generator.generate_text_embedding(query_text)
            
            if query_vector is None:
                logger.error("查询文本向量生成失败")
                return []
            
            # 2. 在文本向量集合中搜索
            text_results = self.vector_db.search_text_vectors(query_vector, top_k * 2)  # 搜索更多结果用于去重
            
            # 3. 按图像ID去重并获取图像信息
            seen_image_ids = set()
            final_results = []
            
            for text_result in text_results:
                image_id = text_result.get("image_id")
                if image_id and image_id not in seen_image_ids:
                    seen_image_ids.add(image_id)
                    
                    # 获取图像信息
                    image_info = self.vector_db.get_image_by_id(image_id)
                    
                    if image_info:
                        final_results.append({
                            "image_id": image_id,
                            "image_path": image_info.get("image_path"),
                            "file_name": image_info.get("file_name"),
                            "text_similarity": text_result.get("score", 0),
                            "matched_description": text_result.get("description_text", "")[:200] + "...",
                            "description_type": text_result.get("description_type", "unknown")
                        })
                    
                    if len(final_results) >= top_k:
                        break
            
            logger.info(f"文本搜索完成，返回 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            logger.error(f"文本搜索失败: {str(e)}")
            return []
    
    def search_by_image(self, image_path: str, top_k: int = 9) -> List[Dict[str, Any]]:
        """
        基于图像搜索相似图像
        
        Args:
            image_path: 查询图像路径
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果
        """
        logger.info(f"图像搜索: {image_path}, top_k: {top_k}")
        
        try:
            # 1. 生成查询图像向量
            query_vector = self.embedding_generator.generate_image_embedding(image_path)
            
            if query_vector is None:
                logger.error("查询图像向量生成失败")
                return []
            
            # 2. 在图像向量集合中搜索
            image_results = self.vector_db.search_image_vectors(query_vector, top_k)
            
            # 3. 处理搜索结果
            final_results = []
            
            for image_result in image_results:
                final_results.append({
                    "image_id": image_result.get("image_id"),
                    "image_path": image_result.get("image_path"),
                    "file_name": image_result.get("file_name"),
                    "image_similarity": image_result.get("score", 0),
                    "upload_time": image_result.get("upload_time")
                })
            
            logger.info(f"图像搜索完成，返回 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            logger.error(f"图像搜索失败: {str(e)}")
            return []
    
    def hybrid_search(self, query_text: str, query_image_path: Optional[str] = None, 
                     top_k: int = 9, text_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        混合搜索（文本+图像）
        
        Args:
            query_text: 查询文本
            query_image_path: 查询图像路径（可选）
            top_k: 返回结果数量
            text_weight: 文本搜索权重
            
        Returns:
            List[Dict]: 搜索结果
        """
        logger.info(f"混合搜索: 文本='{query_text}', 图像={query_image_path}, 权重={text_weight}")
        
        try:
            # 1. 文本搜索
            text_results = self.search_by_text(query_text, top_k * 2)
            
            # 2. 图像搜索（如果提供了查询图像）
            image_results = []
            if query_image_path and os.path.exists(query_image_path):
                image_results = self.search_by_image(query_image_path, top_k * 2)
            
            # 3. 结果融合
            combined_results = {}
            
            # 添加文本搜索结果
            for result in text_results:
                image_id = result["image_id"]
                combined_results[image_id] = {
                    **result,
                    "text_score": result.get("text_similarity", 0) * text_weight,
                    "image_score": 0,
                    "combined_score": result.get("text_similarity", 0) * text_weight
                }
            
            # 添加图像搜索结果
            image_weight = 1.0 - text_weight
            for result in image_results:
                image_id = result["image_id"]
                image_score = result.get("image_similarity", 0) * image_weight
                
                if image_id in combined_results:
                    # 更新现有结果
                    combined_results[image_id]["image_score"] = image_score
                    combined_results[image_id]["combined_score"] += image_score
                else:
                    # 添加新结果
                    combined_results[image_id] = {
                        **result,
                        "text_score": 0,
                        "image_score": image_score,
                        "combined_score": image_score
                    }
            
            # 4. 按综合得分排序
            final_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )[:top_k]
            
            logger.info(f"混合搜索完成，返回 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            logger.error(f"混合搜索失败: {str(e)}")
            return []
    
    def get_image_details(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        获取图像详细信息
        
        Args:
            image_id: 图像ID
            
        Returns:
            Optional[Dict]: 图像详细信息
        """
        try:
            # 获取图像基本信息
            image_info = self.vector_db.get_image_by_id(image_id)
            
            if not image_info:
                return None
            
            # 获取所有文本描述
            text_descriptions = self.vector_db.get_text_descriptions_by_image_id(image_id)
            
            return {
                "image_info": image_info,
                "descriptions": text_descriptions,
                "total_descriptions": len(text_descriptions),
                "total_characters": sum(len(desc.get("description_text", "")) for desc in text_descriptions)
            }
            
        except Exception as e:
            logger.error(f"获取图像详情失败: {str(e)}")
            return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            return self.vector_db.get_collection_stats()
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {"error": str(e)}
    
    def clear_database(self) -> bool:
        """
        清空数据库
        
        Returns:
            bool: 清空是否成功
        """
        try:
            return self.vector_db.clear_collections()
        except Exception as e:
            logger.error(f"清空数据库失败: {str(e)}")
            return False
    
    def close(self):
        """关闭连接"""
        try:
            self.vector_db.close_connection()
            logger.info("双向量处理器已关闭")
        except Exception as e:
            logger.error(f"关闭连接失败: {str(e)}")


if __name__ == "__main__":
    # 测试双向量处理器
    API_KEY = "sk-71f2950a3d704e568ea7ab8ee0567447"
    
    try:
        processor = DualVectorProcessor(API_KEY)
        
        # 获取统计信息
        stats = processor.get_database_stats()
        print("数据库统计信息:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        # 关闭连接
        processor.close()
        
        print("✅ 双向量处理器测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")


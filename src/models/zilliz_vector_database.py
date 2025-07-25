#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zilliz向量数据库连接和管理模块
实现双集合架构：图像多模态向量集合 + 文本描述向量集合
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime
import json

try:
    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False
    logging.warning("pymilvus not available, using mock implementation")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZillizVectorDatabase:
    """Zilliz向量数据库管理类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Zilliz连接
        
        Args:
            config: Zilliz配置信息
        """
        self.config = config
        self.connection_alias = "zilliz_connection"
        self.image_collection_name = "image_multimodal_vectors"
        self.text_collection_name = "text_description_vectors"
        
        # 集合对象
        self.image_collection = None
        self.text_collection = None
        
        if PYMILVUS_AVAILABLE:
            self._connect()
            self._setup_collections()
        else:
            logger.warning("使用模拟模式运行，实际部署时需要安装pymilvus")
            self._setup_mock_collections()
        
        logger.info("Zilliz向量数据库初始化完成")
    
    def _connect(self):
        """连接到Zilliz"""
        try:
            # 建立连接
            connections.connect(
                alias=self.connection_alias,
                uri=self.config["endpoint"],
                token=self.config["token"],
                timeout=self.config.get("timeout", 30)
            )
            logger.info(f"成功连接到Zilliz: {self.config['endpoint']}")
            
        except Exception as e:
            logger.error(f"连接Zilliz失败: {str(e)}")
            raise Exception(f"Zilliz连接失败: {str(e)}")
    
    def _setup_collections(self):
        """设置双集合架构"""
        try:
            # 创建图像向量集合
            self._create_image_collection()
            
            # 创建文本向量集合
            self._create_text_collection()
            
            logger.info("双集合架构设置完成")
            
        except Exception as e:
            logger.error(f"集合设置失败: {str(e)}")
            raise Exception(f"集合设置失败: {str(e)}")
    
    def _create_image_collection(self):
        """创建图像多模态向量集合"""
        collection_name = self.image_collection_name
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name, using=self.connection_alias):
            logger.info(f"图像集合 {collection_name} 已存在，直接加载")
            self.image_collection = Collection(collection_name, using=self.connection_alias)
            return
        
        # 定义字段schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="file_size", dtype=DataType.INT64),
            FieldSchema(name="image_width", dtype=DataType.INT64),
            FieldSchema(name="image_height", dtype=DataType.INT64),
            FieldSchema(name="upload_time", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="vector_type", dtype=DataType.VARCHAR, max_length=20)
        ]
        
        # 创建集合schema
        schema = CollectionSchema(
            fields=fields,
            description="图像多模态向量集合 - 存储multimodal-embedding-v1向量"
        )
        
        # 创建集合
        self.image_collection = Collection(
            name=collection_name,
            schema=schema,
            using=self.connection_alias
        )
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        self.image_collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        # 加载集合
        self.image_collection.load()
        
        logger.info(f"图像集合 {collection_name} 创建完成")
    
    def _create_text_collection(self):
        """创建文本描述向量集合"""
        collection_name = self.text_collection_name
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name, using=self.connection_alias):
            logger.info(f"文本集合 {collection_name} 已存在，直接加载")
            self.text_collection = Collection(collection_name, using=self.connection_alias)
            return
        
        # 定义字段schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="description_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="description_text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="description_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="text_length", dtype=DataType.INT64),
            FieldSchema(name="confidence", dtype=DataType.FLOAT),
            FieldSchema(name="generation_time", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="vector_type", dtype=DataType.VARCHAR, max_length=20)
        ]
        
        # 创建集合schema
        schema = CollectionSchema(
            fields=fields,
            description="文本描述向量集合 - 存储text-embedding-v4向量"
        )
        
        # 创建集合
        self.text_collection = Collection(
            name=collection_name,
            schema=schema,
            using=self.connection_alias
        )
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        self.text_collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        # 加载集合
        self.text_collection.load()
        
        logger.info(f"文本集合 {collection_name} 创建完成")
    
    def _setup_mock_collections(self):
        """设置模拟集合（用于测试）"""
        self.mock_image_data = []
        self.mock_text_data = []
        logger.info("模拟集合设置完成")
    
    def insert_image_vector(self, image_data: Dict[str, Any]) -> bool:
        """
        插入图像向量数据
        
        Args:
            image_data: 图像向量数据
            
        Returns:
            bool: 插入是否成功
        """
        try:
            if not PYMILVUS_AVAILABLE:
                # 模拟插入
                self.mock_image_data.append(image_data)
                logger.info(f"模拟插入图像向量: {image_data.get('image_id', 'unknown')}")
                return True
            
            # 准备插入数据
            insert_data = [
                [image_data["vector"]],
                [image_data["image_id"]],
                [image_data["image_path"]],
                [image_data["file_name"]],
                [image_data["file_size"]],
                [image_data["image_width"]],
                [image_data["image_height"]],
                [image_data["upload_time"]],
                [image_data["vector_type"]]
            ]
            
            # 插入数据
            result = self.image_collection.insert(insert_data)
            
            # 刷新集合
            self.image_collection.flush()
            
            logger.info(f"图像向量插入成功: {image_data['image_id']}")
            return True
            
        except Exception as e:
            logger.error(f"图像向量插入失败: {str(e)}")
            return False
    
    def insert_text_vector(self, text_data: Dict[str, Any]) -> bool:
        """
        插入文本向量数据
        
        Args:
            text_data: 文本向量数据
            
        Returns:
            bool: 插入是否成功
        """
        try:
            if not PYMILVUS_AVAILABLE:
                # 模拟插入
                self.mock_text_data.append(text_data)
                logger.info(f"模拟插入文本向量: {text_data.get('description_id', 'unknown')}")
                return True
            
            # 准备插入数据
            insert_data = [
                [text_data["vector"]],
                [text_data["image_id"]],
                [text_data["description_id"]],
                [text_data["description_text"]],
                [text_data["description_type"]],
                [text_data["text_length"]],
                [text_data["confidence"]],
                [text_data["generation_time"]],
                [text_data["vector_type"]]
            ]
            
            # 插入数据
            result = self.text_collection.insert(insert_data)
            
            # 刷新集合
            self.text_collection.flush()
            
            logger.info(f"文本向量插入成功: {text_data['description_id']}")
            return True
            
        except Exception as e:
            logger.error(f"文本向量插入失败: {str(e)}")
            return False
    
    def search_image_vectors(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        搜索图像向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果
        """
        try:
            if not PYMILVUS_AVAILABLE:
                # 模拟搜索
                logger.info(f"模拟搜索图像向量，返回 {min(top_k, len(self.mock_image_data))} 个结果")
                return self.mock_image_data[:top_k]
            
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            results = self.image_collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["image_id", "image_path", "file_name", "upload_time"]
            )
            
            # 处理结果
            processed_results = []
            for hits in results:
                for hit in hits:
                    processed_results.append({
                        "score": hit.score,
                        "image_id": hit.entity.get("image_id"),
                        "image_path": hit.entity.get("image_path"),
                        "file_name": hit.entity.get("file_name"),
                        "upload_time": hit.entity.get("upload_time")
                    })
            
            logger.info(f"图像向量搜索完成，返回 {len(processed_results)} 个结果")
            return processed_results
            
        except Exception as e:
            logger.error(f"图像向量搜索失败: {str(e)}")
            return []
    
    def search_text_vectors(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        搜索文本向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果
        """
        try:
            if not PYMILVUS_AVAILABLE:
                # 模拟搜索
                logger.info(f"模拟搜索文本向量，返回 {min(top_k, len(self.mock_text_data))} 个结果")
                return self.mock_text_data[:top_k]
            
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            results = self.text_collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["image_id", "description_id", "description_text", "description_type", "text_length"]
            )
            
            # 处理结果
            processed_results = []
            for hits in results:
                for hit in hits:
                    processed_results.append({
                        "score": hit.score,
                        "image_id": hit.entity.get("image_id"),
                        "description_id": hit.entity.get("description_id"),
                        "description_text": hit.entity.get("description_text"),
                        "description_type": hit.entity.get("description_type"),
                        "text_length": hit.entity.get("text_length")
                    })
            
            logger.info(f"文本向量搜索完成，返回 {len(processed_results)} 个结果")
            return processed_results
            
        except Exception as e:
            logger.error(f"文本向量搜索失败: {str(e)}")
            return []
    
    def get_image_by_id(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        根据图像ID获取图像信息
        
        Args:
            image_id: 图像ID
            
        Returns:
            Optional[Dict]: 图像信息
        """
        try:
            if not PYMILVUS_AVAILABLE:
                # 模拟查询
                for data in self.mock_image_data:
                    if data.get("image_id") == image_id:
                        return data
                return None
            
            # 查询条件
            expr = f'image_id == "{image_id}"'
            
            # 执行查询
            results = self.image_collection.query(
                expr=expr,
                output_fields=["image_id", "image_path", "file_name", "file_size", "image_width", "image_height", "upload_time"]
            )
            
            if results:
                return results[0]
            return None
            
        except Exception as e:
            logger.error(f"查询图像信息失败: {str(e)}")
            return None
    
    def get_text_descriptions_by_image_id(self, image_id: str) -> List[Dict[str, Any]]:
        """
        根据图像ID获取所有文本描述
        
        Args:
            image_id: 图像ID
            
        Returns:
            List[Dict]: 文本描述列表
        """
        try:
            if not PYMILVUS_AVAILABLE:
                # 模拟查询
                results = []
                for data in self.mock_text_data:
                    if data.get("image_id") == image_id:
                        results.append(data)
                return results
            
            # 查询条件
            expr = f'image_id == "{image_id}"'
            
            # 执行查询
            results = self.text_collection.query(
                expr=expr,
                output_fields=["description_id", "description_text", "description_type", "text_length", "confidence", "generation_time"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"查询文本描述失败: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            if not PYMILVUS_AVAILABLE:
                return {
                    "image_collection": {
                        "name": self.image_collection_name,
                        "count": len(self.mock_image_data),
                        "status": "模拟模式"
                    },
                    "text_collection": {
                        "name": self.text_collection_name,
                        "count": len(self.mock_text_data),
                        "status": "模拟模式"
                    }
                }
            
            # 获取图像集合统计
            image_stats = self.image_collection.num_entities
            
            # 获取文本集合统计
            text_stats = self.text_collection.num_entities
            
            return {
                "image_collection": {
                    "name": self.image_collection_name,
                    "count": image_stats,
                    "status": "正常"
                },
                "text_collection": {
                    "name": self.text_collection_name,
                    "count": text_stats,
                    "status": "正常"
                }
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {
                "error": str(e)
            }
    
    def clear_collections(self) -> bool:
        """
        清空所有集合
        
        Returns:
            bool: 清空是否成功
        """
        try:
            if not PYMILVUS_AVAILABLE:
                self.mock_image_data.clear()
                self.mock_text_data.clear()
                logger.info("模拟集合已清空")
                return True
            
            # 清空图像集合
            if self.image_collection:
                self.image_collection.drop()
                
            # 清空文本集合
            if self.text_collection:
                self.text_collection.drop()
            
            # 重新创建集合
            self._setup_collections()
            
            logger.info("所有集合已清空并重新创建")
            return True
            
        except Exception as e:
            logger.error(f"清空集合失败: {str(e)}")
            return False
    
    def close_connection(self):
        """关闭连接"""
        try:
            if PYMILVUS_AVAILABLE:
                connections.disconnect(self.connection_alias)
            logger.info("Zilliz连接已关闭")
        except Exception as e:
            logger.error(f"关闭连接失败: {str(e)}")


# 全局配置
ZILLIZ_CONFIG = {
    "cluster_id": "in05-8b029938b95f2b9",
    "endpoint": "https://in05-8b029938b95f2b9.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
    "token": "77cb0581cc572d4ae6ece28240a428760c4039d53aef0205ac63cdd6422f1269f9d35832f8b327822f84a71c24ec800c6ee27a85",
    "timeout": 30
}


if __name__ == "__main__":
    # 测试Zilliz连接
    try:
        db = ZillizVectorDatabase(ZILLIZ_CONFIG)
        
        # 获取统计信息
        stats = db.get_collection_stats()
        print("集合统计信息:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        # 关闭连接
        db.close_connection()
        
        print("Zilliz向量数据库测试完成")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")


"""
图文对齐器
实现图像和文本的对齐和特征提取
"""
import os
import json
import logging
import numpy as np
from PIL import Image
from .qwen_vl_model import QwenVLModel
from .simple_embedding_generator import SimpleEmbeddingGenerator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageTextAligner:
    """图文对齐器类"""
    
    def __init__(self):
        """初始化图文对齐器"""
        self.qwen_model = QwenVLModel()
        self.embedding_generator = SimpleEmbeddingGenerator()
        self.image_features = {}  # 存储图像特征
        self.text_features = {}   # 存储文本特征
        self.alignments = {}      # 存储对齐关系
        logger.info("图文对齐器初始化完成")
    
    def process_image(self, image_path, image_id=None):
        """
        处理单张图像，提取特征和生成描述
        
        Args:
            image_path: 图像文件路径
            image_id: 图像ID，如果为None则使用文件名
            
        Returns:
            dict: 包含图像特征的字典
        """
        try:
            if image_id is None:
                image_id = os.path.basename(image_path)
            
            logger.info(f"处理图像: {image_path}")
            
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                logger.error(f"图像文件不存在: {image_path}")
                return None
            
            # 验证图像文件
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as e:
                logger.error(f"图像文件损坏: {image_path}, {e}")
                return None
            
            # 使用Qwen-VL提取图像特征
            image_description = self.qwen_model.extract_image_features(image_path)
            
            if not image_description:
                logger.error(f"图像特征提取失败: {image_path}")
                return None
            
            # 生成embedding
            image_embedding = self.embedding_generator.text_to_embedding(image_description)
            
            # 存储图像特征
            image_features = {
                'image_id': image_id,
                'image_path': image_path,
                'description': image_description,
                'embedding': image_embedding.tolist(),  # 转换为列表以便JSON序列化
                'embedding_norm': float(np.linalg.norm(image_embedding))
            }
            
            self.image_features[image_id] = image_features
            logger.info(f"图像处理完成: {image_id}")
            
            return image_features
            
        except Exception as e:
            logger.error(f"图像处理失败: {image_path}, {e}")
            return None
    
    def process_text(self, text, text_id=None):
        """
        处理文本，生成特征和embedding
        
        Args:
            text: 输入文本
            text_id: 文本ID，如果为None则使用文本哈希
            
        Returns:
            dict: 包含文本特征的字典
        """
        try:
            if text_id is None:
                import hashlib
                text_id = hashlib.md5(text.encode()).hexdigest()[:8]
            
            logger.info(f"处理文本: {text_id}")
            
            # 标准化文本
            processed_text = self.qwen_model.generate_text_embedding_prompt(text)
            
            # 生成embedding
            text_embedding = self.embedding_generator.text_to_embedding(processed_text)
            
            # 存储文本特征
            text_features = {
                'text_id': text_id,
                'original_text': text,
                'processed_text': processed_text,
                'embedding': text_embedding.tolist(),
                'embedding_norm': float(np.linalg.norm(text_embedding))
            }
            
            self.text_features[text_id] = text_features
            logger.info(f"文本处理完成: {text_id}")
            
            return text_features
            
        except Exception as e:
            logger.error(f"文本处理失败: {text}, {e}")
            return None
    
    def align_image_text(self, image_id, text_id, manual_alignment=False):
        """
        对齐图像和文本
        
        Args:
            image_id: 图像ID
            text_id: 文本ID
            manual_alignment: 是否手动对齐
            
        Returns:
            dict: 对齐结果
        """
        try:
            if image_id not in self.image_features:
                logger.error(f"图像特征不存在: {image_id}")
                return None
            
            if text_id not in self.text_features:
                logger.error(f"文本特征不存在: {text_id}")
                return None
            
            # 获取embeddings
            image_embedding = np.array(self.image_features[image_id]['embedding'])
            text_embedding = np.array(self.text_features[text_id]['embedding'])
            
            # 计算相似度
            similarity = self.embedding_generator.calculate_similarity(
                image_embedding, text_embedding
            )
            
            # 创建对齐记录
            alignment = {
                'image_id': image_id,
                'text_id': text_id,
                'similarity': float(similarity),
                'manual_alignment': manual_alignment,
                'image_description': self.image_features[image_id]['description'],
                'text_content': self.text_features[text_id]['original_text']
            }
            
            alignment_key = f"{image_id}_{text_id}"
            self.alignments[alignment_key] = alignment
            
            logger.info(f"图文对齐完成: {alignment_key}, 相似度: {similarity:.3f}")
            
            return alignment
            
        except Exception as e:
            logger.error(f"图文对齐失败: {image_id}, {text_id}, {e}")
            return None
    
    def find_similar_images(self, query_text, top_k=5):
        """
        根据文本查询相似图像
        
        Args:
            query_text: 查询文本
            top_k: 返回最相似的k个结果
            
        Returns:
            list: 相似图像列表，按相似度排序
        """
        try:
            # 处理查询文本
            query_features = self.process_text(query_text, "query_temp")
            if not query_features:
                return []
            
            query_embedding = np.array(query_features['embedding'])
            
            # 计算与所有图像的相似度
            similarities = []
            for image_id, image_features in self.image_features.items():
                image_embedding = np.array(image_features['embedding'])
                similarity = self.embedding_generator.calculate_similarity(
                    query_embedding, image_embedding
                )
                
                similarities.append({
                    'image_id': image_id,
                    'image_path': image_features['image_path'],
                    'description': image_features['description'],
                    'similarity': float(similarity)
                })
            
            # 按相似度排序
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 返回top_k结果
            result = similarities[:top_k]
            logger.info(f"找到{len(result)}个相似图像")
            
            return result
            
        except Exception as e:
            logger.error(f"相似图像查询失败: {e}")
            return []
    
    def find_similar_texts(self, image_path, top_k=5):
        """
        根据图像查询相似文本
        
        Args:
            image_path: 图像路径
            top_k: 返回最相似的k个结果
            
        Returns:
            list: 相似文本列表，按相似度排序
        """
        try:
            # 处理查询图像
            image_features = self.process_image(image_path, "query_image_temp")
            if not image_features:
                return []
            
            image_embedding = np.array(image_features['embedding'])
            
            # 计算与所有文本的相似度
            similarities = []
            for text_id, text_features in self.text_features.items():
                text_embedding = np.array(text_features['embedding'])
                similarity = self.embedding_generator.calculate_similarity(
                    image_embedding, text_embedding
                )
                
                similarities.append({
                    'text_id': text_id,
                    'text_content': text_features['original_text'],
                    'similarity': float(similarity)
                })
            
            # 按相似度排序
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 返回top_k结果
            result = similarities[:top_k]
            logger.info(f"找到{len(result)}个相似文本")
            
            return result
            
        except Exception as e:
            logger.error(f"相似文本查询失败: {e}")
            return []
    
    def save_features(self, filepath):
        """
        保存所有特征到文件
        
        Args:
            filepath: 保存路径
        """
        try:
            data = {
                'image_features': self.image_features,
                'text_features': self.text_features,
                'alignments': self.alignments
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"特征保存成功: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"特征保存失败: {e}")
            return False
    
    def load_features(self, filepath):
        """
        从文件加载特征
        
        Args:
            filepath: 特征文件路径
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"特征文件不存在: {filepath}")
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.image_features = data.get('image_features', {})
            self.text_features = data.get('text_features', {})
            self.alignments = data.get('alignments', {})
            
            # 重新构建词汇表
            all_texts = []
            for features in self.image_features.values():
                all_texts.append(features['description'])
            for features in self.text_features.values():
                all_texts.append(features['processed_text'])
            
            if all_texts:
                self.embedding_generator.build_vocabulary(all_texts)
            
            logger.info(f"特征加载成功: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"特征加载失败: {e}")
            return False


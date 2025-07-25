"""
零样本分类器
使用Qwen-VL-Plus模型对图像进行零样本分类
"""
import os
import json
import logging
import numpy as np
from .qwen_vl_model import QwenVLModel
from .simple_embedding_generator import SimpleEmbeddingGenerator
from config import ZERO_SHOT_LABELS

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZeroShotClassifier:
    """零样本分类器类"""
    
    def __init__(self, categories=None):
        """
        初始化零样本分类器
        
        Args:
            categories: 分类标签列表，如果为None则使用默认标签
        """
        self.qwen_model = QwenVLModel()
        self.embedding_generator = SimpleEmbeddingGenerator()
        self.categories = categories if categories else ZERO_SHOT_LABELS
        self.category_embeddings = {}
        self.classification_history = []
        
        logger.info(f"零样本分类器初始化完成，类别: {self.categories}")
        
        # 预计算类别embeddings
        self._precompute_category_embeddings()
    
    def _precompute_category_embeddings(self):
        """预计算所有类别的embedding向量"""
        try:
            logger.info("开始预计算类别embeddings...")
            
            # 为每个类别生成详细描述
            category_descriptions = {}
            for category in self.categories:
                description = self._generate_category_description(category)
                category_descriptions[category] = description
            
            # 构建词汇表
            all_descriptions = list(category_descriptions.values())
            self.embedding_generator.build_vocabulary(all_descriptions)
            
            # 计算每个类别的embedding
            for category, description in category_descriptions.items():
                embedding = self.embedding_generator.text_to_embedding(description)
                self.category_embeddings[category] = {
                    'embedding': embedding,
                    'description': description
                }
            
            logger.info(f"类别embeddings预计算完成，共{len(self.category_embeddings)}个类别")
            
        except Exception as e:
            logger.error(f"类别embeddings预计算失败: {e}")
    
    def _generate_category_description(self, category):
        """
        为类别生成详细描述
        
        Args:
            category: 类别名称
            
        Returns:
            str: 类别描述
        """
        descriptions = {
            "动物": "各种动物，包括哺乳动物、鸟类、鱼类、昆虫等，如猫、狗、鸟、鱼、老虎、大象等动物形象",
            "植物": "各种植物，包括花朵、树木、草地、森林、花园、盆栽等植物相关的图像",
            "建筑": "建筑物和建筑结构，包括房屋、大楼、桥梁、古建筑、现代建筑、城市景观等",
            "交通工具": "各种交通工具，包括汽车、火车、飞机、船只、自行车、摩托车等运输工具",
            "食物": "各种食物和饮品，包括水果、蔬菜、肉类、甜点、饮料、餐具等与饮食相关的内容",
            "人物": "人类形象，包括男性、女性、儿童、老人、不同职业的人、人物肖像等",
            "风景": "自然风景和景观，包括山川、河流、海洋、天空、日出日落、自然美景等",
            "科技产品": "科技设备和电子产品，包括手机、电脑、相机、机器人、科技设备等",
            "艺术品": "艺术作品和创作，包括绘画、雕塑、工艺品、艺术装置、创意作品等",
            "运动": "体育运动和健身活动，包括各种球类运动、健身、户外运动、体育器材等"
        }
        
        return descriptions.get(category, f"与{category}相关的图像内容")
    
    def classify_image(self, image_path):
        """
        对单张图像进行零样本分类
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 分类结果
        """
        try:
            logger.info(f"开始分类图像: {image_path}")
            
            # 检查图像文件
            if not os.path.exists(image_path):
                logger.error(f"图像文件不存在: {image_path}")
                return None
            
            # 方法1: 使用Qwen-VL直接分类
            direct_result = self.qwen_model.classify_image(image_path, self.categories)
            
            # 方法2: 使用embedding相似度分类
            embedding_result = self._classify_by_embedding(image_path)
            
            # 综合两种方法的结果
            final_result = self._combine_classification_results(
                direct_result, embedding_result, image_path
            )
            
            # 记录分类历史
            self.classification_history.append(final_result)
            
            logger.info(f"图像分类完成: {image_path}, 类别: {final_result.get('category', 'unknown')}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"图像分类失败: {image_path}, {e}")
            return None
    
    def _classify_by_embedding(self, image_path):
        """
        使用embedding相似度进行分类
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 分类结果
        """
        try:
            # 提取图像特征
            image_description = self.qwen_model.extract_image_features(image_path)
            if not image_description:
                return None
            
            # 生成图像embedding
            image_embedding = self.embedding_generator.text_to_embedding(image_description)
            
            # 计算与各类别的相似度
            similarities = {}
            for category, category_data in self.category_embeddings.items():
                category_embedding = category_data['embedding']
                similarity = self.embedding_generator.calculate_similarity(
                    image_embedding, category_embedding
                )
                similarities[category] = similarity
            
            # 找到最相似的类别
            best_category = max(similarities, key=similarities.get)
            best_similarity = similarities[best_category]
            
            return {
                'method': 'embedding',
                'category': best_category,
                'confidence': float(best_similarity),
                'similarities': similarities,
                'image_description': image_description
            }
            
        except Exception as e:
            logger.error(f"Embedding分类失败: {e}")
            return None
    
    def _combine_classification_results(self, direct_result, embedding_result, image_path):
        """
        综合两种分类方法的结果
        
        Args:
            direct_result: 直接分类结果
            embedding_result: embedding分类结果
            image_path: 图像路径
            
        Returns:
            dict: 综合分类结果
        """
        try:
            combined_result = {
                'image_path': image_path,
                'timestamp': str(np.datetime64('now')),
                'methods_used': []
            }
            
            # 处理直接分类结果
            if direct_result and 'category' in direct_result:
                combined_result['direct_classification'] = direct_result
                combined_result['methods_used'].append('direct')
            
            # 处理embedding分类结果
            if embedding_result and 'category' in embedding_result:
                combined_result['embedding_classification'] = embedding_result
                combined_result['methods_used'].append('embedding')
            
            # 决定最终分类结果
            if direct_result and embedding_result:
                # 两种方法都有结果，选择置信度更高的
                direct_confidence = direct_result.get('confidence', 0)
                embedding_confidence = embedding_result.get('confidence', 0)
                
                if direct_confidence >= embedding_confidence:
                    combined_result['category'] = direct_result['category']
                    combined_result['confidence'] = direct_confidence
                    combined_result['primary_method'] = 'direct'
                else:
                    combined_result['category'] = embedding_result['category']
                    combined_result['confidence'] = embedding_confidence
                    combined_result['primary_method'] = 'embedding'
                
                # 检查两种方法是否一致
                combined_result['methods_agree'] = (
                    direct_result['category'] == embedding_result['category']
                )
                
            elif direct_result:
                # 只有直接分类结果
                combined_result['category'] = direct_result['category']
                combined_result['confidence'] = direct_result.get('confidence', 0.5)
                combined_result['primary_method'] = 'direct'
                combined_result['methods_agree'] = None
                
            elif embedding_result:
                # 只有embedding分类结果
                combined_result['category'] = embedding_result['category']
                combined_result['confidence'] = embedding_result.get('confidence', 0.5)
                combined_result['primary_method'] = 'embedding'
                combined_result['methods_agree'] = None
                
            else:
                # 没有有效结果
                combined_result['category'] = 'unknown'
                combined_result['confidence'] = 0.0
                combined_result['primary_method'] = 'none'
                combined_result['methods_agree'] = None
            
            return combined_result
            
        except Exception as e:
            logger.error(f"结果综合失败: {e}")
            return {
                'image_path': image_path,
                'category': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def batch_classify(self, image_paths):
        """
        批量分类图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            list: 分类结果列表
        """
        results = []
        for image_path in image_paths:
            result = self.classify_image(image_path)
            if result:
                results.append(result)
        
        logger.info(f"批量分类完成，处理了{len(results)}张图像")
        return results
    
    def calculate_accuracy(self, test_data):
        """
        计算分类准确率
        
        Args:
            test_data: 测试数据，格式为[(image_path, true_label), ...]
            
        Returns:
            dict: 准确率统计
        """
        try:
            if not test_data:
                return {'accuracy': 0.0, 'total': 0, 'correct': 0}
            
            correct_predictions = 0
            total_predictions = len(test_data)
            detailed_results = []
            
            for image_path, true_label in test_data:
                result = self.classify_image(image_path)
                
                if result:
                    predicted_label = result.get('category', 'unknown')
                    is_correct = predicted_label == true_label
                    
                    if is_correct:
                        correct_predictions += 1
                    
                    detailed_results.append({
                        'image_path': image_path,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'confidence': result.get('confidence', 0.0),
                        'is_correct': is_correct
                    })
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'total': total_predictions,
                'correct': correct_predictions,
                'detailed_results': detailed_results
            }
            
        except Exception as e:
            logger.error(f"准确率计算失败: {e}")
            return {'accuracy': 0.0, 'total': 0, 'correct': 0, 'error': str(e)}
    
    def save_classification_history(self, filepath):
        """
        保存分类历史
        
        Args:
            filepath: 保存路径
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.classification_history, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分类历史保存成功: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"分类历史保存失败: {e}")
            return False
    
    def get_classification_statistics(self):
        """
        获取分类统计信息
        
        Returns:
            dict: 统计信息
        """
        try:
            if not self.classification_history:
                return {'total': 0, 'categories': {}}
            
            category_counts = {}
            total_count = len(self.classification_history)
            
            for result in self.classification_history:
                category = result.get('category', 'unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            return {
                'total': total_count,
                'categories': category_counts,
                'category_percentages': {
                    cat: count / total_count * 100 
                    for cat, count in category_counts.items()
                }
            }
            
        except Exception as e:
            logger.error(f"统计信息计算失败: {e}")
            return {'total': 0, 'categories': {}, 'error': str(e)}


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
        对单张图像进行零样本分类，使用多角度分析提高准确性
        
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
            
            # 使用qwen-vl-plus进行多角度分析和分类
            qwen_result = self.qwen_model.classify_image(image_path, self.categories)
            
            if qwen_result and qwen_result.get('category'):
                # 验证分类结果的合理性
                validated_result = self._validate_classification(qwen_result, image_path)
                
                # 记录分类历史
                classification_record = {
                    'image_path': image_path,
                    'result': validated_result,
                    'timestamp': self._get_timestamp()
                }
                self.classification_history.append(classification_record)
                
                logger.info(f"分类完成: {validated_result['category']} (置信度: {validated_result['confidence']:.3f})")
                return validated_result
            else:
                # 如果qwen分类失败，使用embedding相似度分类
                logger.warning("qwen分类失败，使用embedding相似度分类")
                return self._embedding_based_classification(image_path)
                
        except Exception as e:
            logger.error(f"零样本分类失败: {e}")
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'method': 'error',
                'reason': f'分类失败: {str(e)}'
            }
    
    def _validate_classification(self, qwen_result, image_path):
        """
        验证和优化分类结果
        
        Args:
            qwen_result: qwen模型的分类结果
            image_path: 图像路径
            
        Returns:
            dict: 验证后的分类结果
        """
        try:
            category = qwen_result.get('category', 'unknown')
            confidence = qwen_result.get('confidence', 0.5)
            reason = qwen_result.get('reason', '')
            method = qwen_result.get('method', 'qwen_vl')
            
            # 特殊验证规则
            validated_result = {
                'category': category,
                'confidence': confidence,
                'reason': reason,
                'method': method,
                'validation_notes': []
            }
            
            # 如果有多角度分析结果，进行额外验证
            if 'analysis_details' in qwen_result:
                analysis = qwen_result['analysis_details']
                validated_result = self._cross_validate_with_analysis(validated_result, analysis)
            
            # 置信度调整
            if confidence > 0.9 and category in ['人物', '运动']:
                # 对于容易混淆的类别，降低过高的置信度
                if '运动' in reason and '人物' in str(analysis.get('person_features', '')):
                    validated_result['confidence'] = min(confidence, 0.8)
                    validated_result['validation_notes'].append('人物-运动混淆风险，置信度调整')
            
            return validated_result
            
        except Exception as e:
            logger.error(f"分类验证失败: {e}")
            return qwen_result
    
    def _cross_validate_with_analysis(self, result, analysis):
        """
        基于多角度分析进行交叉验证
        
        Args:
            result: 初始分类结果
            analysis: 多角度分析结果
            
        Returns:
            dict: 交叉验证后的结果
        """
        try:
            category = result['category']
            person_features = analysis.get('person_features', '').lower()
            activity_features = analysis.get('activity_features', '').lower()
            basic_content = analysis.get('basic_content', '').lower()
            
            # 人物vs运动的特殊验证
            if category == '运动':
                # 检查是否真的是运动场景
                sport_keywords = ['体育', '运动', '健身', '球类', '跑步', '游泳', '篮球', '足球', '网球']
                non_sport_keywords = ['肖像', '特写', '证件照', '头像', '人物照', '无人物', '静态']
                
                has_sport = any(keyword in activity_features for keyword in sport_keywords)
                has_non_sport = any(keyword in person_features for keyword in non_sport_keywords)
                
                if has_non_sport and not has_sport:
                    # 可能是人物被误分类为运动
                    result['category'] = '人物'
                    result['confidence'] = max(0.6, result['confidence'] * 0.8)
                    result['reason'] = f"交叉验证修正：{result['reason']} -> 检测到人物特征，修正为人物分类"
                    result['validation_notes'].append('运动->人物修正')
                    
            elif category == '人物':
                # 检查是否真的是人物场景
                if '运动' in activity_features and '体育' in activity_features:
                    # 可能需要重新考虑是否为运动
                    if result['confidence'] < 0.7:
                        result['validation_notes'].append('人物分类中检测到运动特征，需要人工确认')
            
            # 添加分析摘要
            result['analysis_summary'] = {
                'has_person': '人物' in person_features or '人' in basic_content,
                'has_sport': any(word in activity_features for word in ['运动', '体育', '健身']),
                'scene_type': self._extract_scene_type(analysis.get('scene_features', ''))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"交叉验证失败: {e}")
            return result
    
    def _extract_scene_type(self, scene_features):
        """从场景特征中提取场景类型"""
        scene_features = scene_features.lower()
        
        if '室内' in scene_features:
            return 'indoor'
        elif '室外' in scene_features:
            return 'outdoor'
        elif '运动场' in scene_features or '体育场' in scene_features:
            return 'sports_venue'
        elif '工作室' in scene_features or '摄影棚' in scene_features:
            return 'studio'
        else:
            return 'unknown'
    
    def _embedding_based_classification(self, image_path):
        """
        基于embedding的分类方法（备用）
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 分类结果
        """
        try:
            # 提取图像特征
            image_description = self.qwen_model.extract_image_features(image_path)
            if not image_description:
                return {
                    'category': 'unknown',
                    'confidence': 0.0,
                    'method': 'embedding_failed',
                    'reason': '特征提取失败'
                }
            
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
                'image_description': image_description,
                'reason': f'基于特征相似度分类，相似度: {best_similarity:.3f}'
            }
            
        except Exception as e:
            logger.error(f"Embedding分类失败: {e}")
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'method': 'embedding_error',
                'reason': f'Embedding分类失败: {str(e)}'
            }
    
    def _get_timestamp(self):
        """获取当前时间戳"""
        import datetime
        return datetime.datetime.now().isoformat()
    
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


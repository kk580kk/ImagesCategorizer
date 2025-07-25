"""
阿里云Qwen-VL-Plus模型调用类
实现图像理解和特征提取功能
"""
import dashscope
from dashscope import MultiModalConversation
import base64
import io
import json
import logging
from PIL import Image
import numpy as np
from config import DASHSCOPE_API_KEY, QWEN_VL_MODEL

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenVLModel:
    """Qwen-VL-Plus模型调用类"""
    
    def __init__(self):
        """初始化模型"""
        dashscope.api_key = DASHSCOPE_API_KEY
        self.model_name = QWEN_VL_MODEL
        logger.info(f"初始化{self.model_name}模型")
    
    def encode_image_to_base64(self, image_path):
        """将图像编码为base64格式"""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{encoded_string}"
        except Exception as e:
            logger.error(f"图像编码失败: {e}")
            return None
    
    def describe_image(self, image_path, prompt="请详细描述这张图片的内容"):
        """
        使用Qwen-VL-Plus描述图像内容
        
        Args:
            image_path: 图像文件路径
            prompt: 描述提示词
            
        Returns:
            str: 图像描述文本
        """
        try:
            # 编码图像
            image_base64 = self.encode_image_to_base64(image_path)
            if not image_base64:
                return None
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_base64},
                        {"text": prompt}
                    ]
                }
            ]
            
            # 调用模型
            response = MultiModalConversation.call(
                model=self.model_name,
                messages=messages
            )
            
            if response.status_code == 200:
                result = response.output.choices[0].message.content
                logger.info(f"图像描述成功: {image_path}")
                return result
            else:
                logger.error(f"模型调用失败: {response}")
                return None
                
        except Exception as e:
            logger.error(f"图像描述失败: {e}")
            return None
    
    def multi_angle_analysis(self, image_path):
        """
        多角度分析图像，生成详细的标签和描述
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 包含多个维度分析结果的字典
        """
        try:
            analysis_results = {}
            
            # 1. 基础内容分析
            basic_prompt = """请详细分析这张图片的基础内容：
            1. 主要对象：人物、动物、物品、建筑等
            2. 场景环境：室内/室外、具体场所
            3. 动作状态：静态/动态、具体动作
            4. 数量信息：人数、物品数量等
            请用简洁明确的语言描述。"""
            
            basic_analysis = self.describe_image(image_path, basic_prompt)
            analysis_results['basic_content'] = basic_analysis
            
            # 2. 人物特征分析（如果有人物）
            person_prompt = """请专门分析图片中的人物特征：
            1. 是否包含人物？如果有，有几个人？
            2. 人物的年龄段、性别特征
            3. 人物的服装、姿态、表情
            4. 人物在做什么？是否在运动？
            5. 人物与环境的关系
            如果没有人物，请明确说明"无人物"。"""
            
            person_analysis = self.describe_image(image_path, person_prompt)
            analysis_results['person_features'] = person_analysis
            
            # 3. 运动和活动分析
            activity_prompt = """请分析图片中的运动和活动特征：
            1. 是否包含体育运动？具体是什么运动？
            2. 是否有运动器材、运动场地？
            3. 人物是否在进行体育活动？
            4. 图片的主题是否与运动相关？
            5. 如果不是运动场景，主要活动是什么？
            请明确区分运动和非运动场景。"""
            
            activity_analysis = self.describe_image(image_path, activity_prompt)
            analysis_results['activity_features'] = activity_analysis
            
            # 4. 场景和背景分析
            scene_prompt = """请分析图片的场景和背景：
            1. 拍摄环境：室内/室外、具体场所类型
            2. 背景元素：建筑、自然景观、装饰等
            3. 光线条件：自然光/人工光、明暗程度
            4. 整体氛围：正式/休闲、安静/热闹等
            5. 场景的主要用途和特征"""
            
            scene_analysis = self.describe_image(image_path, scene_prompt)
            analysis_results['scene_features'] = scene_analysis
            
            # 5. 视觉和艺术特征
            visual_prompt = """请分析图片的视觉和艺术特征：
            1. 构图方式：特写/全景、角度、布局
            2. 色彩特征：主要颜色、色调、对比度
            3. 拍摄风格：写实/艺术、专业/业余
            4. 图片质量：清晰度、光线质量
            5. 整体美学特征"""
            
            visual_analysis = self.describe_image(image_path, visual_prompt)
            analysis_results['visual_features'] = visual_analysis
            
            logger.info(f"多角度分析完成: {image_path}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"多角度分析失败: {e}")
            return None
    
    def extract_image_features(self, image_path):
        """
        提取图像特征用于向量化，使用多角度分析
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: 图像特征描述
        """
        try:
            # 使用多角度分析获取详细特征
            analysis = self.multi_angle_analysis(image_path)
            
            if analysis:
                # 整合所有分析结果为特征描述
                feature_parts = []
                
                # 基础内容特征
                if analysis.get('basic_content'):
                    feature_parts.append(f"基础内容：{analysis['basic_content']}")
                
                # 人物特征
                if analysis.get('person_features') and '无人物' not in analysis['person_features']:
                    feature_parts.append(f"人物特征：{analysis['person_features']}")
                
                # 活动特征
                if analysis.get('activity_features'):
                    feature_parts.append(f"活动特征：{analysis['activity_features']}")
                
                # 场景特征
                if analysis.get('scene_features'):
                    feature_parts.append(f"场景特征：{analysis['scene_features']}")
                
                # 视觉特征
                if analysis.get('visual_features'):
                    feature_parts.append(f"视觉特征：{analysis['visual_features']}")
                
                comprehensive_features = " | ".join(feature_parts)
                logger.info(f"多角度特征提取完成: {image_path}")
                return comprehensive_features
            else:
                # 如果多角度分析失败，使用简单特征提取
                logger.warning("多角度分析失败，使用简单特征提取")
                return self._simple_feature_extraction(image_path)
                
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return self._simple_feature_extraction(image_path)
    
    def _simple_feature_extraction(self, image_path):
        """简单特征提取方法（备用）"""
        feature_prompt = """请从以下几个维度详细分析这张图片：
        1. 主要物体和场景
        2. 颜色和光线特征
        3. 构图和风格
        4. 情感和氛围
        5. 技术特征（如摄影技巧、艺术风格等）
        请用简洁但全面的语言描述，用于后续的向量化处理。"""
        
        return self.describe_image(image_path, feature_prompt)
    
    def classify_image(self, image_path, categories):
        """
        对图像进行分类，使用多角度分析提高准确性
        
        Args:
            image_path: 图像文件路径
            categories: 分类标签列表
            
        Returns:
            dict: 分类结果和置信度
        """
        try:
            # 首先进行多角度分析
            analysis = self.multi_angle_analysis(image_path)
            if not analysis:
                # 如果多角度分析失败，使用简单分类
                return self._simple_classify(image_path, categories)
            
            # 基于多角度分析结果进行智能分类
            categories_str = "、".join(categories)
            
            # 构建详细的分类提示词
            classify_prompt = f"""基于以下详细分析结果，请将图片准确分类到这些类别中：{categories_str}

分析结果：
基础内容：{analysis.get('basic_content', '')}
人物特征：{analysis.get('person_features', '')}
活动特征：{analysis.get('activity_features', '')}
场景特征：{analysis.get('scene_features', '')}
视觉特征：{analysis.get('visual_features', '')}

分类规则：
1. 如果图片主要是人物肖像、人物特写、人物摄影，应分类为"人物"
2. 如果图片显示人物在进行体育运动、健身活动，才分类为"运动"
3. 如果图片主要是建筑物、房屋、桥梁等，分类为"建筑"
4. 如果图片主要是动物，分类为"动物"
5. 如果图片主要是植物、花卉、树木，分类为"植物"
6. 如果图片主要是汽车、飞机、船只等，分类为"交通工具"
7. 如果图片主要是食物、饮料，分类为"食物"
8. 如果图片主要是自然风景、山水、城市景观，分类为"风景"
9. 如果图片主要是电子产品、机械设备，分类为"科技产品"
10. 如果图片主要是绘画、雕塑、艺术作品，分类为"艺术品"

请特别注意区分"人物"和"运动"：
- 人物肖像照、证件照、人物特写 → "人物"
- 人物在运动场进行体育活动 → "运动"

请按照以下格式回答：
分类：[类别名称]
置信度：[0-1之间的数值]
理由：[详细说明分类理由，引用分析结果]"""
            
            result = self.describe_image(image_path, classify_prompt)
            
            if result:
                # 解析分类结果
                classification = self._parse_classification_result(result)
                
                # 添加分析详情
                classification['analysis_details'] = analysis
                classification['method'] = 'multi_angle_analysis'
                
                return classification
            
            return None
            
        except Exception as e:
            logger.error(f"图像分类失败: {e}")
            return None
    
    def _simple_classify(self, image_path, categories):
        """简单分类方法（备用）"""
        try:
            categories_str = "、".join(categories)
            classify_prompt = f"""请将这张图片分类到以下类别中的一个：{categories_str}
            
            请按照以下格式回答：
            分类：[类别名称]
            置信度：[0-1之间的数值]
            理由：[简要说明分类理由]"""
            
            result = self.describe_image(image_path, classify_prompt)
            
            if result:
                classification = self._parse_classification_result(result)
                classification['method'] = 'simple_classify'
                return classification
            
            return None
            
        except Exception as e:
            logger.error(f"简单分类失败: {e}")
            return None
    
    def _parse_classification_result(self, result):
        """解析分类结果"""
        classification = {}
        
        try:
            lines = result.split('\n')
            
            for line in lines:
                line = line.strip()
                if '分类：' in line or '分类:' in line:
                    category = line.split('：')[1] if '：' in line else line.split(':')[1]
                    classification['category'] = category.strip()
                elif '置信度：' in line or '置信度:' in line:
                    try:
                        confidence_str = line.split('：')[1] if '：' in line else line.split(':')[1]
                        confidence_str = confidence_str.strip().replace('%', '')
                        classification['confidence'] = float(confidence_str) / 100 if float(confidence_str) > 1 else float(confidence_str)
                    except:
                        classification['confidence'] = 0.5
                elif '理由：' in line or '理由:' in line:
                    reason = line.split('：')[1] if '：' in line else line.split(':')[1]
                    classification['reason'] = reason.strip()
            
            # 设置默认值
            if 'category' not in classification:
                classification['category'] = 'unknown'
            if 'confidence' not in classification:
                classification['confidence'] = 0.5
            if 'reason' not in classification:
                classification['reason'] = '分类结果解析失败'
                
        except Exception as e:
            logger.error(f"分类结果解析失败: {e}")
            classification = {
                'category': 'unknown',
                'confidence': 0.0,
                'reason': f'解析错误: {str(e)}'
            }
        
        return classification
    
    def generate_text_embedding_prompt(self, text):
        """
        为文本生成用于embedding的标准化描述
        
        Args:
            text: 输入文本
            
        Returns:
            str: 标准化的embedding描述
        """
        try:
            embedding_prompt = f"""请将以下文本转换为适合向量化的标准化描述：
            
            原文本：{text}
            
            请提供：
            1. 核心概念和关键词
            2. 语义特征
            3. 情感色彩
            4. 上下文信息
            
            输出格式要求：简洁、准确、适合向量化处理。"""
            
            # 这里可以调用文本处理模型，暂时直接返回处理后的文本
            # 实际应用中可以调用专门的文本embedding模型
            return text  # 简化处理
            
        except Exception as e:
            logger.error(f"文本embedding生成失败: {e}")
            return text


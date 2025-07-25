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
    
    def extract_image_features(self, image_path):
        """
        提取图像特征用于向量化
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: 图像特征描述
        """
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
        对图像进行分类
        
        Args:
            image_path: 图像文件路径
            categories: 分类标签列表
            
        Returns:
            dict: 分类结果和置信度
        """
        try:
            categories_str = "、".join(categories)
            classify_prompt = f"""请将这张图片分类到以下类别中的一个：{categories_str}
            
            请按照以下格式回答：
            分类：[类别名称]
            置信度：[0-1之间的数值]
            理由：[简要说明分类理由]"""
            
            result = self.describe_image(image_path, classify_prompt)
            
            if result:
                # 解析分类结果
                lines = result.split('\n')
                classification = {}
                
                for line in lines:
                    if '分类：' in line:
                        classification['category'] = line.split('分类：')[1].strip()
                    elif '置信度：' in line:
                        try:
                            confidence_str = line.split('置信度：')[1].strip()
                            classification['confidence'] = float(confidence_str)
                        except:
                            classification['confidence'] = 0.5
                    elif '理由：' in line:
                        classification['reason'] = line.split('理由：')[1].strip()
                
                return classification
            
            return None
            
        except Exception as e:
            logger.error(f"图像分类失败: {e}")
            return None
    
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


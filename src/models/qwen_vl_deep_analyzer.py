#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen-vl-plus深度图像理解分析器
实现多角度、多层次的图像内容分析和描述生成
"""

import logging
import base64
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime

try:
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logging.warning("dashscope not available, using mock implementation")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenVLDeepAnalyzer:
    """qwen-vl-plus深度图像分析器"""
    
    def __init__(self, api_key: str):
        """
        初始化分析器
        
        Args:
            api_key: 阿里云API密钥
        """
        self.api_key = api_key
        
        if DASHSCOPE_AVAILABLE:
            dashscope.api_key = api_key
            logger.info("qwen-vl-plus深度分析器初始化完成")
        else:
            logger.warning("使用模拟模式运行，实际部署时需要安装dashscope")
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        将图像编码为base64格式
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: base64编码的图像数据
        """
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{encoded_string}"
        except Exception as e:
            logger.error(f"图像编码失败: {str(e)}")
            return ""
    
    def _call_qwen_vl_plus(self, image_path: str, prompt: str) -> str:
        """
        调用qwen-vl-plus模型进行图像分析
        
        Args:
            image_path: 图像路径
            prompt: 分析提示词
            
        Returns:
            str: 分析结果
        """
        try:
            if not DASHSCOPE_AVAILABLE:
                # 模拟响应
                return f"模拟分析结果: 基于提示词'{prompt[:50]}...'的图像分析内容。这是一个详细的分析结果，包含了丰富的描述信息。"
            
            # 编码图像
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                return "图像编码失败，无法进行分析"
            
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
            
            # 调用API
            response = dashscope.MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=messages,
                top_p=0.8,
                temperature=0.7
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                # 处理可能的列表格式响应
                if isinstance(content, list):
                    content = " ".join(str(item) for item in content)
                elif not isinstance(content, str):
                    content = str(content)
                return content
            else:
                logger.error(f"API调用失败: {response.status_code} - {response.message}")
                return f"API调用失败: {response.message}"
                
        except Exception as e:
            logger.error(f"qwen-vl-plus调用异常: {str(e)}")
            return f"分析异常: {str(e)}"
    
    def generate_basic_visual_description(self, image_path: str) -> str:
        """
        生成基础视觉描述
        
        Args:
            image_path: 图像路径
            
        Returns:
            str: 基础视觉描述
        """
        prompt = """
请详细描述这张图片的基础视觉内容：

1. **主要对象识别**: 详细描述图片中的主要人物、物品、动物等对象，包括它们的外观特征、位置关系、数量等。

2. **视觉特征分析**: 详细描述颜色搭配、形状特征、大小比例、材质质感、光影效果等视觉元素。

3. **场景环境描述**: 详细描述拍摄场景、背景环境、空间布局、氛围营造等环境因素。

4. **构图和风格**: 分析构图方式、拍摄角度、艺术风格、整体美感等艺术特征。

要求：
- 描述要详细具体，用词丰富多样
- 重点关注视觉可见的具体细节
- 字数控制在400-600字
- 语言流畅自然，富有表现力
"""
        
        result = self._call_qwen_vl_plus(image_path, prompt)
        logger.info(f"基础视觉描述生成完成，长度: {len(result)}字符")
        return result
    
    def generate_deep_recognition_description(self, image_path: str) -> str:
        """
        生成深度识别描述
        
        Args:
            image_path: 图像路径
            
        Returns:
            str: 深度识别描述
        """
        prompt = """
请对图片进行深度识别和专业分析：

1. **身份和背景识别**: 
   - 如果是人物，请尽可能识别身份、职业、年龄特征、着装风格等
   - 如果是知名人物，请提供详细的背景信息、职业经历、代表作品等
   - 如果无法确定具体身份，请分析人物的职业特征、社会角色等

2. **物品和品牌识别**:
   - 如果是产品或物品，请识别品牌、型号、技术特征、用途功能等
   - 提供相关的技术参数、市场定位、设计特色等专业信息
   - 分析产品的历史背景、发展演变等

3. **场景和地点识别**:
   - 如果是特定场景，请识别地点、建筑风格、历史背景等
   - 分析场景的文化意义、社会功能、时代特征等
   - 提供相关的地理、历史、文化背景信息

4. **专业知识补充**:
   - 提供与图片内容相关的专业知识和背景信息
   - 分析其在相关领域的重要性和影响力
   - 补充相关的历史、文化、技术等深度信息

要求：
- 提供尽可能详细和准确的识别信息
- 补充丰富的背景知识和专业信息
- 字数控制在600-1000字
- 信息要有深度和专业性
"""
        
        result = self._call_qwen_vl_plus(image_path, prompt)
        logger.info(f"深度识别描述生成完成，长度: {len(result)}字符")
        return result
    
    def generate_emotional_context_description(self, image_path: str) -> str:
        """
        生成情感氛围描述
        
        Args:
            image_path: 图像路径
            
        Returns:
            str: 情感氛围描述
        """
        prompt = """
请深入分析图片的情感氛围和文化内涵：

1. **情感基调分析**:
   - 分析图片传达的整体情感氛围（如温馨、严肃、活泼、忧郁等）
   - 描述观看者可能产生的心理感受和情感共鸣
   - 分析色彩、光线、构图等元素对情感表达的作用

2. **心理层面解读**:
   - 分析图片可能反映的心理状态、情感需求
   - 探讨其可能传达的深层心理信息
   - 分析观者的心理反应和情感体验

3. **文化背景和社会意义**:
   - 分析图片的文化背景和社会语境
   - 探讨其反映的社会现象、文化价值观
   - 分析其在特定文化环境中的意义和影响

4. **艺术价值和美学特征**:
   - 分析图片的艺术价值和美学特点
   - 探讨其艺术表现手法和创意特色
   - 分析其在艺术史或设计史中的地位和影响

要求：
- 深入挖掘情感层面和文化内涵
- 分析要有深度和洞察力
- 字数控制在300-500字
- 语言要富有感染力和表现力
"""
        
        result = self._call_qwen_vl_plus(image_path, prompt)
        logger.info(f"情感氛围描述生成完成，长度: {len(result)}字符")
        return result
    
    def generate_technical_analysis_description(self, image_path: str) -> str:
        """
        生成技术特征描述
        
        Args:
            image_path: 图像路径
            
        Returns:
            str: 技术特征描述
        """
        prompt = """
请从技术和艺术角度专业分析这张图片：

1. **拍摄技术分析**:
   - 分析拍摄设备可能的类型和参数设置
   - 分析拍摄技巧，如焦距选择、光圈设置、快门速度等
   - 评估图像质量、清晰度、噪点控制等技术指标

2. **构图和艺术手法**:
   - 分析构图原理的运用（如三分法、对称构图、引导线等）
   - 分析艺术表现手法和创意技巧
   - 评估视觉平衡、节奏感、层次感等艺术效果

3. **光线和色彩处理**:
   - 分析光线的运用和控制技巧
   - 分析色彩搭配、色调处理、饱和度控制等
   - 评估明暗对比、色彩层次、视觉冲击力等

4. **后期处理和技术效果**:
   - 分析可能的后期处理技术和效果
   - 评估图像的整体技术水准和制作质量
   - 分析特殊效果和技术创新点

要求：
- 提供专业的技术分析和评价
- 使用专业术语和技术概念
- 字数控制在250-400字
- 分析要客观准确，具有专业性
"""
        
        result = self._call_qwen_vl_plus(image_path, prompt)
        logger.info(f"技术特征描述生成完成，长度: {len(result)}字符")
        return result
    
    def generate_comprehensive_analysis(self, image_path: str) -> Dict[str, str]:
        """
        生成全面的多角度分析
        
        Args:
            image_path: 图像路径
            
        Returns:
            Dict[str, str]: 包含四个维度分析结果的字典
        """
        logger.info(f"开始对图像进行全面分析: {image_path}")
        start_time = time.time()
        
        try:
            # 生成四个维度的描述
            analysis_results = {}
            
            # 1. 基础视觉描述
            logger.info("正在生成基础视觉描述...")
            analysis_results["basic_visual"] = self.generate_basic_visual_description(image_path)
            time.sleep(1)  # 避免API调用过快
            
            # 2. 深度识别描述
            logger.info("正在生成深度识别描述...")
            analysis_results["deep_recognition"] = self.generate_deep_recognition_description(image_path)
            time.sleep(1)
            
            # 3. 情感氛围描述
            logger.info("正在生成情感氛围描述...")
            analysis_results["emotional_context"] = self.generate_emotional_context_description(image_path)
            time.sleep(1)
            
            # 4. 技术特征描述
            logger.info("正在生成技术特征描述...")
            analysis_results["technical_analysis"] = self.generate_technical_analysis_description(image_path)
            
            # 计算总时长和字符数
            total_time = time.time() - start_time
            total_chars = sum(len(desc) for desc in analysis_results.values())
            
            logger.info(f"全面分析完成！总耗时: {total_time:.2f}秒，总字符数: {total_chars}")
            
            # 添加元数据
            analysis_results["_metadata"] = {
                "image_path": image_path,
                "analysis_time": datetime.now().isoformat(),
                "total_characters": total_chars,
                "processing_time": total_time,
                "dimensions": list(analysis_results.keys())
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"全面分析失败: {str(e)}")
            return {
                "error": f"分析失败: {str(e)}",
                "image_path": image_path,
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_search_optimized_descriptions(self, image_path: str) -> List[str]:
        """
        生成针对搜索优化的描述列表
        
        Args:
            image_path: 图像路径
            
        Returns:
            List[str]: 优化的描述列表
        """
        # 获取全面分析结果
        analysis = self.generate_comprehensive_analysis(image_path)
        
        if "error" in analysis:
            return [analysis["error"]]
        
        # 提取各维度描述
        descriptions = []
        
        # 合并所有描述为搜索优化的文本块
        if "basic_visual" in analysis:
            descriptions.append(analysis["basic_visual"])
        
        if "deep_recognition" in analysis:
            descriptions.append(analysis["deep_recognition"])
        
        if "emotional_context" in analysis:
            descriptions.append(analysis["emotional_context"])
        
        if "technical_analysis" in analysis:
            descriptions.append(analysis["technical_analysis"])
        
        # 生成综合描述
        if len(descriptions) >= 2:
            combined_desc = " ".join(descriptions[:2])  # 合并前两个维度
            descriptions.append(combined_desc)
        
        logger.info(f"生成了 {len(descriptions)} 个搜索优化描述")
        return descriptions
    
    def analyze_batch_images(self, image_paths: List[str]) -> Dict[str, Dict[str, str]]:
        """
        批量分析多张图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            Dict[str, Dict[str, str]]: 批量分析结果
        """
        logger.info(f"开始批量分析 {len(image_paths)} 张图像")
        
        batch_results = {}
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"正在分析第 {i}/{len(image_paths)} 张图像: {image_path}")
            
            try:
                result = self.generate_comprehensive_analysis(image_path)
                batch_results[image_path] = result
                
                # 批量处理时添加延迟，避免API限流
                if i < len(image_paths):
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"图像 {image_path} 分析失败: {str(e)}")
                batch_results[image_path] = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        logger.info(f"批量分析完成，成功分析 {len([r for r in batch_results.values() if 'error' not in r])} 张图像")
        return batch_results


if __name__ == "__main__":
    # 测试深度分析器
    API_KEY = "sk-71f2950a3d704e568ea7ab8ee0567447"
    
    try:
        analyzer = QwenVLDeepAnalyzer(API_KEY)
        
        # 测试单张图像分析
        test_image = "/path/to/test/image.jpg"  # 需要替换为实际图像路径
        
        if DASHSCOPE_AVAILABLE:
            print("正在进行深度分析测试...")
            result = analyzer.generate_comprehensive_analysis(test_image)
            
            print("分析结果:")
            for dimension, content in result.items():
                if dimension != "_metadata":
                    print(f"\n{dimension}:")
                    print(content[:200] + "..." if len(content) > 200 else content)
        else:
            print("模拟模式测试:")
            result = analyzer.generate_comprehensive_analysis("test_image.jpg")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n✅ qwen-vl-plus深度分析器测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多维度图像信息提取器
实现6个维度的深度图像分析和信息提取
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
import json

from .qwen_vl_model import QwenVLModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiDimensionalExtractor:
    """多维度图像信息提取器"""
    
    def __init__(self):
        """初始化多维度提取器"""
        self.qwen_model = QwenVLModel()
        self.dimensions = self._define_dimensions()
        logger.info("多维度信息提取器初始化完成")
    
    def _define_dimensions(self) -> List[Dict[str, Any]]:
        """定义信息提取维度"""
        return [
            {
                'name': 'basic_visual_description',
                'display_name': '基础视觉描述',
                'description': '基础视觉内容描述',
                'weight': 1.0,
                'extraction_prompt': '''请详细描述这张图片的基础视觉内容，要求非常详细和全面：

1. 主要对象识别：
   - 图片中的人物、物体、动物等的详细描述
   - 每个对象的位置、大小、形状、颜色
   - 对象之间的空间关系和相互作用

2. 场景环境描述：
   - 拍摄地点的详细特征（室内/室外、具体场所）
   - 背景元素的详细描述
   - 环境的整体氛围和特色

3. 视觉特征分析：
   - 颜色搭配和色调分析
   - 光线条件和阴影效果
   - 构图方式和视觉焦点
   - 整体视觉风格特征

4. 技术特征评估：
   - 拍摄角度和视角选择
   - 景深效果和焦点处理
   - 图片清晰度和质量评估
   - 可能的拍摄设备和技术

请用丰富的形容词和详细的描述，让读者能够通过文字完全理解图片内容。'''
            },
            {
                'name': 'person_identification',
                'display_name': '人物信息识别',
                'description': '人物识别和详细信息',
                'weight': 1.2,
                'extraction_prompt': '''如果图片中包含人物，请进行深度分析：

1. 人物身份识别：
   - 是否为知名人物（明星、公众人物、历史人物等）
   - 如果是知名人物，请提供姓名、职业、知名度
   - 如果不是知名人物，请描述人物的基本特征

2. 外观特征详述：
   - 年龄段、性别、种族特征
   - 身高体型、体态特征
   - 发型、发色、面部特征
   - 服装风格、颜色、材质、品牌（如可识别）
   - 配饰、化妆、整体造型风格

3. 表情和姿态分析：
   - 面部表情的详细描述
   - 肢体语言和姿态含义
   - 眼神方向和情感表达
   - 整体给人的印象和感觉

4. 背景信息推测：
   - 如果是知名人物，请提供基本背景信息
   - 可能的职业、社会地位、文化背景
   - 与场景的关系和可能的拍摄目的

请尽可能详细地分析每个人物，提供丰富的信息。'''
            },
            {
                'name': 'emotion_atmosphere',
                'display_name': '情感氛围分析',
                'description': '情感表达和氛围营造',
                'weight': 0.9,
                'extraction_prompt': '''请深入分析图片的情感和氛围：

1. 情感表达分析：
   - 人物（如有）的情感状态和表达
   - 整体画面传达的情感基调
   - 观看者可能产生的情感反应
   - 情感的强度和复杂性

2. 氛围营造评估：
   - 整体氛围特征（温馨、严肃、活泼、神秘、浪漫等）
   - 氛围营造的技巧和手法
   - 色彩、光线对氛围的贡献
   - 构图和元素对氛围的影响

3. 心理感受分析：
   - 观看图片时的直观感受
   - 可能引发的联想和回忆
   - 心理层面的深层含义
   - 情感共鸣的可能性

4. 艺术意境探讨：
   - 图片想要表达的主题或意境
   - 象征意义和隐喻内容
   - 文化内涵和精神层面的表达
   - 艺术价值和审美特征

请用富有感染力的语言描述情感和氛围。'''
            },
            {
                'name': 'scene_context',
                'display_name': '场景上下文',
                'description': '场景背景和上下文信息',
                'weight': 1.0,
                'extraction_prompt': '''请分析图片的场景上下文和背景信息：

1. 拍摄场景详析：
   - 具体地点类型和特征
   - 地理位置的可能推测
   - 场所的功能和用途
   - 环境的独特性和代表性

2. 时间信息推断：
   - 可能的拍摄时间（时段、季节）
   - 时代背景和历史时期特征
   - 时间相关的线索和证据
   - 时效性和时代感

3. 事件背景分析：
   - 可能的事件、活动、场合
   - 事件的性质和重要性
   - 参与者和相关人员
   - 事件的社会意义和影响

4. 文化背景探讨：
   - 涉及的文化元素和符号
   - 地域文化特色和民族特征
   - 社会文化背景和价值观
   - 文化传承和历史意义

5. 社会语境分析：
   - 社会环境和社会关系
   - 社会地位和阶层特征
   - 社会活动和社会现象
   - 时代特征和社会变迁

请提供丰富的背景信息和深度分析。'''
            },
            {
                'name': 'technical_artistic',
                'display_name': '技术艺术特征',
                'description': '技术和艺术特征分析',
                'weight': 0.7,
                'extraction_prompt': '''请分析图片的技术和艺术特征：

1. 摄影技术分析：
   - 拍摄设备类型和规格推测
   - 镜头选择和焦距特征
   - 光圈、快门、ISO等参数分析
   - 对焦方式和景深控制
   - 拍摄角度和构图技巧

2. 艺术风格识别：
   - 摄影风格类型（纪实、艺术、商业等）
   - 色彩处理和色调风格
   - 构图方式和视觉语言
   - 艺术流派和风格特征
   - 创意表现和艺术手法

3. 后期处理评估：
   - 可能的后期处理技术
   - 色彩调整和滤镜效果
   - 锐化、降噪等技术处理
   - 特效和艺术效果
   - 整体后期风格和水准

4. 质量和技术评估：
   - 图片分辨率和清晰度
   - 色彩还原和准确性
   - 噪点控制和画质表现
   - 技术缺陷和改进空间
   - 整体技术水准评价

5. 创作意图分析：
   - 摄影师的创作意图
   - 技术选择的艺术考量
   - 表现手法的创新性
   - 技术与艺术的结合度

请提供专业的技术和艺术分析。'''
            },
            {
                'name': 'semantic_tags',
                'display_name': '语义标签生成',
                'description': '多层次语义标签和关键词',
                'weight': 1.1,
                'extraction_prompt': '''请为图片生成丰富的语义标签和关键词：

1. 主题标签：
   - 图片的主要主题和内容类别
   - 核心概念和中心思想
   - 主题的层次和深度
   - 相关的主题扩展

2. 对象标签：
   - 图片中的具体对象和实体
   - 对象的属性和特征标签
   - 对象的功能和用途标签
   - 对象的品牌和型号（如可识别）

3. 属性标签：
   - 颜色、大小、形状、材质等物理属性
   - 风格、类型、等级等抽象属性
   - 时间、地点、场合等情境属性
   - 情感、氛围、意境等感受属性

4. 关联标签：
   - 相关的概念、场景、活动标签
   - 文化、历史、社会相关标签
   - 行业、专业、技术相关标签
   - 情感、心理、精神相关标签

5. 检索关键词：
   - 适合搜索的关键词组合
   - 不同语言的关键词
   - 同义词和近义词
   - 专业术语和俗语表达

请生成尽可能多的相关标签和关键词，用逗号分隔。'''
            }
        ]
    
    def extract_all_dimensions(self, image_path: str) -> Dict[str, Any]:
        """提取所有维度的信息"""
        logger.info(f"开始多维度信息提取: {image_path}")
        
        results = {
            'image_path': image_path,
            'extraction_time': datetime.now().isoformat(),
            'dimensions': {}
        }
        
        total_dimensions = len(self.dimensions)
        
        for i, dimension in enumerate(self.dimensions, 1):
            logger.info(f"提取维度 {i}/{total_dimensions}: {dimension['display_name']}")
            
            try:
                content = self.extract_dimension(image_path, dimension)
                results['dimensions'][dimension['name']] = {
                    'display_name': dimension['display_name'],
                    'description': dimension['description'],
                    'weight': dimension['weight'],
                    'content': content,
                    'content_length': len(content) if isinstance(content, str) else 0,
                    'extraction_time': datetime.now().isoformat()
                }
                logger.info(f"维度 {dimension['display_name']} 提取完成，内容长度: {len(content) if isinstance(content, str) else 0}")
                
            except Exception as e:
                logger.error(f"维度 {dimension['display_name']} 提取失败: {str(e)}")
                results['dimensions'][dimension['name']] = {
                    'display_name': dimension['display_name'],
                    'description': dimension['description'],
                    'weight': dimension['weight'],
                    'content': f"提取失败: {str(e)}",
                    'content_length': 0,
                    'extraction_time': datetime.now().isoformat(),
                    'error': str(e)
                }
        
        # 计算总体统计信息
        total_content_length = sum(
            dim.get('content_length', 0) 
            for dim in results['dimensions'].values()
        )
        successful_dimensions = sum(
            1 for dim in results['dimensions'].values() 
            if 'error' not in dim
        )
        
        results['statistics'] = {
            'total_dimensions': total_dimensions,
            'successful_dimensions': successful_dimensions,
            'total_content_length': total_content_length,
            'average_content_length': total_content_length / max(successful_dimensions, 1),
            'success_rate': successful_dimensions / total_dimensions
        }
        
        logger.info(f"多维度信息提取完成: {successful_dimensions}/{total_dimensions} 成功")
        logger.info(f"总内容长度: {total_content_length} 字符")
        
        return results
    
    def extract_dimension(self, image_path: str, dimension: Dict[str, Any]) -> str:
        """提取单个维度的信息"""
        prompt = dimension['extraction_prompt']
        
        # 调用qwen-vl-plus模型
        result = self.qwen_model.describe_image(image_path, prompt)
        
        # 处理返回结果
        if isinstance(result, list) and len(result) > 0:
            content = result[0].get('text', '')
        elif isinstance(result, str):
            content = result
        else:
            content = str(result)
        
        return content
    
    def get_dimension_summary(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """获取维度提取摘要"""
        if 'dimensions' not in extraction_results:
            return {}
        
        summary = {}
        for dim_name, dim_data in extraction_results['dimensions'].items():
            content = dim_data.get('content', '')
            summary[dim_name] = {
                'display_name': dim_data.get('display_name', dim_name),
                'content_preview': content[:200] + '...' if len(content) > 200 else content,
                'content_length': dim_data.get('content_length', 0),
                'weight': dim_data.get('weight', 1.0),
                'has_error': 'error' in dim_data
            }
        
        return summary
    
    def format_for_display(self, extraction_results: Dict[str, Any]) -> str:
        """格式化结果用于显示"""
        if 'dimensions' not in extraction_results:
            return "提取结果为空"
        
        formatted_text = []
        formatted_text.append("=== 多维度图像信息提取结果 ===\n")
        
        # 添加统计信息
        if 'statistics' in extraction_results:
            stats = extraction_results['statistics']
            formatted_text.append(f"📊 提取统计:")
            formatted_text.append(f"- 成功维度: {stats['successful_dimensions']}/{stats['total_dimensions']}")
            formatted_text.append(f"- 总内容长度: {stats['total_content_length']} 字符")
            formatted_text.append(f"- 平均内容长度: {stats['average_content_length']:.0f} 字符")
            formatted_text.append(f"- 成功率: {stats['success_rate']:.1%}\n")
        
        # 添加各维度内容
        for dim_name, dim_data in extraction_results['dimensions'].items():
            if 'error' in dim_data:
                continue
                
            formatted_text.append(f"## {dim_data['display_name']}")
            formatted_text.append(f"**权重**: {dim_data['weight']} | **长度**: {dim_data['content_length']} 字符")
            formatted_text.append(f"{dim_data['content']}\n")
        
        return '\n'.join(formatted_text)
    
    def get_combined_description(self, extraction_results: Dict[str, Any]) -> str:
        """获取组合描述用于向量化"""
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
                # 添加维度标识
                combined_parts.append(f"[{dim_data['display_name']}] {content}")
        
        return '\n\n'.join(combined_parts)


if __name__ == "__main__":
    # 测试多维度提取器
    extractor = MultiDimensionalExtractor()
    
    # 测试图片路径
    test_image = "static/test_images/cat1.jpg"
    
    print("开始多维度信息提取测试...")
    results = extractor.extract_all_dimensions(test_image)
    
    print("\n=== 提取结果摘要 ===")
    summary = extractor.get_dimension_summary(results)
    for dim_name, dim_summary in summary.items():
        print(f"{dim_summary['display_name']}: {dim_summary['content_length']} 字符")
    
    print("\n=== 格式化显示 ===")
    formatted = extractor.format_for_display(results)
    print(formatted[:1000] + "..." if len(formatted) > 1000 else formatted)


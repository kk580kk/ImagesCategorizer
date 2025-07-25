"""
智能标签生成器
基于qwen-vl-plus多角度分析生成详细的图像标签和描述
"""
import logging
from typing import List, Dict, Any
from .qwen_vl_model import QwenVLModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartTagGenerator:
    """智能标签生成器类"""
    
    def __init__(self):
        """初始化智能标签生成器"""
        self.qwen_model = QwenVLModel()
        logger.info("智能标签生成器初始化完成")
    
    def generate_comprehensive_tags(self, image_path):
        """
        生成全面的图像标签和描述
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 包含多维度标签和描述的字典
        """
        try:
            logger.info(f"开始生成图像标签: {image_path}")
            
            # 进行多角度分析
            analysis = self.qwen_model.multi_angle_analysis(image_path)
            if not analysis:
                logger.error("多角度分析失败")
                return None
            
            # 基于分析结果生成标签
            tags_result = {
                'primary_tags': self._extract_primary_tags(analysis),
                'detailed_description': self._generate_detailed_description(analysis),
                'semantic_tags': self._extract_semantic_tags(analysis),
                'visual_attributes': self._extract_visual_attributes(analysis),
                'context_tags': self._extract_context_tags(analysis),
                'emotion_tags': self._extract_emotion_tags(analysis),
                'technical_tags': self._extract_technical_tags(analysis),
                'comprehensive_summary': self._generate_comprehensive_summary(analysis)
            }
            
            logger.info(f"标签生成完成: {len(tags_result['primary_tags'])}个主要标签")
            return tags_result
            
        except Exception as e:
            logger.error(f"标签生成失败: {e}")
            return None
    
    def _extract_primary_tags(self, analysis):
        """提取主要标签"""
        try:
            primary_tags = []
            
            # 从基础内容中提取主要对象
            basic_content = analysis.get('basic_content', '').lower()
            person_features = analysis.get('person_features', '').lower()
            activity_features = analysis.get('activity_features', '').lower()
            
            # 人物相关标签
            if '人物' in person_features or '人' in basic_content:
                primary_tags.append('人物')
                if '男' in person_features:
                    primary_tags.append('男性')
                if '女' in person_features:
                    primary_tags.append('女性')
                if '儿童' in person_features or '小孩' in person_features:
                    primary_tags.append('儿童')
                if '老人' in person_features:
                    primary_tags.append('老人')
            
            # 动物相关标签
            animal_keywords = ['猫', '狗', '鸟', '鱼', '马', '牛', '羊', '兔', '熊猫', '老虎', '狮子']
            for keyword in animal_keywords:
                if keyword in basic_content:
                    primary_tags.extend(['动物', keyword])
            
            # 建筑相关标签
            building_keywords = ['建筑', '房屋', '大楼', '桥梁', '塔', '城堡', '教堂']
            for keyword in building_keywords:
                if keyword in basic_content:
                    primary_tags.extend(['建筑', keyword])
            
            # 交通工具相关标签
            vehicle_keywords = ['汽车', '火车', '飞机', '船', '自行车', '摩托车', '公交车']
            for keyword in vehicle_keywords:
                if keyword in basic_content:
                    primary_tags.extend(['交通工具', keyword])
            
            # 运动相关标签
            sport_keywords = ['运动', '体育', '健身', '篮球', '足球', '网球', '游泳', '跑步']
            for keyword in sport_keywords:
                if keyword in activity_features:
                    primary_tags.extend(['运动', keyword])
            
            return list(set(primary_tags))  # 去重
            
        except Exception as e:
            logger.error(f"主要标签提取失败: {e}")
            return []
    
    def _generate_detailed_description(self, analysis):
        """生成详细描述"""
        try:
            description_parts = []
            
            # 基础内容描述
            if analysis.get('basic_content'):
                description_parts.append(f"基础内容：{analysis['basic_content']}")
            
            # 人物特征描述
            if analysis.get('person_features') and '无人物' not in analysis['person_features']:
                description_parts.append(f"人物特征：{analysis['person_features']}")
            
            # 活动特征描述
            if analysis.get('activity_features'):
                description_parts.append(f"活动特征：{analysis['activity_features']}")
            
            # 场景特征描述
            if analysis.get('scene_features'):
                description_parts.append(f"场景特征：{analysis['scene_features']}")
            
            # 视觉特征描述
            if analysis.get('visual_features'):
                description_parts.append(f"视觉特征：{analysis['visual_features']}")
            
            return " | ".join(description_parts)
            
        except Exception as e:
            logger.error(f"详细描述生成失败: {e}")
            return "描述生成失败"
    
    def _extract_semantic_tags(self, analysis):
        """提取语义标签"""
        try:
            semantic_tags = []
            
            # 合并所有分析文本
            all_text = " ".join([
                analysis.get('basic_content', ''),
                analysis.get('person_features', ''),
                analysis.get('activity_features', ''),
                analysis.get('scene_features', ''),
                analysis.get('visual_features', '')
            ]).lower()
            
            # 语义关键词映射
            semantic_mapping = {
                '室内': ['室内', '房间', '屋内'],
                '室外': ['室外', '户外', '外面'],
                '自然': ['自然', '天然', '野生'],
                '人工': ['人工', '人造', '制造'],
                '现代': ['现代', '当代', '新式'],
                '传统': ['传统', '古典', '老式'],
                '正式': ['正式', '庄重', '严肃'],
                '休闲': ['休闲', '轻松', '随意'],
                '动态': ['动态', '运动', '活跃'],
                '静态': ['静态', '静止', '安静'],
                '明亮': ['明亮', '光亮', '清晰'],
                '昏暗': ['昏暗', '暗淡', '模糊']
            }
            
            for tag, keywords in semantic_mapping.items():
                if any(keyword in all_text for keyword in keywords):
                    semantic_tags.append(tag)
            
            return semantic_tags
            
        except Exception as e:
            logger.error(f"语义标签提取失败: {e}")
            return []
    
    def _extract_visual_attributes(self, analysis):
        """提取视觉属性"""
        try:
            visual_attrs = {}
            
            visual_features = analysis.get('visual_features', '').lower()
            
            # 颜色属性
            color_keywords = {
                '红色': ['红', '红色'],
                '蓝色': ['蓝', '蓝色'],
                '绿色': ['绿', '绿色'],
                '黄色': ['黄', '黄色'],
                '黑色': ['黑', '黑色'],
                '白色': ['白', '白色'],
                '彩色': ['彩色', '多彩'],
                '单色': ['单色', '黑白']
            }
            
            colors = []
            for color, keywords in color_keywords.items():
                if any(keyword in visual_features for keyword in keywords):
                    colors.append(color)
            visual_attrs['colors'] = colors
            
            # 构图属性
            composition_keywords = {
                '特写': ['特写', '近景'],
                '全景': ['全景', '远景'],
                '中景': ['中景', '中等'],
                '俯视': ['俯视', '从上'],
                '仰视': ['仰视', '从下'],
                '侧面': ['侧面', '侧视']
            }
            
            composition = []
            for comp, keywords in composition_keywords.items():
                if any(keyword in visual_features for keyword in keywords):
                    composition.append(comp)
            visual_attrs['composition'] = composition
            
            # 光线属性
            lighting_keywords = {
                '自然光': ['自然光', '阳光'],
                '人工光': ['人工光', '灯光'],
                '强光': ['强光', '明亮'],
                '柔光': ['柔光', '温和']
            }
            
            lighting = []
            for light, keywords in lighting_keywords.items():
                if any(keyword in visual_features for keyword in keywords):
                    lighting.append(light)
            visual_attrs['lighting'] = lighting
            
            return visual_attrs
            
        except Exception as e:
            logger.error(f"视觉属性提取失败: {e}")
            return {}
    
    def _extract_context_tags(self, analysis):
        """提取上下文标签"""
        try:
            context_tags = []
            
            scene_features = analysis.get('scene_features', '').lower()
            
            # 场所类型
            location_keywords = {
                '家庭': ['家', '家庭', '住宅'],
                '办公': ['办公', '工作', '公司'],
                '学校': ['学校', '教室', '校园'],
                '医院': ['医院', '诊所', '医疗'],
                '商店': ['商店', '商场', '购物'],
                '餐厅': ['餐厅', '饭店', '用餐'],
                '公园': ['公园', '花园', '绿地'],
                '街道': ['街道', '马路', '道路'],
                '海滩': ['海滩', '沙滩', '海边'],
                '山区': ['山', '山区', '山峰']
            }
            
            for location, keywords in location_keywords.items():
                if any(keyword in scene_features for keyword in keywords):
                    context_tags.append(location)
            
            return context_tags
            
        except Exception as e:
            logger.error(f"上下文标签提取失败: {e}")
            return []
    
    def _extract_emotion_tags(self, analysis):
        """提取情感标签"""
        try:
            emotion_tags = []
            
            # 合并所有分析文本
            all_text = " ".join([
                analysis.get('basic_content', ''),
                analysis.get('person_features', ''),
                analysis.get('visual_features', '')
            ]).lower()
            
            # 情感关键词
            emotion_keywords = {
                '快乐': ['快乐', '开心', '愉快', '微笑', '笑容'],
                '平静': ['平静', '安静', '宁静', '祥和'],
                '活跃': ['活跃', '活泼', '生动', '充满活力'],
                '严肃': ['严肃', '庄重', '正式', '端庄'],
                '温馨': ['温馨', '温暖', '舒适', '和谐'],
                '神秘': ['神秘', '朦胧', '模糊', '隐秘'],
                '壮观': ['壮观', '宏伟', '震撼', '雄伟']
            }
            
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in all_text for keyword in keywords):
                    emotion_tags.append(emotion)
            
            return emotion_tags
            
        except Exception as e:
            logger.error(f"情感标签提取失败: {e}")
            return []
    
    def _extract_technical_tags(self, analysis):
        """提取技术标签"""
        try:
            technical_tags = []
            
            visual_features = analysis.get('visual_features', '').lower()
            
            # 技术特征关键词
            technical_keywords = {
                '高清': ['高清', '清晰', '锐利'],
                '模糊': ['模糊', '不清', '朦胧'],
                '专业': ['专业', '精美', '高质量'],
                '业余': ['业余', '普通', '简单'],
                '艺术': ['艺术', '创意', '美学'],
                '写实': ['写实', '真实', '自然']
            }
            
            for tech, keywords in technical_keywords.items():
                if any(keyword in visual_features for keyword in keywords):
                    technical_tags.append(tech)
            
            return technical_tags
            
        except Exception as e:
            logger.error(f"技术标签提取失败: {e}")
            return []
    
    def _generate_comprehensive_summary(self, analysis):
        """生成综合摘要"""
        try:
            # 提取关键信息
            basic = analysis.get('basic_content', '')
            person = analysis.get('person_features', '')
            activity = analysis.get('activity_features', '')
            scene = analysis.get('scene_features', '')
            visual = analysis.get('visual_features', '')
            
            # 生成摘要
            summary_parts = []
            
            # 主要内容
            if basic:
                main_objects = self._extract_main_objects(basic)
                if main_objects:
                    summary_parts.append(f"主要内容：{', '.join(main_objects)}")
            
            # 人物信息
            if person and '无人物' not in person:
                person_info = self._extract_person_info(person)
                if person_info:
                    summary_parts.append(f"人物：{person_info}")
            
            # 活动信息
            if activity:
                activity_info = self._extract_activity_info(activity)
                if activity_info:
                    summary_parts.append(f"活动：{activity_info}")
            
            # 场景信息
            if scene:
                scene_info = self._extract_scene_info(scene)
                if scene_info:
                    summary_parts.append(f"场景：{scene_info}")
            
            return " | ".join(summary_parts) if summary_parts else "图像内容分析"
            
        except Exception as e:
            logger.error(f"综合摘要生成失败: {e}")
            return "摘要生成失败"
    
    def _extract_main_objects(self, text):
        """从文本中提取主要对象"""
        objects = []
        object_keywords = ['人物', '动物', '建筑', '车辆', '植物', '食物', '器具', '设备']
        
        for keyword in object_keywords:
            if keyword in text:
                objects.append(keyword)
        
        return objects[:3]  # 最多返回3个主要对象
    
    def _extract_person_info(self, text):
        """从文本中提取人物信息"""
        person_info = []
        
        if '男' in text:
            person_info.append('男性')
        if '女' in text:
            person_info.append('女性')
        if '儿童' in text or '小孩' in text:
            person_info.append('儿童')
        if '老人' in text:
            person_info.append('老人')
        
        return ', '.join(person_info) if person_info else '人物'
    
    def _extract_activity_info(self, text):
        """从文本中提取活动信息"""
        activities = []
        activity_keywords = ['运动', '工作', '学习', '休息', '娱乐', '用餐', '购物', '旅行']
        
        for keyword in activity_keywords:
            if keyword in text:
                activities.append(keyword)
        
        return ', '.join(activities[:2]) if activities else None
    
    def _extract_scene_info(self, text):
        """从文本中提取场景信息"""
        if '室内' in text:
            return '室内场景'
        elif '室外' in text:
            return '室外场景'
        else:
            return None


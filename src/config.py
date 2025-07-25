"""
配置文件 - 存储API密钥和数据库连接信息
"""
import os

# 阿里云API配置
DASHSCOPE_API_KEY = "sk-71f2950a3d704e568ea7ab8ee0567447"

# Zilliz向量数据库配置
ZILLIZ_CLUSTER_ID = "in05-8b029938b95f2b9"
ZILLIZ_ENDPOINT = "https://in05-8b029938b95f2b9.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
ZILLIZ_TOKEN = "77cb0581cc572d4ae6ece28240a428760c4039d53aef0205ac63cdd6422f1269f9d35832f8b327822f84a71c24ec800c6ee27a85"

# Flask应用配置
FLASK_SECRET_KEY = "your-secret-key-here"
UPLOAD_FOLDER = "static/uploads"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# 模型配置
QWEN_VL_MODEL = "qwen-vl-plus"
EMBEDDING_DIMENSION = 1024  # 根据实际模型调整

# 零样本分类标签
ZERO_SHOT_LABELS = [
    "动物", "植物", "建筑", "交通工具", "食物", 
    "人物", "风景", "科技产品", "艺术品", "运动"
]

# 检索配置
TOP_K_RESULTS = 9  # 返回最相似的9个结果
SIMILARITY_THRESHOLD = 0.5  # 相似度阈值


# 图文检索系统 (ImagesCategorizer)

基于AI的智能图文检索与分类系统，支持多模态向量化和零样本分类。

## 🚀 功能特性

### 核心功能
- **图文检索**: 通过文本描述搜索相似图像
- **批量上传**: 支持多文件拖拽上传
- **智能分类**: 基于qwen-vl-plus的零样本分类
- **可视化展示**: 搜索结果和统计信息的图表化展示

### 技术架构
- **图像向量化**: multimodal-embedding-v1 (1024维)
- **文本向量化**: text-embedding-v4 (1024维)
- **图像理解**: qwen-vl-plus
- **向量存储**: 简单向量数据库
- **Web框架**: Flask + 响应式前端

## 🛠️ 安装部署

### 环境要求
- Python 3.11+
- Flask
- dashscope
- numpy
- scikit-learn
- PIL

### 快速启动
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置API密钥
# 编辑 src/config.py，设置 DASHSCOPE_API_KEY

# 3. 启动应用
cd src
python main.py
```

### 使用启动脚本
```bash
chmod +x start.sh
./start.sh
```

## 📖 使用说明

### 1. 图像上传
- 支持PNG、JPG、JPEG、GIF、BMP格式
- 可批量上传多个文件
- 自动生成向量和分类标签

### 2. 文本检索
- 输入描述性文本
- 返回最相似的9张图像
- 显示相似度评分

### 3. 图像分类
- 基于qwen-vl-plus的智能分析
- 支持多维度标签生成
- 零样本分类能力

### 4. 统计信息
- 数据库状态监控
- 分类统计图表
- 实时数据更新

## 🔧 配置说明

### API配置
在 `src/config.py` 中配置：
```python
DASHSCOPE_API_KEY = "your-api-key-here"
TOP_K_RESULTS = 9
```

### 模型配置
- **multimodal-embedding-v1**: 图像多模态向量化
- **text-embedding-v4**: 高质量文本向量化
- **qwen-vl-plus**: 图像理解和分析

## 📁 项目结构

```
image_retrieval_app/
├── src/
│   ├── main.py                 # Flask应用入口
│   ├── config.py              # 配置文件
│   ├── models/                # 核心模型
│   │   ├── hybrid_embedding_generator.py  # 混合向量生成器
│   │   ├── retrieval_engine.py           # 检索引擎
│   │   ├── zero_shot_classifier.py       # 零样本分类器
│   │   └── vector_database.py            # 向量数据库
│   ├── routes/                # 路由处理
│   │   └── retrieval.py       # 检索相关API
│   └── static/                # 静态文件
│       ├── index.html         # 前端页面
│       └── script.js          # 前端逻辑
├── requirements.txt           # 依赖列表
├── start.sh                  # 启动脚本
├── README.md                 # 项目文档
└── TROUBLESHOOTING.md        # 问题排查指南
```

## 🧪 测试

### 运行测试
```bash
# 测试混合向量生成器
python test_hybrid_system.py

# 测试高级模型
python test_advanced_models.py
```

### 功能验证
- ✅ multimodal-embedding-v1 图像向量化
- ✅ text-embedding-v4 文本向量化
- ✅ 混合检索架构
- ✅ Web界面交互

## 🔍 API接口

### 上传图像
```
POST /upload
Content-Type: multipart/form-data
```

### 文本检索
```
POST /search
Content-Type: application/json
{
  "query": "搜索文本"
}
```

### 获取统计
```
GET /stats
```

## 🎯 性能特点

- **高质量向量**: 1024维专业向量空间
- **快速检索**: 优化的相似度计算
- **智能分类**: 基于先进的视觉语言模型
- **响应式设计**: 支持桌面和移动设备

## 📝 更新日志

### v2.0 (当前版本)
- ✅ 集成multimodal-embedding-v1和text-embedding-v4
- ✅ 移除降级机制，使用纯净混合架构
- ✅ 优化API调用格式
- ✅ 改进Web界面和用户体验
- ✅ 增强错误处理和日志记录

### v1.0
- 基础图文检索功能
- 简单向量化和分类
- Web界面原型

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目采用MIT许可证。

## 🔗 相关链接

- [阿里云DashScope](https://dashscope.console.aliyun.com/)
- [qwen-vl-plus模型文档](https://help.aliyun.com/zh/dashscope/)

---

**基于AI的智能图文检索系统 - 让图像搜索更智能！** 🚀


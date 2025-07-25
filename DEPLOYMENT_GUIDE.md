# 部署指南

## 🚀 GitHub部署步骤

### 1. 推送到GitHub仓库

项目已经初始化了Git仓库并创建了初始提交。要推送到GitHub：

```bash
# 进入项目目录
cd image_retrieval_app

# 推送到GitHub (需要您的认证信息)
git push -u origin master
```

### 2. GitHub认证

推送时需要提供：
- **用户名**: 您的GitHub用户名
- **密码**: 建议使用Personal Access Token而不是密码

#### 创建Personal Access Token:
1. 访问 GitHub Settings > Developer settings > Personal access tokens
2. 点击 "Generate new token"
3. 选择权限: `repo` (完整仓库访问权限)
4. 复制生成的token作为密码使用

### 3. 替代方案 - 手动上传

如果Git推送有问题，可以：

1. 在GitHub上创建新仓库 `ImagesCategorizer`
2. 下载本项目的压缩包
3. 解压后上传到GitHub仓库

## 📦 项目结构

```
ImagesCategorizer/
├── .git/                     # Git仓库信息
├── .gitignore               # Git忽略文件
├── README.md                # 项目文档
├── requirements.txt         # Python依赖
├── DEPLOYMENT_GUIDE.md      # 部署指南
├── TROUBLESHOOTING.md       # 问题排查
├── start.sh                 # 启动脚本
├── test_hybrid_system.py    # 测试脚本
└── src/                     # 源代码
    ├── main.py              # Flask应用入口
    ├── config.py            # 配置文件
    ├── models/              # 核心模型
    ├── routes/              # API路由
    ├── static/              # 静态文件
    └── utils/               # 工具函数
```

## 🔧 本地开发

### 环境配置
```bash
# 1. 克隆仓库
git clone https://github.com/kk580kk/ImagesCategorizer.git
cd ImagesCategorizer

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置API密钥
# 编辑 src/config.py
DASHSCOPE_API_KEY = "your-api-key-here"

# 4. 启动应用
chmod +x start.sh
./start.sh
```

### 手动启动
```bash
cd src
python main.py
```

## 🌐 生产部署

### 使用Gunicorn (推荐)
```bash
pip install gunicorn
cd src
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

### 使用Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
EXPOSE 5000

CMD ["python", "src/main.py"]
```

## 📝 Git提交信息

当前提交包含：
- ✅ 完整的图文检索系统 v2.0
- ✅ 混合向量架构 (multimodal-embedding-v1 + text-embedding-v4)
- ✅ Web界面和API接口
- ✅ 测试脚本和文档
- ✅ 部署配置文件

## 🔗 相关链接

- **GitHub仓库**: https://github.com/kk580kk/ImagesCategorizer.git
- **在线演示**: https://5000-ipkpjn3nf6es0kmzeb8zs-ab9ff248.manusvm.computer
- **技术文档**: README.md
- **问题排查**: TROUBLESHOOTING.md

---

**准备就绪，可以推送到GitHub了！** 🚀


import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.models.user import db
from src.routes.user import user_bp
from src.routes.retrieval import retrieval_bp
from src.routes.dual_vector_routes import dual_vector_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# 启用CORS
CORS(app)

# 配置文件上传
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'data', 'images')

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(retrieval_bp, url_prefix='/api/retrieval')
app.register_blueprint(dual_vector_bp, url_prefix='/api/dual-vector')

# uncomment if you need to use database
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404

# 提供上传的图像文件
@app.route('/images/<filename>')
def uploaded_file(filename):
    """提供上传的图像文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 错误处理
@app.errorhandler(413)
def too_large(e):
    return {'success': False, 'error': '文件太大，最大支持16MB'}, 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

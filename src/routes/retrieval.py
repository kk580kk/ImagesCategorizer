"""
图文检索API路由
"""
import os
import json
import logging
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
from datetime import datetime

# 导入我们的模型
from models.retrieval_engine_enhanced import EnhancedRetrievalEngine
from src.utils.visualization import VisualizationTools

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建蓝图
retrieval_bp = Blueprint('retrieval', __name__)

# 全局变量
retrieval_engine = None
viz_tools = None

def init_retrieval_engine():
    """初始化检索引擎"""
    global retrieval_engine, viz_tools
    if retrieval_engine is None:
        try:
            # 创建数据目录
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # 初始化检索引擎
            db_path = os.path.join(data_dir, 'vector_db')
            retrieval_engine = EnhancedRetrievalEngine()
            
            # 初始化可视化工具
            viz_dir = os.path.join(data_dir, 'visualizations')
            viz_tools = VisualizationTools(viz_dir)
            
            logger.info("检索引擎初始化成功")
            return True
        except Exception as e:
            logger.error(f"检索引擎初始化失败: {e}")
            return False
    return True

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@retrieval_bp.route('/upload', methods=['POST'])
def upload_image():
    """上传图像到向量数据库"""
    try:
        if not init_retrieval_engine():
            return jsonify({'error': '检索引擎初始化失败'}), 500
        
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '没有文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if file and allowed_file(file.filename):
            # 生成安全的文件名
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            
            # 创建上传目录
            upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # 保存文件
            file_path = os.path.join(upload_dir, unique_filename)
            file.save(file_path)
            
            # 验证图像文件
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                os.remove(file_path)
                return jsonify({'error': f'无效的图像文件: {str(e)}'}), 400
            
            # 添加到检索引擎
            result = retrieval_engine.add_image_to_database(file_path)
            
            if result.get('success', False):
                # 保存数据库
                retrieval_engine.save_all_data()
                
                return jsonify({
                    'success': True,
                    'message': '图像上传成功',
                    'file_path': f'/static/uploads/{unique_filename}',
                    'vector_id': result.get('vector_id'),
                    'description': result.get('description'),
                    'category': result.get('category'),
                    'classification_result': result.get('classification_result')
                })
            else:
                # 删除文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': f'图像处理失败: {result.get("error", "未知错误")}'}), 500
        
        return jsonify({'error': '不支持的文件类型'}), 400
        
    except Exception as e:
        logger.error(f"图像上传失败: {e}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@retrieval_bp.route('/search', methods=['POST'])
def search_images():
    """根据文本搜索相似图像"""
    try:
        if not init_retrieval_engine():
            return jsonify({'error': '检索引擎初始化失败'}), 500
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': '缺少查询文本'}), 400
        
        query_text = data['query'].strip()
        if not query_text:
            return jsonify({'error': '查询文本不能为空'}), 400
        
        top_k = data.get('top_k', 9)
        
        # 执行搜索
        search_results = retrieval_engine.search_images_by_text(query_text, top_k)
        
        # 处理结果，确保图像路径正确
        processed_results = []
        for result in search_results:
            image_path = result['image_path']
            # 转换为相对于static目录的路径
            if 'static/uploads' in image_path:
                relative_path = '/' + image_path.split('static/')[-1]
            else:
                relative_path = image_path
            
            processed_result = {
                'image_path': relative_path,
                'description': result['description'],
                'category': result['category'],
                'similarity': result['similarity'],
                'similarity_percentage': result['similarity_percentage']
            }
            processed_results.append(processed_result)
        
        # 创建搜索结果可视化
        viz_path = None
        try:
            viz_path = viz_tools.create_search_results_visualization(
                search_results, query_text
            )
            if viz_path:
                # 转换为相对路径
                viz_relative_path = '/' + viz_path.split('static/')[-1] if 'static/' in viz_path else viz_path
            else:
                viz_relative_path = None
        except Exception as e:
            logger.warning(f"可视化创建失败: {e}")
            viz_relative_path = None
        
        return jsonify({
            'success': True,
            'query': query_text,
            'results': processed_results,
            'total_results': len(processed_results),
            'visualization': viz_relative_path
        })
        
    except Exception as e:
        logger.error(f"图像搜索失败: {e}")
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500

@retrieval_bp.route('/classify', methods=['POST'])
def classify_image():
    """对上传的图像进行分类"""
    try:
        if not init_retrieval_engine():
            return jsonify({'error': '检索引擎初始化失败'}), 500
        
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '没有文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if file and allowed_file(file.filename):
            # 生成临时文件名
            filename = secure_filename(file.filename)
            temp_filename = f"temp_{uuid.uuid4()}_{filename}"
            
            # 创建临时目录
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # 保存临时文件
            temp_path = os.path.join(temp_dir, temp_filename)
            file.save(temp_path)
            
            try:
                # 验证图像文件
                with Image.open(temp_path) as img:
                    img.verify()
                
                # 进行分类
                classification_result = retrieval_engine.classify_image(temp_path)
                
                if classification_result:
                    return jsonify({
                        'success': True,
                        'classification': classification_result
                    })
                else:
                    return jsonify({'error': '分类失败'}), 500
                    
            finally:
                # 删除临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return jsonify({'error': '不支持的文件类型'}), 400
        
    except Exception as e:
        logger.error(f"图像分类失败: {e}")
        return jsonify({'error': f'分类失败: {str(e)}'}), 500

@retrieval_bp.route('/stats', methods=['GET'])
def get_statistics():
    """获取数据库统计信息"""
    try:
        if not init_retrieval_engine():
            return jsonify({'error': '检索引擎初始化失败'}), 500
        
        stats = retrieval_engine.get_database_statistics()
        
        # 创建统计可视化
        viz_files = {}
        try:
            if 'classification' in stats:
                classification_viz = viz_tools.create_classification_statistics(
                    stats['classification']
                )
                if classification_viz:
                    viz_files['classification'] = '/' + classification_viz.split('static/')[-1] if 'static/' in classification_viz else classification_viz
        except Exception as e:
            logger.warning(f"统计可视化创建失败: {e}")
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'visualizations': viz_files
        })
        
    except Exception as e:
        logger.error(f"统计信息获取失败: {e}")
        return jsonify({'error': f'统计信息获取失败: {str(e)}'}), 500

@retrieval_bp.route('/clear', methods=['POST'])
def clear_database():
    """清空数据库"""
    try:
        if not init_retrieval_engine():
            return jsonify({'error': '检索引擎初始化失败'}), 500
        
        success = retrieval_engine.clear_all_data()
        
        if success:
            return jsonify({
                'success': True,
                'message': '数据库清空成功'
            })
        else:
            return jsonify({'error': '数据库清空失败'}), 500
        
    except Exception as e:
        logger.error(f"数据库清空失败: {e}")
        return jsonify({'error': f'数据库清空失败: {str(e)}'}), 500

@retrieval_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        engine_status = init_retrieval_engine()
        return jsonify({
            'status': 'healthy' if engine_status else 'unhealthy',
            'engine_initialized': engine_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向量系统Web路由
提供图像上传、处理、搜索等Web接口
"""

import os
import json
import time
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import logging

# 导入双向量处理器
from ..models.dual_vector_processor import DualVectorProcessor

# 配置日志
logger = logging.getLogger(__name__)

# 创建蓝图
dual_vector_bp = Blueprint('dual_vector', __name__)

# 全局处理器实例
processor = None

def init_processor():
    """初始化双向量处理器"""
    global processor
    if processor is None:
        try:
            api_key = "sk-71f2950a3d704e568ea7ab8ee0567447"  # 应该从环境变量获取
            processor = DualVectorProcessor(api_key)
            logger.info("双向量处理器初始化成功")
        except Exception as e:
            logger.error(f"双向量处理器初始化失败: {str(e)}")
            processor = None
    return processor

def allowed_file(filename):
    """检查文件类型是否允许"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@dual_vector_bp.route('/api/dual-vector/upload', methods=['POST'])
def upload_and_process_image():
    """上传并处理图像"""
    try:
        # 初始化处理器
        proc = init_processor()
        if not proc:
            return jsonify({
                'success': False,
                'error': '双向量处理器未初始化'
            }), 500
        
        # 检查文件
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有上传文件'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': '不支持的文件类型'
            }), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        
        upload_dir = os.path.join(current_app.root_path, 'data', 'images')
        os.makedirs(upload_dir, exist_ok=True)
        
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        
        # 处理图像
        logger.info(f"开始处理上传的图像: {filename}")
        result = proc.process_single_image(filepath)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': '图像上传和处理成功',
                'data': {
                    'image_id': result['image_id'],
                    'filename': filename,
                    'processing_time': result['processing_time'],
                    'text_vectors_count': result['text_vectors_inserted'],
                    'text_characters': result['total_text_characters'],
                    'analysis_dimensions': result['analysis_dimensions']
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': f"图像处理失败: {result.get('error', '未知错误')}"
            }), 500
            
    except Exception as e:
        logger.error(f"上传处理异常: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'服务器错误: {str(e)}'
        }), 500

@dual_vector_bp.route('/api/dual-vector/search/text', methods=['POST'])
def search_by_text():
    """基于文本搜索图像"""
    try:
        # 初始化处理器
        proc = init_processor()
        if not proc:
            return jsonify({
                'success': False,
                'error': '双向量处理器未初始化'
            }), 500
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': '缺少查询文本'
            }), 400
        
        query_text = data['query']
        top_k = data.get('top_k', 9)
        
        logger.info(f"文本搜索: '{query_text}', top_k: {top_k}")
        
        # 执行搜索
        results = proc.search_by_text(query_text, top_k)
        
        return jsonify({
            'success': True,
            'query': query_text,
            'results_count': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"文本搜索异常: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'搜索失败: {str(e)}'
        }), 500

@dual_vector_bp.route('/api/dual-vector/search/image', methods=['POST'])
def search_by_image():
    """基于图像搜索相似图像"""
    try:
        # 初始化处理器
        proc = init_processor()
        if not proc:
            return jsonify({
                'success': False,
                'error': '双向量处理器未初始化'
            }), 500
        
        # 检查文件
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有上传查询图像'
            }), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': '不支持的文件类型'
            }), 400
        
        top_k = int(request.form.get('top_k', 9))
        
        # 保存临时文件
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(current_app.root_path, 'data', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_filepath = os.path.join(temp_dir, f"query_{int(time.time())}_{filename}")
        file.save(temp_filepath)
        
        try:
            # 执行搜索
            results = proc.search_by_image(temp_filepath, top_k)
            
            return jsonify({
                'success': True,
                'query_image': filename,
                'results_count': len(results),
                'results': results
            })
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
        
    except Exception as e:
        logger.error(f"图像搜索异常: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'搜索失败: {str(e)}'
        }), 500

@dual_vector_bp.route('/api/dual-vector/search/hybrid', methods=['POST'])
def hybrid_search():
    """混合搜索（文本+图像）"""
    try:
        # 初始化处理器
        proc = init_processor()
        if not proc:
            return jsonify({
                'success': False,
                'error': '双向量处理器未初始化'
            }), 500
        
        # 获取文本查询
        query_text = request.form.get('query', '')
        if not query_text:
            return jsonify({
                'success': False,
                'error': '缺少查询文本'
            }), 400
        
        top_k = int(request.form.get('top_k', 9))
        text_weight = float(request.form.get('text_weight', 0.7))
        
        # 处理可选的查询图像
        query_image_path = None
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_dir = os.path.join(current_app.root_path, 'data', 'temp')
                os.makedirs(temp_dir, exist_ok=True)
                
                query_image_path = os.path.join(temp_dir, f"hybrid_query_{int(time.time())}_{filename}")
                file.save(query_image_path)
        
        try:
            # 执行混合搜索
            results = proc.hybrid_search(
                query_text=query_text,
                query_image_path=query_image_path,
                top_k=top_k,
                text_weight=text_weight
            )
            
            return jsonify({
                'success': True,
                'query_text': query_text,
                'has_query_image': query_image_path is not None,
                'text_weight': text_weight,
                'results_count': len(results),
                'results': results
            })
            
        finally:
            # 清理临时文件
            if query_image_path and os.path.exists(query_image_path):
                os.remove(query_image_path)
        
    except Exception as e:
        logger.error(f"混合搜索异常: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'搜索失败: {str(e)}'
        }), 500

@dual_vector_bp.route('/api/dual-vector/image/<image_id>', methods=['GET'])
def get_image_details(image_id):
    """获取图像详细信息"""
    try:
        # 初始化处理器
        proc = init_processor()
        if not proc:
            return jsonify({
                'success': False,
                'error': '双向量处理器未初始化'
            }), 500
        
        # 获取图像详情
        details = proc.get_image_details(image_id)
        
        if details:
            return jsonify({
                'success': True,
                'image_id': image_id,
                'details': details
            })
        else:
            return jsonify({
                'success': False,
                'error': '图像不存在'
            }), 404
            
    except Exception as e:
        logger.error(f"获取图像详情异常: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取失败: {str(e)}'
        }), 500

@dual_vector_bp.route('/api/dual-vector/stats', methods=['GET'])
def get_database_stats():
    """获取数据库统计信息"""
    try:
        # 初始化处理器
        proc = init_processor()
        if not proc:
            return jsonify({
                'success': False,
                'error': '双向量处理器未初始化'
            }), 500
        
        # 获取统计信息
        stats = proc.get_database_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"获取统计信息异常: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取失败: {str(e)}'
        }), 500

@dual_vector_bp.route('/api/dual-vector/batch-process', methods=['POST'])
def batch_process_images():
    """批量处理图像"""
    try:
        # 初始化处理器
        proc = init_processor()
        if not proc:
            return jsonify({
                'success': False,
                'error': '双向量处理器未初始化'
            }), 500
        
        # 获取images目录下的所有图像
        images_dir = os.path.join(current_app.root_path, 'data', 'images')
        if not os.path.exists(images_dir):
            return jsonify({
                'success': False,
                'error': '图像目录不存在'
            }), 400
        
        # 获取图像文件列表
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if not image_files:
            return jsonify({
                'success': False,
                'error': '没有找到图像文件'
            }), 400
        
        # 构建完整路径
        image_paths = [os.path.join(images_dir, f) for f in image_files]
        
        # 获取批处理参数
        data = request.get_json() or {}
        batch_size = data.get('batch_size', 5)
        max_images = data.get('max_images', 10)  # 限制处理数量
        
        # 限制处理的图像数量
        if len(image_paths) > max_images:
            image_paths = image_paths[:max_images]
        
        logger.info(f"开始批量处理 {len(image_paths)} 张图像")
        
        # 执行批量处理
        result = proc.process_batch_images(image_paths, batch_size)
        
        return jsonify({
            'success': True,
            'message': '批量处理完成',
            'result': result
        })
        
    except Exception as e:
        logger.error(f"批量处理异常: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'批量处理失败: {str(e)}'
        }), 500

@dual_vector_bp.route('/api/dual-vector/clear', methods=['POST'])
def clear_database():
    """清空数据库"""
    try:
        # 初始化处理器
        proc = init_processor()
        if not proc:
            return jsonify({
                'success': False,
                'error': '双向量处理器未初始化'
            }), 500
        
        # 清空数据库
        success = proc.clear_database()
        
        if success:
            return jsonify({
                'success': True,
                'message': '数据库清空成功'
            })
        else:
            return jsonify({
                'success': False,
                'error': '数据库清空失败'
            }), 500
            
    except Exception as e:
        logger.error(f"清空数据库异常: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'清空失败: {str(e)}'
        }), 500

# 应用关闭时清理资源
@dual_vector_bp.teardown_app_request
def cleanup_processor(exception):
    """清理处理器资源"""
    global processor
    if processor:
        try:
            processor.close()
            processor = None
        except Exception as e:
            logger.error(f"清理处理器失败: {str(e)}")


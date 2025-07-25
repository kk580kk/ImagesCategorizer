"""
可视化工具模块
用于生成相似度矩阵、检索结果展示等可视化内容
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import logging
import json
from datetime import datetime

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationTools:
    """可视化工具类"""
    
    def __init__(self, output_dir="visualizations"):
        """
        初始化可视化工具
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"可视化工具初始化完成，输出目录: {output_dir}")
    
    def create_similarity_matrix(self, similarities, labels=None, title="相似度矩阵", save_path=None):
        """
        创建相似度矩阵可视化
        
        Args:
            similarities: 相似度矩阵 (numpy array)
            labels: 标签列表
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        try:
            # 确保相似度矩阵是numpy数组
            sim_matrix = np.array(similarities)
            
            # 创建图表
            plt.figure(figsize=(10, 8))
            
            # 使用seaborn创建热力图
            sns.heatmap(
                sim_matrix,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                square=True,
                xticklabels=labels if labels else False,
                yticklabels=labels if labels else False,
                cbar_kws={'label': '相似度'}
            )
            
            plt.title(title, fontsize=16, pad=20)
            plt.xlabel('图像索引', fontsize=12)
            plt.ylabel('图像索引', fontsize=12)
            plt.tight_layout()
            
            # 保存图表
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"similarity_matrix_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"相似度矩阵保存成功: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"相似度矩阵创建失败: {e}")
            return None
    
    def create_search_results_visualization(self, search_results, query_text, save_path=None):
        """
        创建搜索结果可视化
        
        Args:
            search_results: 搜索结果列表
            query_text: 查询文本
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        try:
            if not search_results:
                logger.warning("没有搜索结果可视化")
                return None
            
            # 提取相似度数据
            similarities = [result['similarity'] for result in search_results]
            categories = [result.get('category', 'unknown') for result in search_results]
            descriptions = [result.get('description', '')[:30] + '...' if len(result.get('description', '')) > 30 
                          else result.get('description', '') for result in search_results]
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 相似度条形图
            bars = ax1.barh(range(len(similarities)), similarities, color='skyblue')
            ax1.set_yticks(range(len(similarities)))
            ax1.set_yticklabels([f"结果 {i+1}" for i in range(len(similarities))])
            ax1.set_xlabel('相似度')
            ax1.set_title(f'搜索结果相似度\n查询: "{query_text}"')
            ax1.set_xlim(0, 1)
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            # 类别分布饼图
            category_counts = {}
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            if category_counts:
                ax2.pie(category_counts.values(), labels=category_counts.keys(), 
                       autopct='%1.1f%%', startangle=90)
                ax2.set_title('结果类别分布')
            
            plt.tight_layout()
            
            # 保存图表
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_query = "".join(c for c in query_text if c.isalnum() or c in (' ', '-', '_'))[:20]
                save_path = os.path.join(self.output_dir, f"search_results_{safe_query}_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"搜索结果可视化保存成功: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"搜索结果可视化创建失败: {e}")
            return None
    
    def create_classification_statistics(self, classification_stats, save_path=None):
        """
        创建分类统计可视化
        
        Args:
            classification_stats: 分类统计数据
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        try:
            categories = classification_stats.get('categories', {})
            if not categories:
                logger.warning("没有分类统计数据")
                return None
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 类别数量条形图
            category_names = list(categories.keys())
            category_counts = list(categories.values())
            
            bars = ax1.bar(category_names, category_counts, color='lightcoral')
            ax1.set_xlabel('类别')
            ax1.set_ylabel('数量')
            ax1.set_title('各类别图像数量统计')
            ax1.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            # 类别百分比饼图
            percentages = classification_stats.get('category_percentages', {})
            if percentages:
                ax2.pie(percentages.values(), labels=percentages.keys(), 
                       autopct='%1.1f%%', startangle=90)
                ax2.set_title('各类别占比分布')
            
            plt.tight_layout()
            
            # 保存图表
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"classification_stats_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"分类统计可视化保存成功: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"分类统计可视化创建失败: {e}")
            return None
    
    def create_accuracy_chart(self, accuracy_data, save_path=None):
        """
        创建准确率图表
        
        Args:
            accuracy_data: 准确率数据
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        try:
            if 'detailed_results' not in accuracy_data:
                logger.warning("没有详细的准确率数据")
                return None
            
            detailed_results = accuracy_data['detailed_results']
            
            # 按类别统计准确率
            category_accuracy = {}
            for result in detailed_results:
                true_label = result['true_label']
                is_correct = result['is_correct']
                
                if true_label not in category_accuracy:
                    category_accuracy[true_label] = {'correct': 0, 'total': 0}
                
                category_accuracy[true_label]['total'] += 1
                if is_correct:
                    category_accuracy[true_label]['correct'] += 1
            
            # 计算各类别准确率
            categories = []
            accuracies = []
            for category, stats in category_accuracy.items():
                categories.append(category)
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                accuracies.append(accuracy)
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            
            bars = plt.bar(categories, accuracies, color='lightgreen')
            plt.xlabel('类别')
            plt.ylabel('准确率')
            plt.title('各类别分类准确率')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            # 添加总体准确率线
            overall_accuracy = accuracy_data.get('accuracy', 0)
            plt.axhline(y=overall_accuracy, color='red', linestyle='--', 
                       label=f'总体准确率: {overall_accuracy:.3f}')
            plt.legend()
            
            plt.tight_layout()
            
            # 保存图表
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"accuracy_chart_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"准确率图表保存成功: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"准确率图表创建失败: {e}")
            return None
    
    def create_embedding_visualization(self, embeddings, labels=None, method='pca', save_path=None):
        """
        创建embedding向量可视化（降维）
        
        Args:
            embeddings: embedding向量列表
            labels: 标签列表
            method: 降维方法 ('pca' 或 'tsne')
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            
            # 确保embeddings是numpy数组
            embeddings_array = np.array(embeddings)
            
            if embeddings_array.shape[0] < 2:
                logger.warning("embedding数量太少，无法进行可视化")
                return None
            
            # 降维
            if method == 'pca':
                reducer = PCA(n_components=2)
                reduced_embeddings = reducer.fit_transform(embeddings_array)
                title = 'Embedding向量PCA可视化'
            elif method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42)
                reduced_embeddings = reducer.fit_transform(embeddings_array)
                title = 'Embedding向量t-SNE可视化'
            else:
                logger.error(f"不支持的降维方法: {method}")
                return None
            
            # 创建图表
            plt.figure(figsize=(10, 8))
            
            if labels:
                # 按标签着色
                unique_labels = list(set(labels))
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = np.array(labels) == label
                    plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                              c=[colors[i]], label=label, alpha=0.7)
                
                plt.legend()
            else:
                plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
            
            plt.xlabel('第一主成分' if method == 'pca' else 't-SNE维度1')
            plt.ylabel('第二主成分' if method == 'pca' else 't-SNE维度2')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"embedding_{method}_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Embedding可视化保存成功: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Embedding可视化创建失败: {e}")
            return None
    
    def create_comprehensive_report(self, engine_stats, search_results=None, accuracy_data=None):
        """
        创建综合报告
        
        Args:
            engine_stats: 引擎统计数据
            search_results: 搜索结果
            accuracy_data: 准确率数据
            
        Returns:
            dict: 生成的可视化文件路径
        """
        try:
            report_files = {}
            
            # 创建分类统计图表
            if 'classification' in engine_stats:
                classification_path = self.create_classification_statistics(engine_stats['classification'])
                if classification_path:
                    report_files['classification_stats'] = classification_path
            
            # 创建搜索结果可视化
            if search_results:
                search_path = self.create_search_results_visualization(
                    search_results, "综合搜索结果"
                )
                if search_path:
                    report_files['search_results'] = search_path
            
            # 创建准确率图表
            if accuracy_data:
                accuracy_path = self.create_accuracy_chart(accuracy_data)
                if accuracy_path:
                    report_files['accuracy_chart'] = accuracy_path
            
            logger.info(f"综合报告生成完成，包含{len(report_files)}个图表")
            return report_files
            
        except Exception as e:
            logger.error(f"综合报告生成失败: {e}")
            return {}
    
    def save_visualization_data(self, data, filename):
        """
        保存可视化数据到JSON文件
        
        Args:
            data: 要保存的数据
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        try:
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"可视化数据保存成功: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"可视化数据保存失败: {e}")
            return None


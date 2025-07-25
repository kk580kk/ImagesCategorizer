// 图文检索系统前端脚本

// API基础URL
const API_BASE = '/api/retrieval';

// 显示消息
function showMessage(message, type = 'info') {
    const alertClass = type === 'error' ? 'alert-danger' : 
                      type === 'success' ? 'alert-success' : 'alert-info';
    
    const alertHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 
                              type === 'success' ? 'check-circle' : 'info-circle'}"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // 在当前活动的tab中显示消息
    const activeTab = document.querySelector('.tab-pane.active');
    if (activeTab) {
        activeTab.insertAdjacentHTML('afterbegin', alertHtml);
    }
}

// 搜索图像
async function searchImages() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) {
        showMessage('请输入搜索文本', 'error');
        return;
    }
    
    const loadingElement = document.getElementById('searchLoading');
    const resultsElement = document.getElementById('searchResults');
    
    // 显示加载状态
    loadingElement.style.display = 'block';
    resultsElement.innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query, top_k: 9 })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displaySearchResults(data);
            showMessage(`找到 ${data.total_results} 个相似图像`, 'success');
        } else {
            showMessage(data.error || '搜索失败', 'error');
        }
    } catch (error) {
        showMessage(`搜索失败: ${error.message}`, 'error');
    } finally {
        loadingElement.style.display = 'none';
    }
}

// 显示搜索结果
function displaySearchResults(data) {
    const resultsElement = document.getElementById('searchResults');
    
    if (!data.results || data.results.length === 0) {
        resultsElement.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle"></i>
                没有找到相似的图像，请尝试其他搜索词。
            </div>
        `;
        return;
    }
    
    let html = `
        <div class="search-container">
            <h4><i class="fas fa-images"></i> 搜索结果 (${data.total_results})</h4>
            <p class="text-muted">查询: "${data.query}"</p>
        </div>
        <div class="row">
    `;
    
    data.results.forEach((result, index) => {
        html += `
            <div class="col-md-4 mb-4">
                <div class="result-card">
                    <img src="${result.image_path}" alt="搜索结果" class="result-image" 
                         onerror="this.src='/static/placeholder.png'">
                    <div class="result-info">
                        <div class="mb-2">
                            <span class="similarity-badge">${result.similarity_percentage}%</span>
                            <span class="category-badge">${result.category}</span>
                        </div>
                        <p class="mb-0 text-muted">${result.description}</p>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    
    // 如果有可视化图表，添加显示
    if (data.visualization) {
        html += `
            <div class="search-container mt-4">
                <h5><i class="fas fa-chart-bar"></i> 搜索结果分析</h5>
                <img src="${data.visualization}" alt="搜索结果可视化" class="img-fluid" 
                     style="max-height: 400px;">
            </div>
        `;
    }
    
    resultsElement.innerHTML = html;
}

// 上传图像
async function uploadImage() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showMessage('请选择图像文件', 'error');
        return;
    }
    
    const loadingElement = document.getElementById('uploadLoading');
    const resultElement = document.getElementById('uploadResult');
    
    // 显示加载状态
    loadingElement.style.display = 'block';
    resultElement.innerHTML = '';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayUploadResult(data);
            showMessage('图像上传成功！', 'success');
            // 清空文件输入
            fileInput.value = '';
        } else {
            showMessage(data.error || '上传失败', 'error');
        }
    } catch (error) {
        showMessage(`上传失败: ${error.message}`, 'error');
    } finally {
        loadingElement.style.display = 'none';
    }
}

// 显示上传结果
function displayUploadResult(data) {
    const resultElement = document.getElementById('uploadResult');
    
    const html = `
        <div class="search-container mt-4">
            <h4><i class="fas fa-check-circle text-success"></i> 上传成功</h4>
            <div class="row">
                <div class="col-md-6">
                    <img src="${data.file_path}" alt="上传的图像" class="img-fluid rounded">
                </div>
                <div class="col-md-6">
                    <h5>图像信息</h5>
                    <p><strong>向量ID:</strong> ${data.vector_id}</p>
                    <p><strong>分类:</strong> <span class="category-badge">${data.category}</span></p>
                    <p><strong>描述:</strong> ${data.description}</p>
                    ${data.classification_result ? `
                        <h6>分类详情</h6>
                        <p><strong>置信度:</strong> ${(data.classification_result.confidence * 100).toFixed(1)}%</p>
                        <p><strong>方法:</strong> ${data.classification_result.primary_method}</p>
                    ` : ''}
                </div>
            </div>
        </div>
    `;
    
    resultElement.innerHTML = html;
}

// 分类图像
async function classifyImage() {
    const fileInput = document.getElementById('classifyFileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showMessage('请选择图像文件', 'error');
        return;
    }
    
    const loadingElement = document.getElementById('classifyLoading');
    const resultElement = document.getElementById('classifyResult');
    
    // 显示加载状态
    loadingElement.style.display = 'block';
    resultElement.innerHTML = '';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE}/classify`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayClassifyResult(data, file);
            showMessage('图像分类完成！', 'success');
            // 清空文件输入
            fileInput.value = '';
        } else {
            showMessage(data.error || '分类失败', 'error');
        }
    } catch (error) {
        showMessage(`分类失败: ${error.message}`, 'error');
    } finally {
        loadingElement.style.display = 'none';
    }
}

// 显示分类结果
function displayClassifyResult(data, file) {
    const resultElement = document.getElementById('classifyResult');
    
    // 创建图像预览URL
    const imageUrl = URL.createObjectURL(file);
    
    const classification = data.classification;
    
    let html = `
        <div class="search-container mt-4">
            <h4><i class="fas fa-tags"></i> 分类结果</h4>
            <div class="row">
                <div class="col-md-6">
                    <img src="${imageUrl}" alt="分类图像" class="img-fluid rounded">
                </div>
                <div class="col-md-6">
                    <h5>分类信息</h5>
                    <p><strong>类别:</strong> <span class="category-badge">${classification.category}</span></p>
                    <p><strong>置信度:</strong> ${(classification.confidence * 100).toFixed(1)}%</p>
                    <p><strong>主要方法:</strong> ${classification.primary_method}</p>
                    <p><strong>方法一致性:</strong> ${classification.methods_agree ? '是' : '否'}</p>
                    
                    ${classification.direct_classification ? `
                        <h6>直接分类结果</h6>
                        <p><strong>类别:</strong> ${classification.direct_classification.category}</p>
                        <p><strong>置信度:</strong> ${(classification.direct_classification.confidence * 100).toFixed(1)}%</p>
                    ` : ''}
                    
                    ${classification.embedding_classification ? `
                        <h6>Embedding分类结果</h6>
                        <p><strong>类别:</strong> ${classification.embedding_classification.category}</p>
                        <p><strong>置信度:</strong> ${(classification.embedding_classification.confidence * 100).toFixed(1)}%</p>
                    ` : ''}
                </div>
            </div>
        </div>
    `;
    
    resultElement.innerHTML = html;
}

// 加载统计信息
async function loadStatistics() {
    const loadingElement = document.getElementById('statsLoading');
    const resultElement = document.getElementById('statsResult');
    
    // 显示加载状态
    loadingElement.style.display = 'block';
    resultElement.innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const data = await response.json();
        
        if (data.success) {
            displayStatistics(data);
            showMessage('统计信息加载成功', 'success');
        } else {
            showMessage(data.error || '加载统计信息失败', 'error');
        }
    } catch (error) {
        showMessage(`加载失败: ${error.message}`, 'error');
    } finally {
        loadingElement.style.display = 'none';
    }
}

// 显示统计信息
function displayStatistics(data) {
    const resultElement = document.getElementById('statsResult');
    
    const stats = data.statistics;
    
    let html = `
        <div class="row">
            <div class="col-md-6">
                <div class="search-container">
                    <h5><i class="fas fa-database"></i> 数据库统计</h5>
                    <p><strong>总向量数:</strong> ${stats.database.total_vectors}</p>
                    <p><strong>数据库路径:</strong> ${stats.database.database_path}</p>
                    
                    <h6>类型统计</h6>
                    <ul class="list-unstyled">
    `;
    
    for (const [type, count] of Object.entries(stats.database.type_counts || {})) {
        html += `<li><span class="badge bg-primary">${type}</span> ${count}</li>`;
    }
    
    html += `
                    </ul>
                    
                    <h6>类别统计</h6>
                    <ul class="list-unstyled">
    `;
    
    for (const [category, count] of Object.entries(stats.database.category_counts || {})) {
        html += `<li><span class="category-badge">${category}</span> ${count}</li>`;
    }
    
    html += `
                    </ul>
                </div>
            </div>
            <div class="col-md-6">
                <div class="search-container">
                    <h5><i class="fas fa-chart-pie"></i> 分类统计</h5>
                    <p><strong>总分类数:</strong> ${stats.classification.total}</p>
                    
                    <h6>各类别分类数量</h6>
                    <ul class="list-unstyled">
    `;
    
    for (const [category, count] of Object.entries(stats.classification.categories || {})) {
        const percentage = stats.classification.category_percentages[category] || 0;
        html += `
            <li class="mb-2">
                <span class="category-badge">${category}</span> 
                ${count} (${percentage.toFixed(1)}%)
                <div class="progress mt-1" style="height: 5px;">
                    <div class="progress-bar" style="width: ${percentage}%"></div>
                </div>
            </li>
        `;
    }
    
    html += `
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    // 如果有可视化图表，添加显示
    if (data.visualizations && data.visualizations.classification) {
        html += `
            <div class="search-container mt-4">
                <h5><i class="fas fa-chart-bar"></i> 分类统计图表</h5>
                <img src="${data.visualizations.classification}" alt="分类统计图表" class="img-fluid">
            </div>
        `;
    }
    
    resultElement.innerHTML = html;
}

// 清空数据库
async function clearDatabase() {
    if (!confirm('确定要清空数据库吗？此操作不可恢复！')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/clear`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showMessage('数据库清空成功', 'success');
            // 重新加载统计信息
            loadStatistics();
        } else {
            showMessage(data.error || '清空失败', 'error');
        }
    } catch (error) {
        showMessage(`清空失败: ${error.message}`, 'error');
    }
}

// 拖拽上传功能
function setupDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight(e) {
        uploadArea.classList.add('dragover');
    }
    
    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }
    
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            document.getElementById('fileInput').files = files;
            uploadImage();
        }
    }
}

// 搜索框回车事件
document.getElementById('searchInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        searchImages();
    }
});

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    setupDragAndDrop();
    
    // 检查健康状态
    fetch(`${API_BASE}/health`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'healthy') {
                console.log('检索引擎状态正常');
            } else {
                showMessage('检索引擎状态异常，部分功能可能不可用', 'error');
            }
        })
        .catch(error => {
            console.error('健康检查失败:', error);
        });
});


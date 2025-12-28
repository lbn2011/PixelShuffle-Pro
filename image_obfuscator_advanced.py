# image_obfuscator.py
import sys
import os
import time
import threading
import hashlib
from datetime import datetime
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QComboBox, QSlider, QSpinBox, QCheckBox, QGroupBox,
                            QProgressBar, QTableWidget, QTableWidgetItem, 
                            QMessageBox, QSplitter, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QDesktopServices
import numba
from numba import jit, prange
from cachetools import LRUCache, TTLCache
import json
import pickle

# ============= 数据结构定义 =============
@dataclass
class ImageInfo:
    """图片信息"""
    path: str
    width: int
    height: int
    size: int
    format: str
    hash: str  # 用于缓存键
    
@dataclass
class ProcessResult:
    """处理结果"""
    image_array: np.ndarray
    time_cost: float
    file_size: int

# ============= 缓存管理器 =============
class CurveCacheManager:
    """曲线映射缓存管理器"""
    
    def __init__(self, max_size=100, ttl=3600):
        # 缓存Gilbert曲线映射
        self.curve_cache = LRUCache(maxsize=max_size)
        # 缓存处理结果（基于图片哈希和参数）
        self.result_cache = TTLCache(maxsize=50, ttl=ttl)
        
    def get_curve_key(self, width: int, height: int) -> str:
        """生成曲线缓存键"""
        return f"{width}x{height}"
    
    def get_result_key(self, image_hash: str, mode: str, 
                      quality: int, format: str) -> str:
        """生成结果缓存键"""
        return f"{image_hash}_{mode}_{quality}_{format}"
    
    def save_curve(self, width: int, height: int, curve_map: np.ndarray):
        """保存曲线映射"""
        key = self.get_curve_key(width, height)
        self.curve_cache[key] = curve_map
        
    def get_curve(self, width: int, height: int) -> Optional[np.ndarray]:
        """获取曲线映射"""
        key = self.get_curve_key(width, height)
        return self.curve_cache.get(key)
    
    def save_result(self, key: str, result: ProcessResult):
        """保存处理结果"""
        self.result_cache[key] = result
        
    def get_result(self, key: str) -> Optional[ProcessResult]:
        """获取处理结果"""
        return self.result_cache.get(key)
    
    def clear(self):
        """清空缓存"""
        self.curve_cache.clear()
        self.result_cache.clear()

# ============= Numba加速算法 =============
@jit(nopython=True, nogil=True, cache=True, parallel=True)
def build_gilbert_curve_map_numba(width: int, height: int) -> np.ndarray:
    """使用Numba加速构建Gilbert曲线映射"""
    total_pixels = width * height
    curve_map = np.zeros(total_pixels, dtype=np.uint32)
    emit_index = 0
    
    # 迭代栈结构
    stack = np.zeros((total_pixels * 2, 6), dtype=np.int32)
    stack_size = 0
    
    # 初始参数入栈
    if width >= height:
        stack[stack_size] = [0, 0, width, 0, 0, height]
    else:
        stack[stack_size] = [0, 0, 0, height, width, 0]
    stack_size += 1
    
    while stack_size > 0:
        stack_size -= 1
        x0, y0, ax0, ay0, bx0, by0 = stack[stack_size]
        
        w = abs(ax0 + ay0)
        h = abs(bx0 + by0)
        
        dax = 1 if ax0 > 0 else -1 if ax0 < 0 else 0
        day = 1 if ay0 > 0 else -1 if ay0 < 0 else 0
        dbx = 1 if bx0 > 0 else -1 if bx0 < 0 else 0
        dby = 1 if by0 > 0 else -1 if by0 < 0 else 0
        
        if h == 1:
            for i in range(w):
                curve_map[emit_index] = x0 + y0 * width
                emit_index += 1
                x0 += dax
                y0 += day
            continue
            
        if w == 1:
            for i in range(h):
                curve_map[emit_index] = x0 + y0 * width
                emit_index += 1
                x0 += dbx
                y0 += dby
            continue
        
        ax2 = ax0 // 2
        ay2 = ay0 // 2
        bx2 = bx0 // 2
        by2 = by0 // 2
        
        w2 = abs(ax2 + ay2)
        h2 = abs(bx2 + by2)
        
        if 2 * w > 3 * h:
            if (w2 % 2) and (w > 2):
                ax2 += dax
                ay2 += day
            
            # 先入栈后处理，所以顺序要反
            if stack_size + 2 < len(stack):
                stack[stack_size] = [x0 + ax2, y0 + ay2, 
                                     ax0 - ax2, ay0 - ay2, bx0, by0]
                stack[stack_size + 1] = [x0, y0, ax2, ay2, bx0, by0]
                stack_size += 2
        else:
            if (h2 % 2) and (h > 2):
                bx2 += dbx
                by2 += dby
            
            if stack_size + 3 < len(stack):
                stack[stack_size] = [
                    x0 + (ax0 - dax) + (bx2 - dbx),
                    y0 + (ay0 - day) + (by2 - dby),
                    -bx2, -by2, -(ax0 - ax2), -(ay0 - ay2)
                ]
                stack[stack_size + 1] = [x0 + bx2, y0 + by2, 
                                         ax0, ay0, bx0 - bx2, by0 - by2]
                stack[stack_size + 2] = [x0, y0, bx2, by2, ax2, ay2]
                stack_size += 3
    
    return curve_map

@jit(nopython=True, nogil=True, cache=True, parallel=True)
def apply_curve_mapping_numba(
    pixels: np.ndarray,
    curve_map: np.ndarray,
    mode: str,
    offset: int
) -> np.ndarray:
    """应用曲线映射（Numba加速）"""
    total_pixels = len(curve_map)
    output = np.zeros_like(pixels)
    
    if mode == 'encrypt':
        for i in prange(total_pixels):
            old_pos = curve_map[i]
            new_pos = curve_map[(i + offset) % total_pixels]
            
            old_idx = old_pos * 4
            new_idx = new_pos * 4
            
            output[new_idx] = pixels[old_idx]
            output[new_idx + 1] = pixels[old_idx + 1]
            output[new_idx + 2] = pixels[old_idx + 2]
            output[new_idx + 3] = pixels[old_idx + 3]
    else:  # decrypt
        for i in prange(total_pixels):
            old_pos = curve_map[i]
            new_pos = curve_map[(i + offset) % total_pixels]
            
            old_idx = old_pos * 4
            new_idx = new_pos * 4
            
            output[old_idx] = pixels[new_idx]
            output[old_idx + 1] = pixels[new_idx + 1]
            output[old_idx + 2] = pixels[new_idx + 2]
            output[old_idx + 3] = pixels[new_idx + 3]
    
    return output

# ============= 工作线程 =============
class ProcessWorker(QThread):
    """图片处理工作线程"""
    
    # 信号定义
    progress_updated = pyqtSignal(int, str)  # 进度百分比, 状态文本
    image_processed = pyqtSignal(int, np.ndarray, float)  # 索引, 图片数组, 耗时
    batch_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, cache_manager: CurveCacheManager):
        super().__init__()
        self.cache_manager = cache_manager
        self.images: List[ImageInfo] = []
        self.mode = 'encrypt'  # 'encrypt' or 'decrypt'
        self.output_format = 'PNG'
        self.quality = 95
        self.is_running = True
        self.current_index = 0
        
    def setup_batch(self, images: List[ImageInfo], mode: str, 
                   output_format: str, quality: int):
        """设置批量处理参数"""
        self.images = images
        self.mode = mode
        self.output_format = output_format
        self.quality = quality
        self.current_index = 0
        
    def stop(self):
        """停止处理"""
        self.is_running = False
        
    def calculate_image_hash(self, image_path: str) -> str:
        """计算图片哈希（用于缓存键）"""
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
            return file_hash.hexdigest()
    
    def process_single_image(self, image_info: ImageInfo, index: int) -> Optional[ProcessResult]:
        """处理单张图片"""
        try:
            start_time = time.time()
            
            # 检查缓存
            cache_key = self.cache_manager.get_result_key(
                image_info.hash, self.mode, self.quality, self.output_format
            )
            cached_result = self.cache_manager.get_result(cache_key)
            
            if cached_result:
                self.progress_updated.emit(100, f"图片 {index+1}/{len(self.images)} (使用缓存)")
                self.image_processed.emit(index, cached_result.image_array, 0.001)
                return cached_result
            
            # 加载图片
            img = Image.open(image_info.path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            pixels = np.array(img).astype(np.uint8).flatten()
            
            # 获取或生成曲线映射
            curve_key = self.cache_manager.get_curve_key(image_info.width, image_info.height)
            curve_map = self.cache_manager.get_curve(image_info.width, image_info.height)
            
            if curve_map is None:
                self.progress_updated.emit(10, f"图片 {index+1}/{len(self.images)} (生成曲线映射)")
                curve_map = build_gilbert_curve_map_numba(image_info.width, image_info.height)
                self.cache_manager.save_curve(image_info.width, image_info.height, curve_map)
            
            # 计算偏移量
            total_pixels = image_info.width * image_info.height
            offset = int((np.sqrt(5) - 1) / 2 * total_pixels)
            
            # 应用映射
            self.progress_updated.emit(50, f"图片 {index+1}/{len(self.images)} (应用混淆)")
            result_pixels = apply_curve_mapping_numba(pixels, curve_map, self.mode, offset)
            
            # 重塑为图像数组
            result_array = result_pixels.reshape((image_info.height, image_info.width, 4))
            
            time_cost = time.time() - start_time
            
            # 保存到缓存
            result = ProcessResult(
                image_array=result_array,
                time_cost=time_cost,
                file_size=0  # 实际大小在保存时计算
            )
            self.cache_manager.save_result(cache_key, result)
            
            self.progress_updated.emit(100, f"图片 {index+1}/{len(self.images)} (完成)")
            self.image_processed.emit(index, result_array, time_cost)
            
            return result
            
        except Exception as e:
            self.error_occurred.emit(f"处理图片 {image_info.path} 时出错: {str(e)}")
            return None
    
    def run(self):
        """线程主函数"""
        self.is_running = True
        
        for i, image_info in enumerate(self.images):
            if not self.is_running:
                break
                
            self.current_index = i
            self.process_single_image(image_info, i)
            
            # 模拟进度更新（实际进度在process_single_image中更新）
            progress = int((i + 1) / len(self.images) * 100)
            self.progress_updated.emit(progress, f"批量处理中... ({i+1}/{len(self.images)})")
        
        if self.is_running:
            self.batch_finished.emit()

# ============= 主界面 =============
class ImageObfuscatorGUI(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.cache_manager = CurveCacheManager(max_size=50, ttl=7200)
        self.worker: Optional[ProcessWorker] = None
        self.images: List[ImageInfo] = []
        self.processed_images: List[Optional[np.ndarray]] = []
        self.current_image_index = 0
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("无损图片混淆工具 (Python版)")
        self.setGeometry(100, 100, 1200, 800)
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QProgressBar {
                border: 1px solid #3498db;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 5px;
            }
            QTableWidget {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: white;
            }
        """)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("无损图片混淆工具 - 基于空间填充曲线")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            padding: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #3498db, stop:1 #2ecc71);
            border-radius: 10px;
            color: white;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 分割器：左侧控制面板，右侧预览
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 上传区域
        upload_group = QGroupBox("上传图片")
        upload_layout = QVBoxLayout()
        
        self.upload_btn = QPushButton("📁 选择图片 (支持多选)")
        self.upload_btn.clicked.connect(self.select_images)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        self.clear_btn = QPushButton("🗑️ 清空列表")
        self.clear_btn.clicked.connect(self.clear_images)
        self.clear_btn.setEnabled(False)
        
        upload_btn_layout = QHBoxLayout()
        upload_btn_layout.addWidget(self.upload_btn)
        upload_btn_layout.addWidget(self.clear_btn)
        upload_layout.addLayout(upload_btn_layout)
        
        # 图片列表
        self.image_table = QTableWidget()
        self.image_table.setColumnCount(5)
        self.image_table.setHorizontalHeaderLabels(["文件名", "尺寸", "大小", "格式", "状态"])
        self.image_table.horizontalHeader().setStretchLastSection(True)
        upload_layout.addWidget(self.image_table)
        
        upload_group.setLayout(upload_layout)
        control_layout.addWidget(upload_group)
        
        # 设置区域
        settings_group = QGroupBox("输出设置")
        settings_layout = QVBoxLayout()
        
        # 输出格式
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("输出格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG (无损)", "JPEG", "WebP"])
        self.format_combo.currentIndexChanged.connect(self.on_format_changed)
        format_layout.addWidget(self.format_combo)
        settings_layout.addLayout(format_layout)
        
        # 压缩质量
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("压缩质量:"))
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(0, 100)
        self.quality_slider.setValue(95)
        self.quality_slider.valueChanged.connect(self.on_quality_changed)
        self.quality_label = QLabel("95%")
        quality_layout.addWidget(self.quality_slider)
        quality_layout.addWidget(self.quality_label)
        settings_layout.addLayout(quality_layout)
        
        # 缓存设置
        cache_layout = QHBoxLayout()
        cache_layout.addWidget(QLabel("缓存大小:"))
        self.cache_spinbox = QSpinBox()
        self.cache_spinbox.setRange(10, 200)
        self.cache_spinbox.setValue(50)
        self.cache_spinbox.setSuffix(" MB")
        cache_layout.addWidget(self.cache_spinbox)
        
        self.enable_cache_check = QCheckBox("启用缓存")
        self.enable_cache_check.setChecked(True)
        cache_layout.addWidget(self.enable_cache_check)
        
        self.clear_cache_btn = QPushButton("清空缓存")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        cache_layout.addWidget(self.clear_cache_btn)
        
        settings_layout.addLayout(cache_layout)
        settings_group.setLayout(settings_layout)
        control_layout.addWidget(settings_group)
        
        # 操作按钮区域
        action_group = QGroupBox("图片操作")
        action_layout = QVBoxLayout()
        
        # 单张操作按钮
        single_btn_layout = QHBoxLayout()
        self.encrypt_btn = QPushButton("🔒 混淆当前图片")
        self.encrypt_btn.clicked.connect(self.encrypt_current)
        self.encrypt_btn.setEnabled(False)
        self.encrypt_btn.setStyleSheet("background-color: #3498db; color: white;")
        
        self.decrypt_btn = QPushButton("🔓 解混淆当前图片")
        self.decrypt_btn.clicked.connect(self.decrypt_current)
        self.decrypt_btn.setEnabled(False)
        self.decrypt_btn.setStyleSheet("background-color: #2ecc71; color: white;")
        
        single_btn_layout.addWidget(self.encrypt_btn)
        single_btn_layout.addWidget(self.decrypt_btn)
        action_layout.addLayout(single_btn_layout)
        
        # 批量操作按钮
        batch_btn_layout = QHBoxLayout()
        self.batch_encrypt_btn = QPushButton("🔒 批量混淆")
        self.batch_encrypt_btn.clicked.connect(lambda: self.batch_process('encrypt'))
        self.batch_encrypt_btn.setEnabled(False)
        
        self.batch_decrypt_btn = QPushButton("🔓 批量解混淆")
        self.batch_decrypt_btn.clicked.connect(lambda: self.batch_process('decrypt'))
        self.batch_decrypt_btn.setEnabled(False)
        
        batch_btn_layout.addWidget(self.batch_encrypt_btn)
        batch_btn_layout.addWidget(self.batch_decrypt_btn)
        action_layout.addLayout(batch_btn_layout)
        
        # 下载按钮
        download_btn_layout = QHBoxLayout()
        self.download_btn = QPushButton("💾 下载当前图片")
        self.download_btn.clicked.connect(self.download_current)
        self.download_btn.setEnabled(False)
        self.download_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        
        self.batch_download_btn = QPushButton("💾 批量下载")
        self.batch_download_btn.clicked.connect(self.batch_download)
        self.batch_download_btn.setEnabled(False)
        
        self.zip_download_btn = QPushButton("📦 打包下载")
        self.zip_download_btn.clicked.connect(self.zip_download)
        self.zip_download_btn.setEnabled(False)
        
        download_btn_layout.addWidget(self.download_btn)
        download_btn_layout.addWidget(self.batch_download_btn)
        download_btn_layout.addWidget(self.zip_download_btn)
        action_layout.addLayout(download_btn_layout)
        
        action_group.setLayout(action_layout)
        control_layout.addWidget(action_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("等待操作")
        control_layout.addWidget(self.progress_bar)
        
        control_layout.addStretch()
        splitter.addWidget(control_panel)
        
        # 右侧预览面板
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        
        # 预览标签
        preview_group = QGroupBox("图片预览")
        preview_inner_layout = QVBoxLayout()
        
        # 导航控制
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.prev_btn.setEnabled(False)
        
        self.page_label = QLabel("1 / 1")
        self.page_label.setAlignment(Qt.AlignCenter)
        
        self.next_btn = QPushButton("▶")
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.page_label)
        nav_layout.addWidget(self.next_btn)
        preview_inner_layout.addLayout(nav_layout)
        
        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 2px dashed #bdc3c7;
                border-radius: 10px;
            }
        """)
        self.image_label.setText("图片预览区域")
        preview_inner_layout.addWidget(self.image_label)
        
        # 图片信息
        info_group = QGroupBox("图片信息")
        info_layout = QVBoxLayout()
        
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setRowCount(5)
        self.info_table.setHorizontalHeaderLabels(["属性", "值"])
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        info_items = [
            ("状态", "等待上传"),
            ("文件名", "-"),
            ("尺寸", "-"),
            ("原图大小", "-"),
            ("处理后大小", "-")
        ]
        
        for i, (key, value) in enumerate(info_items):
            self.info_table.setItem(i, 0, QTableWidgetItem(key))
            self.info_table.setItem(i, 1, QTableWidgetItem(value))
        
        info_layout.addWidget(self.info_table)
        info_group.setLayout(info_layout)
        preview_inner_layout.addWidget(info_group)
        
        preview_group.setLayout(preview_inner_layout)
        preview_layout.addWidget(preview_group)
        
        splitter.addWidget(preview_panel)
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
        
        # 定时器用于更新UI
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)  # 100ms更新一次
        
    def load_settings(self):
        """加载设置"""
        # 这里可以添加从文件加载设置的代码
        pass
    
    def save_settings(self):
        """保存设置"""
        # 这里可以添加保存设置到文件的代码
        pass
    
    def select_images(self):
        """选择图片文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片文件", "",
            "图片文件 (*.png *.jpg *.jpeg *.gif *.bmp *.webp);;所有文件 (*.*)"
        )
        
        if files:
            self.add_images(files)
    
    def add_images(self, file_paths: List[str]):
        """添加图片到列表"""
        for file_path in file_paths:
            try:
                # 获取图片信息
                with Image.open(file_path) as img:
                    width, height = img.size
                    format = img.format
                
                file_size = os.path.getsize(file_path)
                file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
                
                image_info = ImageInfo(
                    path=file_path,
                    width=width,
                    height=height,
                    size=file_size,
                    format=format,
                    hash=file_hash
                )
                
                self.images.append(image_info)
                self.processed_images.append(None)
                
                # 更新表格
                row = self.image_table.rowCount()
                self.image_table.insertRow(row)
                
                self.image_table.setItem(row, 0, QTableWidgetItem(os.path.basename(file_path)))
                self.image_table.setItem(row, 1, QTableWidgetItem(f"{width}x{height}"))
                self.image_table.setItem(row, 2, QTableWidgetItem(f"{file_size/1024:.1f} KB"))
                self.image_table.setItem(row, 3, QTableWidgetItem(format))
                self.image_table.setItem(row, 4, QTableWidgetItem("等待处理"))
                
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载图片失败: {str(e)}")
        
        if self.images:
            self.clear_btn.setEnabled(True)
            self.batch_encrypt_btn.setEnabled(True)
            self.batch_decrypt_btn.setEnabled(True)
            self.show_image(0)
    
    def clear_images(self):
        """清空图片列表"""
        self.images.clear()
        self.processed_images.clear()
        self.image_table.setRowCount(0)
        self.clear_btn.setEnabled(False)
        self.batch_encrypt_btn.setEnabled(False)
        self.batch_decrypt_btn.setEnabled(False)
        self.encrypt_btn.setEnabled(False)
        self.decrypt_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        self.image_label.setText("图片预览区域")
        self.update_info_table()
    
    def show_image(self, index: int):
        """显示指定索引的图片"""
        if 0 <= index < len(self.images):
            self.current_image_index = index
            
            # 更新导航
            self.page_label.setText(f"{index + 1} / {len(self.images)}")
            self.prev_btn.setEnabled(index > 0)
            self.next_btn.setEnabled(index < len(self.images) - 1)
            
            # 显示图片
            if self.processed_images[index] is not None:
                self.display_numpy_image(self.processed_images[index])
                self.info_table.item(0, 1).setText("已处理")
                self.download_btn.setEnabled(True)
            else:
                # 显示原图
                pixmap = QPixmap(self.images[index].path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        self.image_label.size() * 0.9,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.image_label.setPixmap(scaled_pixmap)
                    self.info_table.item(0, 1).setText("原始图片")
                    self.download_btn.setEnabled(False)
            
            # 更新信息
            self.update_info_table()
            self.encrypt_btn.setEnabled(True)
            self.decrypt_btn.setEnabled(True)
    
    def display_numpy_image(self, image_array: np.ndarray):
        """显示numpy数组图片"""
        height, width, channel = image_array.shape
        
        if channel == 4:
            qimage = QImage(image_array.data, width, height, width * 4, QImage.Format_RGBA8888)
        else:
            qimage = QImage(image_array.data, width, height, width * 3, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size() * 0.9,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    def update_info_table(self):
        """更新信息表格"""
        if self.images and self.current_image_index < len(self.images):
            img_info = self.images[self.current_image_index]
            
            self.info_table.item(1, 1).setText(os.path.basename(img_info.path))
            self.info_table.item(2, 1).setText(f"{img_info.width} × {img_info.height}")
            self.info_table.item(3, 1).setText(f"{img_info.size / 1024:.1f} KB")
            
            # 处理后大小
            if self.processed_images[self.current_image_index] is not None:
                # 这里可以计算实际文件大小，简化处理
                self.info_table.item(4, 1).setText("计算中...")
            else:
                self.info_table.item(4, 1).setText("-")
    
    def show_prev_image(self):
        """显示上一张图片"""
        if self.current_image_index > 0:
            self.show_image(self.current_image_index - 1)
    
    def show_next_image(self):
        """显示下一张图片"""
        if self.current_image_index < len(self.images) - 1:
            self.show_image(self.current_image_index + 1)
    
    def encrypt_current(self):
        """混淆当前图片"""
        if self.current_image_index < len(self.images):
            self.process_single_image(self.current_image_index, 'encrypt')
    
    def decrypt_current(self):
        """解混淆当前图片"""
        if self.current_image_index < len(self.images):
            self.process_single_image(self.current_image_index, 'decrypt')
    
    def process_single_image(self, index: int, mode: str):
        """处理单张图片"""
        if index >= len(self.images):
            return
        
        # 创建临时工作线程
        self.worker = ProcessWorker(self.cache_manager)
        self.worker.setup_batch([self.images[index]], mode, 
                               self.format_combo.currentText(),
                               self.quality_slider.value())
        
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.image_processed.connect(self.on_image_processed)
        self.worker.error_occurred.connect(self.on_error_occurred)
        
        self.worker.start()
        
        # 禁用按钮
        self.set_buttons_enabled(False)
    
    def batch_process(self, mode: str):
        """批量处理图片"""
        if not self.images:
            return
        
        # 创建批处理工作线程
        self.worker = ProcessWorker(self.cache_manager)
        self.worker.setup_batch(self.images, mode, 
                               self.format_combo.currentText(),
                               self.quality_slider.value())
        
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.image_processed.connect(self.on_image_processed)
        self.worker.batch_finished.connect(self.on_batch_finished)
        self.worker.error_occurred.connect(self.on_error_occurred)
        
        self.worker.start()
        
        # 禁用按钮
        self.set_buttons_enabled(False)
    
    def on_progress_updated(self, progress: int, status: str):
        """处理进度更新"""
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{status} - {progress}%")
        self.status_bar.showMessage(status)
    
    def on_image_processed(self, index: int, image_array: np.ndarray, time_cost: float):
        """单张图片处理完成"""
        self.processed_images[index] = image_array
        
        # 更新表格状态
        mode = "混淆" if self.worker and self.worker.mode == 'encrypt' else "解混淆"
        self.image_table.item(index, 4).setText(f"{mode}完成 ({time_cost:.2f}s)")
        
        # 如果是当前显示的图片，更新显示
        if index == self.current_image_index:
            self.show_image(index)
        
        # 启用批量下载按钮
        if any(img is not None for img in self.processed_images):
            self.batch_download_btn.setEnabled(True)
            self.zip_download_btn.setEnabled(True)
    
    def on_batch_finished(self):
        """批量处理完成"""
        self.set_buttons_enabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("批量处理完成")
        self.status_bar.showMessage("批量处理完成")
        
        QMessageBox.information(self, "完成", "批量处理完成！")
    
    def on_error_occurred(self, error_msg: str):
        """处理错误"""
        self.set_buttons_enabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("处理出错")
        
        QMessageBox.critical(self, "错误", error_msg)
    
    def set_buttons_enabled(self, enabled: bool):
        """设置按钮启用状态"""
        self.encrypt_btn.setEnabled(enabled and bool(self.images))
        self.decrypt_btn.setEnabled(enabled and bool(self.images))
        self.batch_encrypt_btn.setEnabled(enabled and bool(self.images))
        self.batch_decrypt_btn.setEnabled(enabled and bool(self.images))
        self.download_btn.setEnabled(enabled and self.processed_images[self.current_image_index] is not None)
        self.clear_btn.setEnabled(enabled)
    
    def download_current(self):
        """下载当前图片"""
        if self.current_image_index < len(self.processed_images):
            processed_img = self.processed_images[self.current_image_index]
            if processed_img is not None:
                self.save_image(processed_img, self.current_image_index)
    
    def batch_download(self):
        """批量下载"""
        if not any(img is not None for img in self.processed_images):
            return
        
        # 选择保存目录
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not save_dir:
            return
        
        for i, img_array in enumerate(self.processed_images):
            if img_array is not None:
                try:
                    self.save_image_to_path(img_array, i, save_dir)
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"保存图片 {i+1} 失败: {str(e)}")
        
        QMessageBox.information(self, "完成", f"已保存 {len(self.processed_images)} 张图片到 {save_dir}")
    
    def zip_download(self):
        """打包下载"""
        QMessageBox.information(self, "提示", "打包下载功能需要zipfile库支持，请参考注释代码实现")
        # 这里可以实现ZIP打包功能
    
    def save_image(self, image_array: np.ndarray, index: int):
        """保存图片"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", 
            f"image_{index+1}.png",
            f"图片文件 (*.png *.jpg *.jpeg *.webp)"
        )
        
        if file_path:
            self.save_image_to_path(image_array, index, os.path.dirname(file_path), 
                                  os.path.basename(file_path))
    
    def save_image_to_path(self, image_array: np.ndarray, index: int, 
                          directory: str, filename: str = None):
        """保存图片到指定路径"""
        if filename is None:
            ext = self.get_format_extension()
            filename = f"image_{index+1}.{ext}"
        
        save_path = os.path.join(directory, filename)
        
        # 转换格式
        img = Image.fromarray(image_array)
        
        # 根据选择的格式保存
        format_text = self.format_combo.currentText()
        if "PNG" in format_text:
            img.save(save_path, "PNG")
        elif "JPEG" in format_text:
            img = img.convert("RGB")  # JPEG不支持透明度
            img.save(save_path, "JPEG", quality=self.quality_slider.value())
        elif "WebP" in format_text:
            img.save(save_path, "WebP", quality=self.quality_slider.value())
        
        # 更新文件大小信息
        file_size = os.path.getsize(save_path)
        self.info_table.item(4, 1).setText(f"{file_size / 1024:.1f} KB")
    
    def get_format_extension(self) -> str:
        """获取当前格式的扩展名"""
        format_text = self.format_combo.currentText()
        if "PNG" in format_text:
            return "png"
        elif "JPEG" in format_text:
            return "jpg"
        elif "WebP" in format_text:
            return "webp"
        return "png"
    
    def on_format_changed(self, index: int):
        """格式改变事件"""
        if index == 0:  # PNG
            self.quality_slider.setEnabled(False)
        else:
            self.quality_slider.setEnabled(True)
    
    def on_quality_changed(self, value: int):
        """质量滑块改变事件"""
        self.quality_label.setText(f"{value}%")
    
    def clear_cache(self):
        """清空缓存"""
        self.cache_manager.clear()
        self.status_bar.showMessage("缓存已清空", 3000)
    
    def update_ui(self):
        """定时更新UI"""
        # 这里可以添加需要定时更新的UI元素
        pass
    
    def closeEvent(self, event):
        """关闭事件"""
        # 停止工作线程
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        # 保存设置
        self.save_settings()
        
        # 清理资源
        self.cache_manager.clear()
        
        event.accept()

# ============= 主程序入口 =============
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("无损图片混淆工具")
    app.setStyle("Fusion")
    
    # 设置深色主题
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = ImageObfuscatorGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    # 预热Numba JIT编译器
    print("预热Numba JIT编译器...")
    test_array = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8).flatten()
    test_map = build_gilbert_curve_map_numba(100, 100)
    apply_curve_mapping_numba(test_array, test_map, 'encrypt', 1000)
    print("预热完成，启动应用...")
    
    main()
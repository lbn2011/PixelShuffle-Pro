# image_obfuscator_advanced.py
import sys
import os
import time
import threading
import hashlib
import logging
import traceback
import concurrent.futures
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Any, Callable
from dataclasses import dataclass, asdict
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QComboBox, QSlider, QSpinBox, QCheckBox, QGroupBox,
                            QProgressBar, QTableWidget, QTableWidgetItem, 
                            QMessageBox, QSplitter, QTabWidget, QTextEdit,
                            QDockWidget, QTextBrowser, QListWidget, QListWidgetItem,
                            QDialog, QFormLayout, QLineEdit, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl, QDateTime
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QDesktopServices, QFont, QIcon
import numba
from numba import jit, prange, cuda
from cachetools import LRUCache, TTLCache
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, Manager
import warnings
warnings.filterwarnings('ignore')

# ============= 日志配置 =============
def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件名格式
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"image_obfuscator_{timestamp}.log")
    
    # 日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(process)d:%(thread)d] - %(filename)s:%(lineno)d - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 获取日志级别
    level = getattr(logging, log_level.upper())
    
    # 配置根日志
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    # 创建主日志记录器
    logger = logging.getLogger("ImageObfuscator")
    logger.setLevel(level)
    
    # 避免日志重复
    logger.propagate = False
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger.addHandler(console_handler)
    
    return logger, log_file

# 初始化日志
logger, log_file = setup_logging()

# ============= 日志窗口处理器 =============
class QtLogHandler(logging.Handler):
    """将日志发送到Qt窗口的处理器"""
    
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.signal.emit(msg, record.levelno)
        except Exception:
            pass

# ============= 错误处理装饰器 =============
def handle_exceptions(func: Callable) -> Callable:
    """异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行出错: {str(e)}", exc_info=True)
            # 如果是UI相关函数，显示错误对话框
            if len(args) > 0 and hasattr(args[0], 'show_error_dialog'):
                args[0].show_error_dialog(str(e), traceback.format_exc())
            raise
    return wrapper

# ============= 数据结构定义 =============
@dataclass
class ImageInfo:
    """图片信息"""
    path: str
    width: int
    height: int
    size: int
    format: str
    hash: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ProcessResult:
    """处理结果"""
    image_array: np.ndarray
    time_cost: float
    file_size: int
    cache_hit: bool = False

@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_time: float = 0.0
    avg_time_per_image: float = 0.0
    images_processed: int = 0
    cache_hits: int = 0
    cpu_cores_used: int = 1

# ============= 缓存管理器 (带序列化) =============
class CurveCacheManager:
    """曲线映射缓存管理器（支持多进程）"""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 内存缓存
        self.curve_cache = LRUCache(maxsize=max_size)
        self.result_cache = TTLCache(maxsize=50, ttl=ttl)
        
        # 磁盘缓存文件
        self.curve_cache_file = self.cache_dir / "curve_cache.pkl"
        self.result_cache_file = self.cache_dir / "result_cache.pkl"
        
        # 加载磁盘缓存
        self.load_from_disk()
        
        logger.info(f"缓存管理器初始化完成，内存缓存大小: {max_size}, TTL: {ttl}秒")
    
    def save_to_disk(self):
        """保存缓存到磁盘"""
        try:
            # 保存曲线缓存
            with open(self.curve_cache_file, 'wb') as f:
                pickle.dump(dict(self.curve_cache), f)
            
            # 保存结果缓存
            with open(self.result_cache_file, 'wb') as f:
                pickle.dump(dict(self.result_cache), f)
                
            logger.debug("缓存已保存到磁盘")
        except Exception as e:
            logger.error(f"保存缓存到磁盘失败: {e}")
    
    def load_from_disk(self):
        """从磁盘加载缓存"""
        try:
            if self.curve_cache_file.exists():
                with open(self.curve_cache_file, 'rb') as f:
                    curve_data = pickle.load(f)
                    self.curve_cache.update(curve_data)
            
            if self.result_cache_file.exists():
                with open(self.result_cache_file, 'rb') as f:
                    result_data = pickle.load(f)
                    self.result_cache.update(result_data)
                    
            logger.info(f"从磁盘加载缓存完成，曲线缓存: {len(self.curve_cache)}项，结果缓存: {len(self.result_cache)}项")
        except Exception as e:
            logger.error(f"从磁盘加载缓存失败: {e}")
    
    def get_curve_key(self, width: int, height: int) -> str:
        return f"{width}x{height}"
    
    def get_result_key(self, image_hash: str, mode: str, 
                      quality: int, format: str) -> str:
        return f"{image_hash}_{mode}_{quality}_{format}"
    
    def save_curve(self, width: int, height: int, curve_map: np.ndarray):
        key = self.get_curve_key(width, height)
        self.curve_cache[key] = curve_map
        logger.debug(f"保存曲线映射缓存: {key}")
    
    def get_curve(self, width: int, height: int) -> Optional[np.ndarray]:
        key = self.get_curve_key(width, height)
        curve = self.curve_cache.get(key)
        if curve is not None:
            logger.debug(f"曲线映射缓存命中: {key}")
        return curve
    
    def save_result(self, key: str, result: ProcessResult):
        self.result_cache[key] = result
        logger.debug(f"保存处理结果缓存: {key}")
    
    def get_result(self, key: str) -> Optional[ProcessResult]:
        result = self.result_cache.get(key)
        if result is not None:
            logger.debug(f"处理结果缓存命中: {key}")
        return result
    
    def clear(self, clear_disk: bool = False):
        """清空缓存"""
        self.curve_cache.clear()
        self.result_cache.clear()
        
        if clear_disk and self.cache_dir.exists():
            for file in self.cache_dir.glob("*.pkl"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"删除缓存文件失败 {file}: {e}")
        
        logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            "curve_cache_size": len(self.curve_cache),
            "result_cache_size": len(self.result_cache),
            "curve_cache_max": self.curve_cache.maxsize,
            "result_cache_max": self.result_cache.maxsize
        }

# ============= Numba加速算法 (多核优化) =============
@jit(nopython=True, nogil=True, cache=True, parallel=True)
def build_gilbert_curve_map_numba(width: int, height: int) -> np.ndarray:
    """使用Numba加速构建Gilbert曲线映射（多核并行）"""
    total_pixels = width * height
    curve_map = np.zeros(total_pixels, dtype=np.uint32)
    emit_index = 0
    
    # 预分配栈空间
    stack_size = 0
    max_stack_size = total_pixels * 2
    stack_x = np.zeros(max_stack_size, dtype=np.int32)
    stack_y = np.zeros(max_stack_size, dtype=np.int32)
    stack_ax = np.zeros(max_stack_size, dtype=np.int32)
    stack_ay = np.zeros(max_stack_size, dtype=np.int32)
    stack_bx = np.zeros(max_stack_size, dtype=np.int32)
    stack_by = np.zeros(max_stack_size, dtype=np.int32)
    
    # 初始参数入栈
    if width >= height:
        stack_x[stack_size] = 0
        stack_y[stack_size] = 0
        stack_ax[stack_size] = width
        stack_ay[stack_size] = 0
        stack_bx[stack_size] = 0
        stack_by[stack_size] = height
    else:
        stack_x[stack_size] = 0
        stack_y[stack_size] = 0
        stack_ax[stack_size] = 0
        stack_ay[stack_size] = height
        stack_bx[stack_size] = width
        stack_by[stack_size] = 0
    stack_size += 1
    
    while stack_size > 0:
        stack_size -= 1
        x0 = stack_x[stack_size]
        y0 = stack_y[stack_size]
        ax0 = stack_ax[stack_size]
        ay0 = stack_ay[stack_size]
        bx0 = stack_bx[stack_size]
        by0 = stack_by[stack_size]
        
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
            
            # 入栈顺序反
            if stack_size + 2 < max_stack_size:
                stack_x[stack_size] = x0 + ax2
                stack_y[stack_size] = y0 + ay2
                stack_ax[stack_size] = ax0 - ax2
                stack_ay[stack_size] = ay0 - ay2
                stack_bx[stack_size] = bx0
                stack_by[stack_size] = by0
                
                stack_x[stack_size + 1] = x0
                stack_y[stack_size + 1] = y0
                stack_ax[stack_size + 1] = ax2
                stack_ay[stack_size + 1] = ay2
                stack_bx[stack_size + 1] = bx0
                stack_by[stack_size + 1] = by0
                
                stack_size += 2
        else:
            if (h2 % 2) and (h > 2):
                bx2 += dbx
                by2 += dby
            
            if stack_size + 3 < max_stack_size:
                stack_x[stack_size] = x0 + (ax0 - dax) + (bx2 - dbx)
                stack_y[stack_size] = y0 + (ay0 - day) + (by2 - dby)
                stack_ax[stack_size] = -bx2
                stack_ay[stack_size] = -by2
                stack_bx[stack_size] = -(ax0 - ax2)
                stack_by[stack_size] = -(ay0 - ay2)
                
                stack_x[stack_size + 1] = x0 + bx2
                stack_y[stack_size + 1] = y0 + by2
                stack_ax[stack_size + 1] = ax0
                stack_ay[stack_size + 1] = ay0
                stack_bx[stack_size + 1] = bx0 - bx2
                stack_by[stack_size + 1] = by0 - by2
                
                stack_x[stack_size + 2] = x0
                stack_y[stack_size + 2] = y0
                stack_ax[stack_size + 2] = bx2
                stack_ay[stack_size + 2] = by2
                stack_bx[stack_size + 2] = ax2
                stack_by[stack_size + 2] = ay2
                
                stack_size += 3
    
    return curve_map

@jit(nopython=True, nogil=True, cache=True, parallel=True)
def apply_curve_mapping_numba(
    pixels: np.ndarray,
    curve_map: np.ndarray,
    mode: str,
    offset: int
) -> np.ndarray:
    """应用曲线映射（Numba加速，并行处理）"""
    total_pixels = len(curve_map)
    output = np.zeros_like(pixels)
    
    # 使用多线程并行处理
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

# ============= 多进程工作函数 =============
def process_single_image_worker(args):
    """多进程工作函数（单张图片处理）"""
    try:
        index, image_info_dict, mode, quality, output_format = args
        
        # 重建图像信息
        image_info = ImageInfo(**image_info_dict)
        
        start_time = time.time()
        
        # 创建本地缓存管理器
        cache_manager = CurveCacheManager(max_size=10, ttl=1800)
        
        # 检查缓存
        cache_key = cache_manager.get_result_key(
            image_info.hash, mode, quality, output_format
        )
        cached_result = cache_manager.get_result(cache_key)
        
        if cached_result:
            logger.debug(f"进程 {mp.current_process().pid}: 图片 {index} 缓存命中")
            result_array = cached_result.image_array
            time_cost = 0.001
            cache_hit = True
        else:
            # 加载图片
            img = Image.open(image_info.path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            pixels = np.array(img).astype(np.uint8).flatten()
            
            # 获取或生成曲线映射
            curve_key = cache_manager.get_curve_key(image_info.width, image_info.height)
            curve_map = cache_manager.get_curve(image_info.width, image_info.height)
            
            if curve_map is None:
                curve_map = build_gilbert_curve_map_numba(image_info.width, image_info.height)
                cache_manager.save_curve(image_info.width, image_info.height, curve_map)
            
            # 计算偏移量
            total_pixels = image_info.width * image_info.height
            offset = int((np.sqrt(5) - 1) / 2 * total_pixels)
            
            # 应用映射
            result_pixels = apply_curve_mapping_numba(pixels, curve_map, mode, offset)
            
            # 重塑为图像数组
            result_array = result_pixels.reshape((image_info.height, image_info.width, 4))
            
            time_cost = time.time() - start_time
            cache_hit = False
        
        return {
            'index': index,
            'success': True,
            'image_array': result_array,
            'time_cost': time_cost,
            'cache_hit': cache_hit,
            'message': f"处理完成 ({time_cost:.2f}s)"
        }
        
    except Exception as e:
        logger.error(f"进程 {mp.current_process().pid}: 处理图片 {index} 失败: {e}")
        return {
            'index': index,
            'success': False,
            'error': str(e),
            'message': f"处理失败: {str(e)[:50]}"
        }

# ============= 多进程管理器 =============
class MultiprocessManager:
    """多进程管理器"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        self.pool = None
        self.is_running = False
        logger.info(f"多进程管理器初始化，最大工作进程数: {self.max_workers}")
    
    def start_pool(self):
        """启动进程池"""
        if self.pool is None:
            self.pool = Pool(processes=self.max_workers)
            logger.debug(f"进程池已启动，使用 {self.max_workers} 个工作进程")
    
    def stop_pool(self):
        """停止进程池"""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
            logger.debug("进程池已停止")
    
    def process_images(self, images: List[ImageInfo], mode: str, 
                      quality: int, output_format: str) -> List[Dict]:
        """批量处理图片（多进程）"""
        self.start_pool()
        self.is_running = True
        
        try:
            # 准备任务参数
            tasks = []
            for i, img_info in enumerate(images):
                tasks.append((
                    i,
                    img_info.to_dict(),
                    mode,
                    quality,
                    output_format
                ))
            
            logger.info(f"开始多进程批量处理，共 {len(tasks)} 张图片，使用 {self.max_workers} 个进程")
            
            # 使用imap_unordered获取实时进度
            results = []
            for result in self.pool.imap_unordered(process_single_image_worker, tasks):
                if not self.is_running:
                    break
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"多进程处理失败: {e}")
            raise
        finally:
            self.is_running = False
    
    def __del__(self):
        self.stop_pool()

# ============= 工作线程 (UI线程) =============
class ProcessWorker(QThread):
    """图片处理工作线程（UI线程）"""
    
    # 信号定义
    progress_updated = pyqtSignal(int, str)  # 进度百分比, 状态文本
    image_processed = pyqtSignal(int, np.ndarray, float, bool)  # 索引, 图片数组, 耗时, 缓存命中
    batch_finished = pyqtSignal(PerformanceMetrics)
    error_occurred = pyqtSignal(str, str)  # 错误标题, 错误信息
    log_message = pyqtSignal(str, int)  # 日志消息, 日志级别
    
    def __init__(self, cache_manager: CurveCacheManager):
        super().__init__()
        self.cache_manager = cache_manager
        self.multiprocess_manager = MultiprocessManager()
        self.images: List[ImageInfo] = []
        self.mode = 'encrypt'
        self.output_format = 'PNG'
        self.quality = 95
        self.is_running = False
        self.metrics = PerformanceMetrics()
        
        # 连接日志信号
        self.log_message.connect(self.log_to_ui)
    
    def log_to_ui(self, message: str, level: int):
        """转发日志到UI"""
        # 这里会由主窗口处理，显示在日志窗口中
        pass
    
    def setup_batch(self, images: List[ImageInfo], mode: str, 
                   output_format: str, quality: int):
        """设置批量处理参数"""
        self.images = images
        self.mode = mode
        self.output_format = output_format
        self.quality = quality
        self.metrics = PerformanceMetrics()
        
        logger.info(f"工作线程设置: 模式={mode}, 格式={output_format}, 质量={quality}, 图片数={len(images)}")
    
    @handle_exceptions
    def run(self):
        """线程主函数"""
        self.is_running = True
        start_time = time.time()
        
        try:
            if not self.images:
                logger.warning("没有图片需要处理")
                return
            
            # 使用多进程处理
            logger.info(f"开始处理 {len(self.images)} 张图片")
            
            results = self.multiprocess_manager.process_images(
                self.images, self.mode, self.quality, self.output_format
            )
            
            # 统计结果
            successful = 0
            failed = 0
            cache_hits = 0
            
            for result in results:
                if not self.is_running:
                    break
                
                index = result['index']
                if result['success']:
                    successful += 1
                    if result.get('cache_hit', False):
                        cache_hits += 1
                    
                    # 发射处理完成的信号
                    self.image_processed.emit(
                        index,
                        result['image_array'],
                        result['time_cost'],
                        result['cache_hit']
                    )
                    
                    logger.info(f"图片 {index+1}/{len(self.images)} 处理成功: {result['message']}")
                else:
                    failed += 1
                    logger.error(f"图片 {index+1}/{len(self.images)} 处理失败: {result.get('error', '未知错误')}")
                
                # 更新进度
                progress = int((successful + failed) / len(self.images) * 100)
                self.progress_updated.emit(progress, 
                                         f"处理中... ({successful+failed}/{len(self.images)})")
            
            # 计算性能指标
            total_time = time.time() - start_time
            self.metrics.total_time = total_time
            self.metrics.avg_time_per_image = total_time / successful if successful > 0 else 0
            self.metrics.images_processed = successful
            self.metrics.cache_hits = cache_hits
            self.metrics.cpu_cores_used = self.multiprocess_manager.max_workers
            
            logger.info(f"批量处理完成: 成功={successful}, 失败={failed}, "
                       f"总耗时={total_time:.2f}s, 缓存命中={cache_hits}")
            
            if failed > 0:
                self.error_occurred.emit("部分图片处理失败", 
                                       f"{failed} 张图片处理失败，请查看日志了解详情")
            
            # 发射完成信号
            self.batch_finished.emit(self.metrics)
            
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            error_detail = traceback.format_exc()
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit("处理错误", f"{error_msg}\n\n详细错误:\n{error_detail}")
        
        finally:
            self.is_running = False
            logger.debug("工作线程结束")

# ============= 设置对话框 =============
class SettingsDialog(QDialog):
    """设置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("程序设置")
        self.setMinimumWidth(400)
        
        layout = QFormLayout(self)
        
        # 日志级别
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        layout.addRow("日志级别:", self.log_level_combo)
        
        # 最大进程数
        self.max_process_spin = QSpinBox()
        self.max_process_spin.setRange(1, cpu_count())
        self.max_process_spin.setValue(max(1, cpu_count() - 1))
        layout.addRow("最大进程数:", self.max_process_spin)
        
        # 缓存大小
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(10, 500)
        self.cache_size_spin.setValue(100)
        self.cache_size_spin.setSuffix(" MB")
        layout.addRow("缓存大小:", self.cache_size_spin)
        
        # 缓存过期时间
        self.cache_ttl_spin = QSpinBox()
        self.cache_ttl_spin.setRange(300, 86400)
        self.cache_ttl_spin.setValue(3600)
        self.cache_ttl_spin.setSuffix(" 秒")
        layout.addRow("缓存过期时间:", self.cache_ttl_spin)
        
        # 自动保存日志
        self.auto_save_check = QCheckBox("自动保存处理日志")
        self.auto_save_check.setChecked(True)
        layout.addRow("", self.auto_save_check)
        
        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
        
    def get_settings(self):
        """获取设置"""
        return {
            'log_level': self.log_level_combo.currentText(),
            'max_processes': self.max_process_spin.value(),
            'cache_size': self.cache_size_spin.value(),
            'cache_ttl': self.cache_ttl_spin.value(),
            'auto_save_log': self.auto_save_check.isChecked()
        }

# ============= 日志窗口 =============
class LogWindow(QDockWidget):
    """日志窗口"""
    
    def __init__(self, parent=None):
        super().__init__("日志窗口", parent)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 日志文本显示
        self.log_text = QTextBrowser()
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setReadOnly(True)
        
        # 添加颜色标记
        self.log_text.document().setDefaultStyleSheet("""
            .debug { color: gray; }
            .info { color: black; }
            .warning { color: orange; }
            .error { color: red; }
            .critical { color: darkred; font-weight: bold; }
        """)
        
        layout.addWidget(self.log_text)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.clicked.connect(self.clear_log)
        
        self.save_btn = QPushButton("保存日志")
        self.save_btn.clicked.connect(self.save_log)
        
        self.auto_scroll_check = QCheckBox("自动滚动")
        self.auto_scroll_check.setChecked(True)
        
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.auto_scroll_check)
        
        layout.addLayout(button_layout)
        
        self.setWidget(widget)
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        
    def add_log_message(self, message: str, level: int = logging.INFO):
        """添加日志消息"""
        # 根据级别确定CSS类
        if level >= logging.CRITICAL:
            css_class = "critical"
        elif level >= logging.ERROR:
            css_class = "error"
        elif level >= logging.WARNING:
            css_class = "warning"
        elif level >= logging.INFO:
            css_class = "info"
        else:
            css_class = "debug"
        
        # 添加时间戳
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        formatted_msg = f'<span class="{css_class}">[{timestamp}] {message}</span>'
        
        # 添加HTML格式的消息
        self.log_text.append(formatted_msg)
        
        # 自动滚动
        if self.auto_scroll_check.isChecked():
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
    
    def clear_log(self, checked=False):
        """清空日志"""
        self.log_text.clear()
    
    def save_log(self, checked=False):
        """保存日志到文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存日志",
            f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML文件 (*.html);;文本文件 (*.txt)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.html'):
                    # 保存为HTML格式
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.log_text.toHtml())
                else:
                    # 保存为纯文本格式
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.log_text.toPlainText())
                
                logger.info(f"日志已保存到: {file_path}")
                QMessageBox.information(self, "成功", f"日志已保存到:\n{file_path}")
            except Exception as e:
                logger.error(f"保存日志失败: {e}")
                QMessageBox.critical(self, "错误", f"保存日志失败:\n{str(e)}")

# ============= 主界面 =============
class ImageObfuscatorGUI(QMainWindow):
    """主窗口"""
    
    # 自定义信号
    log_signal = pyqtSignal(str, int)
    
    def __init__(self):
        super().__init__()
        
        # 初始化设置
        self.settings = self.load_settings()
        
        # 初始化缓存管理器
        self.cache_manager = CurveCacheManager(
            max_size=self.settings.get('cache_size', 100),
            ttl=self.settings.get('cache_ttl', 3600)
        )
        
        # 初始化日志处理器
        self.log_handler = QtLogHandler(self.log_signal)
        self.log_handler.setLevel(getattr(logging, self.settings.get('log_level', 'INFO')))
        logger.addHandler(self.log_handler)
        
        # 连接日志信号
        self.log_signal.connect(self.on_log_message)
        
        # 初始化变量
        self.worker: Optional[ProcessWorker] = None
        self.images: List[ImageInfo] = []
        self.processed_images: List[Optional[np.ndarray]] = []
        self.current_image_index = 0
        self.performance_metrics = PerformanceMetrics()
        
        # 初始化UI
        self.init_ui()
        
        logger.info("应用程序初始化完成")
    
    @handle_exceptions
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("无损图片混淆工具 - 高级版")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置应用图标
        self.setWindowIcon(QIcon(self.create_icon()))
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #87CEEB;
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
                min-height: 30px;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
            QProgressBar {
                border: 1px solid #3498db;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3498db, stop: 1 #2ecc71
                );
                border-radius: 5px;
            }
            QTableWidget {
                border: 1px solid #87CEEB;
                border-radius: 5px;
                background-color: #e6f3ff;
                alternate-background-color: #d4e7ff;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QLabel#titleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 15px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:0.5 #2ecc71, stop:1 #e74c3c);
                border-radius: 10px;
                color: white;
            }
            QLineEdit, QComboBox, QSpinBox, QTextEdit, QTextBrowser {
                border: 1px solid #87CEEB;
                border-radius: 5px;
                background-color: #e6f3ff;
                padding: 5px;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #87CEEB;
                background-color: #e6f3ff;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #3498db;
                background-color: #3498db;
            }
            QSlider::groove:horizontal {
                border: 1px solid #87CEEB;
                height: 8px;
                background: #e6f3ff;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("无损图片混淆工具 - 基于空间填充曲线 (多核加速版)")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 分割器：左侧控制面板，右侧预览
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 上传区域
        upload_group = QGroupBox("📁 上传图片")
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
        
        self.drag_drop_label = QLabel("或拖拽图片文件到此处")
        self.drag_drop_label.setAlignment(Qt.AlignCenter)
        self.drag_drop_label.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-style: italic;
                padding: 10px;
                border: 2px dashed #bdc3c7;
                border-radius: 8px;
                margin: 5px;
            }
        """)
        
        self.clear_btn = QPushButton("🗑️ 清空列表")
        self.clear_btn.clicked.connect(self.clear_images)
        self.clear_btn.setEnabled(False)
        self.clear_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        
        upload_btn_layout = QHBoxLayout()
        upload_btn_layout.addWidget(self.upload_btn)
        upload_btn_layout.addWidget(self.clear_btn)
        upload_layout.addLayout(upload_btn_layout)
        upload_layout.addWidget(self.drag_drop_label)
        
        # 图片列表
        self.image_table = QTableWidget()
        self.image_table.setColumnCount(6)
        self.image_table.setHorizontalHeaderLabels(["文件名", "尺寸", "大小", "格式", "状态", "耗时"])
        self.image_table.horizontalHeader().setStretchLastSection(True)
        self.image_table.setAlternatingRowColors(True)
        self.image_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.image_table.doubleClicked.connect(self.on_image_double_clicked)
        upload_layout.addWidget(self.image_table)
        
        upload_group.setLayout(upload_layout)
        control_layout.addWidget(upload_group)
        
        # 设置区域
        settings_group = QGroupBox("⚙️ 输出设置")
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
        self.quality_label.setMinimumWidth(40)
        quality_layout.addWidget(self.quality_slider)
        quality_layout.addWidget(self.quality_label)
        settings_layout.addLayout(quality_layout)
        
        # 处理模式
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("处理模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["混淆 (加密)", "解混淆 (解密)"])
        mode_layout.addWidget(self.mode_combo)
        settings_layout.addLayout(mode_layout)
        
        settings_group.setLayout(settings_layout)
        control_layout.addWidget(settings_group)
        
        # 缓存控制区域
        cache_group = QGroupBox("💾 缓存控制")
        cache_layout = QVBoxLayout()
        
        cache_info_layout = QHBoxLayout()
        self.cache_info_label = QLabel("缓存: 0曲线, 0结果")
        self.cache_info_label.setStyleSheet("color: #3498db; font-weight: bold;")
        cache_info_layout.addWidget(self.cache_info_label)
        
        self.update_cache_btn = QPushButton("🔄 更新缓存信息")
        self.update_cache_btn.clicked.connect(self.update_cache_info)
        cache_info_layout.addWidget(self.update_cache_btn)
        cache_info_layout.addStretch()
        cache_layout.addLayout(cache_info_layout)
        
        cache_btn_layout = QHBoxLayout()
        self.clear_cache_btn = QPushButton("🗑️ 清空内存缓存")
        self.clear_cache_btn.clicked.connect(lambda: self.clear_cache(False))
        self.clear_cache_btn.setStyleSheet("background-color: #f39c12; color: white;")
        
        self.clear_disk_cache_btn = QPushButton("🗑️ 清空磁盘缓存")
        self.clear_disk_cache_btn.clicked.connect(lambda: self.clear_cache(True))
        self.clear_disk_cache_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        
        cache_btn_layout.addWidget(self.clear_cache_btn)
        cache_btn_layout.addWidget(self.clear_disk_cache_btn)
        cache_layout.addLayout(cache_btn_layout)
        
        cache_group.setLayout(cache_layout)
        control_layout.addWidget(cache_group)
        
        # 操作按钮区域
        action_group = QGroupBox("🚀 图片操作")
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
        
        self.restore_btn = QPushButton("↩️ 还原为原图")
        self.restore_btn.clicked.connect(self.restore_current)
        self.restore_btn.setEnabled(False)
        self.restore_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        
        single_btn_layout.addWidget(self.encrypt_btn)
        single_btn_layout.addWidget(self.decrypt_btn)
        single_btn_layout.addWidget(self.restore_btn)
        action_layout.addLayout(single_btn_layout)
        
        # 批量操作按钮
        batch_btn_layout = QHBoxLayout()
        self.batch_encrypt_btn = QPushButton("🔒 批量混淆")
        self.batch_encrypt_btn.clicked.connect(lambda: self.batch_process('encrypt'))
        self.batch_encrypt_btn.setEnabled(False)
        self.batch_encrypt_btn.setStyleSheet("background-color: #3498db; color: white;")
        
        self.batch_decrypt_btn = QPushButton("🔓 批量解混淆")
        self.batch_decrypt_btn.clicked.connect(lambda: self.batch_process('decrypt'))
        self.batch_decrypt_btn.setEnabled(False)
        self.batch_decrypt_btn.setStyleSheet("background-color: #2ecc71; color: white;")
        
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
        self.batch_download_btn.setStyleSheet("background-color: #e67e22; color: white;")
        
        self.zip_download_btn = QPushButton("📦 打包下载 (ZIP)")
        self.zip_download_btn.clicked.connect(self.zip_download)
        self.zip_download_btn.setEnabled(False)
        self.zip_download_btn.setStyleSheet("background-color: #1abc9c; color: white;")
        
        download_btn_layout.addWidget(self.download_btn)
        download_btn_layout.addWidget(self.batch_download_btn)
        download_btn_layout.addWidget(self.zip_download_btn)
        action_layout.addLayout(download_btn_layout)
        
        action_group.setLayout(action_layout)
        control_layout.addWidget(action_group)
        
        # 性能信息
        perf_group = QGroupBox("📊 性能信息")
        perf_layout = QVBoxLayout()
        
        self.perf_label = QLabel("就绪")
        self.perf_label.setStyleSheet("color: #2c3e50; font-size: 11px;")
        perf_layout.addWidget(self.perf_label)
        
        perf_group.setLayout(perf_layout)
        control_layout.addWidget(perf_group)
        
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
        preview_group = QGroupBox("🖼️ 图片预览")
        preview_inner_layout = QVBoxLayout()
        
        # 导航控制
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一张")
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.prev_btn.setEnabled(False)
        
        self.page_label = QLabel("1 / 1")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        
        self.next_btn = QPushButton("下一张 ▶")
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.page_label)
        nav_layout.addWidget(self.next_btn)
        preview_inner_layout.addLayout(nav_layout)
        
        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #e6f3ff;
                border: 2px dashed #F0F8FF;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        self.image_label.setText("图片预览区域\n\n拖拽图片文件到左侧区域或点击上传按钮")
        self.image_label.setAlignment(Qt.AlignCenter)
        preview_inner_layout.addWidget(self.image_label)
        
        # 图片信息
        info_group = QGroupBox("📋 图片信息")
        info_layout = QVBoxLayout()
        
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setRowCount(6)
        self.info_table.setHorizontalHeaderLabels(["属性", "值"])
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.info_table.horizontalHeader().setStretchLastSection(True)
        
        info_items = [
            ("状态", "等待上传"),
            ("文件名", "-"),
            ("尺寸", "-"),
            ("原图大小", "-"),
            ("处理后大小", "-"),
            ("处理时间", "-")
        ]
        
        for i, (key, value) in enumerate(info_items):
            self.info_table.setItem(i, 0, QTableWidgetItem(key))
            self.info_table.setItem(i, 1, QTableWidgetItem(value))
            self.info_table.item(i, 0).setForeground(QColor("#3498db"))
        
        info_layout.addWidget(self.info_table)
        info_group.setLayout(info_layout)
        preview_inner_layout.addWidget(info_group)
        
        preview_group.setLayout(preview_inner_layout)
        preview_layout.addWidget(preview_group)
        
        splitter.addWidget(preview_panel)
        splitter.setSizes([500, 900])
        
        main_layout.addWidget(splitter)
        
        # 添加日志窗口
        self.log_window = LogWindow(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_window)
        self.log_window = LogWindow(self)
        if hasattr(self.log_window, 'log_text'):  # 简单检查是否成功创建
            self.addDockWidget(Qt.BottomDockWidgetArea, self.log_window)
        else:
            logger.error("日志窗口创建失败")
            self.log_window = None  # 设置为None避免后续调用出错
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_label = QLabel("就绪 | CPU核心数: {} | 内存缓存: 启用".format(cpu_count()))
        self.status_bar.addPermanentWidget(self.status_label)
        self.status_bar.showMessage("欢迎使用无损图片混淆工具")
        
        # 菜单栏
        self.create_menu_bar()
        
        # 启用拖放
        self.setAcceptDrops(True)
        self.drag_drop_label.setAcceptDrops(True)
        
        # 定时器用于更新UI
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(200)  # 200ms更新一次
        
        # 初始化缓存信息
        self.update_cache_info()
        
        logger.info("用户界面初始化完成")
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("📁 文件")
        
        open_action = file_menu.addAction("📂 打开图片")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.select_images)
        
        file_menu.addSeparator()
        
        save_action = file_menu.addAction("💾 保存当前图片")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.download_current)
        
        save_all_action = file_menu.addAction("💾 保存所有图片")
        save_all_action.setShortcut("Ctrl+Shift+S")
        save_all_action.triggered.connect(self.batch_download)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("🚪 退出")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("✏️ 编辑")
        
        clear_action = edit_menu.addAction("🗑️ 清空列表")
        clear_action.triggered.connect(self.clear_images)
        
        edit_menu.addSeparator()
        
        settings_action = edit_menu.addAction("⚙️ 程序设置")
        settings_action.triggered.connect(self.open_settings)
        
        # 处理菜单
        process_menu = menubar.addMenu("🔄 处理")
        
        encrypt_action = process_menu.addAction("🔒 混淆当前图片")
        encrypt_action.setShortcut("Ctrl+E")
        encrypt_action.triggered.connect(self.encrypt_current)
        
        decrypt_action = process_menu.addAction("🔓 解混淆当前图片")
        decrypt_action.setShortcut("Ctrl+D")
        decrypt_action.triggered.connect(self.decrypt_current)
        
        process_menu.addSeparator()
        
        batch_encrypt_action = process_menu.addAction("🔒 批量混淆")
        batch_encrypt_action.setShortcut("Ctrl+Shift+E")
        batch_encrypt_action.triggered.connect(lambda: self.batch_process('encrypt'))
        
        batch_decrypt_action = process_menu.addAction("🔓 批量解混淆")
        batch_decrypt_action.setShortcut("Ctrl+Shift+D")
        batch_decrypt_action.triggered.connect(lambda: self.batch_process('decrypt'))
        
        # 视图菜单
        view_menu = menubar.addMenu("👁️ 视图")
        
        toggle_log_action = view_menu.addAction("📝 显示/隐藏日志窗口")
        toggle_log_action.setShortcut("Ctrl+L")
        toggle_log_action.triggered.connect(self.toggle_log_window)
        
        # 帮助菜单
        help_menu = menubar.addMenu("❓ 帮助")
        
        about_action = help_menu.addAction("ℹ️ 关于")
        about_action.triggered.connect(self.show_about)
        
        docs_action = help_menu.addAction("📚 使用说明")
        docs_action.triggered.connect(self.show_documentation)
        
        help_menu.addSeparator()
        
        view_log_action = help_menu.addAction("📋 查看日志文件")
        view_log_action.triggered.connect(self.view_log_file)
    
    def create_icon(self):
        """创建应用图标（简单实现）"""
        from PyQt5.QtGui import QPainter, QPen, QBrush
        from PyQt5.QtCore import QRect
        
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.setBrush(QBrush(QColor(52, 152, 219)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 4, 56, 56)
        
        # 绘制锁图标
        painter.setBrush(QBrush(Qt.white))
        painter.drawRect(20, 25, 24, 20)
        painter.drawEllipse(26, 15, 12, 12)
        
        # 绘制曲线
        painter.setPen(QPen(Qt.white, 2))
        for i in range(8):
            x = 10 + i * 6
            y = 40 + int(10 * np.sin(i * 0.8))
            if i > 0:
                painter.drawLine(old_x, old_y, x, y)
            old_x, old_y = x, y
        
        painter.end()
        return pixmap
    
    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """拖放事件"""
        urls = event.mimeData().urls()
        file_paths = []
        
        for url in urls:
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                # 检查是否为图片文件
                try:
                    Image.open(file_path)
                    file_paths.append(file_path)
                except:
                    pass
        
        if file_paths:
            self.add_images(file_paths)
            event.acceptProposedAction()
        else:
            self.show_error_dialog("无效文件", "拖放的文件不是有效的图片文件")
            event.ignore()
    
    # ============= 设置管理 =============
    def load_settings(self) -> Dict:
        """加载设置"""
        settings_file = Path("settings.json")
        default_settings = {
            'log_level': 'INFO',
            'max_processes': max(1, cpu_count() - 1),
            'cache_size': 100,
            'cache_ttl': 3600,
            'auto_save_log': True,
            'window_geometry': None
        }
        
        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    default_settings.update(loaded_settings)
                logger.info("设置已从文件加载")
            except Exception as e:
                logger.error(f"加载设置失败: {e}")
        
        return default_settings
    
    def save_settings(self):
        """保存设置"""
        settings_file = Path("settings.json")
        
        # 更新当前设置
        self.settings['window_geometry'] = {
            'x': self.x(),
            'y': self.y(),
            'width': self.width(),
            'height': self.height()
        }
        
        try:
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            logger.debug("设置已保存到文件")
        except Exception as e:
            logger.error(f"保存设置失败: {e}")
    
    def open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self)
        
        # 设置当前值
        dialog.log_level_combo.setCurrentText(self.settings.get('log_level', 'INFO'))
        dialog.max_process_spin.setValue(self.settings.get('max_processes', max(1, cpu_count() - 1)))
        dialog.cache_size_spin.setValue(self.settings.get('cache_size', 100))
        dialog.cache_ttl_spin.setValue(self.settings.get('cache_ttl', 3600))
        dialog.auto_save_check.setChecked(self.settings.get('auto_save_log', True))
        
        if dialog.exec_() == QDialog.Accepted:
            new_settings = dialog.get_settings()
            
            # 更新设置
            self.settings.update(new_settings)
            
            # 更新日志级别
            log_level = getattr(logging, new_settings['log_level'])
            logger.setLevel(log_level)
            for handler in logger.handlers:
                handler.setLevel(log_level)
            
            # 保存设置
            self.save_settings()
            
            logger.info(f"设置已更新: {new_settings}")
            QMessageBox.information(self, "设置已保存", "程序设置已保存并生效")
    
    # ============= 错误处理 =============
    @handle_exceptions
    def show_error_dialog(self, title: str, message: str, detailed: str = None):
        """显示错误对话框"""
        logger.error(f"{title}: {message}")
        
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle(f"错误 - {title}")
        error_dialog.setText(message)
        
        if detailed:
            error_dialog.setDetailedText(detailed)
        
        error_dialog.exec_()
    
    def on_log_message(self, message: str, level: int):
        """处理日志消息"""
        if hasattr(self, 'log_window') and self.log_window is not None:
            self.log_window.add_log_message(message, level)
        else:
            # 如果日志窗口不可用，至少打印到控制台
            print(f"[{logging.getLevelName(level)}] {message}")
    
    # ============= 图片管理 =============
    @handle_exceptions
    def select_images(self, checked=False):
        """选择图片文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片文件", "",
            "图片文件 (*.png *.jpg *.jpeg *.gif *.bmp *.webp *.tiff);;所有文件 (*.*)"
        )
        
        if files:
            logger.info(f"选择了 {len(files)} 个图片文件")
            self.add_images(files)
    
    @handle_exceptions
    def add_images(self, file_paths: List[str]):
        """添加图片到列表"""
        added_count = 0
        
        for file_path in file_paths:
            try:
                # 检查文件是否已存在
                existing_paths = [img.path for img in self.images]
                if file_path in existing_paths:
                    logger.warning(f"图片已存在: {file_path}")
                    continue
                
                # 获取图片信息
                with Image.open(file_path) as img:
                    width, height = img.size
                    format = img.format or os.path.splitext(file_path)[1][1:].upper()
                
                file_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
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
                self.image_table.setItem(row, 5, QTableWidgetItem("-"))
                
                added_count += 1
                logger.debug(f"添加图片: {os.path.basename(file_path)} ({width}x{height}, {format})")
                
            except Exception as e:
                logger.error(f"加载图片失败 {file_path}: {e}")
                self.show_error_dialog("加载图片失败", f"无法加载图片: {os.path.basename(file_path)}\n错误: {str(e)}")
        
        if added_count > 0:
            logger.info(f"成功添加 {added_count} 张图片")
            
            self.clear_btn.setEnabled(True)
            self.batch_encrypt_btn.setEnabled(True)
            self.batch_decrypt_btn.setEnabled(True)
            
            # 显示第一张图片
            self.show_image(0)
            
            self.status_bar.showMessage(f"已添加 {added_count} 张图片")
    
    @handle_exceptions
    def clear_images(self):
        """清空图片列表"""
        if self.images:
            reply = QMessageBox.question(
                self, "确认清空",
                f"确定要清空所有 {len(self.images)} 张图片吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 释放内存
                self.images.clear()
                self.processed_images.clear()
                
                # 清空表格
                self.image_table.setRowCount(0)
                
                # 重置状态
                self.clear_btn.setEnabled(False)
                self.batch_encrypt_btn.setEnabled(False)
                self.batch_decrypt_btn.setEnabled(False)
                self.encrypt_btn.setEnabled(False)
                self.decrypt_btn.setEnabled(False)
                self.download_btn.setEnabled(False)
                self.batch_download_btn.setEnabled(False)
                self.zip_download_btn.setEnabled(False)
                
                # 重置预览
                self.image_label.setText("图片预览区域\n\n拖拽图片文件到左侧区域或点击上传按钮")
                self.update_info_table()
                self.update_navigation()
                
                logger.info("已清空所有图片")
                self.status_bar.showMessage("图片列表已清空")
    
    @handle_exceptions
    def on_image_double_clicked(self, index):
        """双击图片事件"""
        row = index.row()
        if 0 <= row < len(self.images):
            self.show_image(row)
    
    # ============= 图片显示 =============
    @handle_exceptions
    def show_image(self, index: int):
        """显示指定索引的图片"""
        if 0 <= index < len(self.images):
            self.current_image_index = index
            
            # 更新导航
            self.update_navigation()
            
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
                        self.image_label.size() * 0.8,
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
            
            logger.debug(f"显示图片 {index+1}/{len(self.images)}")
    
    def display_numpy_image(self, image_array: np.ndarray):
        """显示numpy数组图片"""
        try:
            height, width, channel = image_array.shape
            
            if channel == 4:
                qimage = QImage(image_array.data, width, height, width * 4, QImage.Format_RGBA8888)
            elif channel == 3:
                qimage = QImage(image_array.data, width, height, width * 3, QImage.Format_RGB888)
            else:
                # 转换为RGB
                img = Image.fromarray(image_array)
                img = img.convert("RGB")
                image_array = np.array(img)
                qimage = QImage(image_array.data, width, height, width * 3, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size() * 0.8,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            logger.error(f"显示图片失败: {e}")
            self.image_label.setText("显示图片失败")
    
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
                self.info_table.item(4, 1).setText("已处理")
            else:
                self.info_table.item(4, 1).setText("-")
            
            # 处理时间
            processed_time = self.image_table.item(self.current_image_index, 5)
            if processed_time:
                self.info_table.item(5, 1).setText(processed_time.text())
    
    def update_navigation(self):
        """更新导航控件"""
        if len(self.images) <= 1:
            self.page_label.setText("1 / 1")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
        
        self.page_label.setText(f"{self.current_image_index + 1} / {len(self.images)}")
        self.prev_btn.setEnabled(self.current_image_index > 0)
        self.next_btn.setEnabled(self.current_image_index < len(self.images) - 1)
    
    def show_prev_image(self, checked=False):
        """显示上一张图片"""
        if self.current_image_index > 0:
            self.show_image(self.current_image_index - 1)
    
    def show_next_image(self, checked=False):
        """显示下一张图片"""
        if self.current_image_index < len(self.images) - 1:
            self.show_image(self.current_image_index + 1)
    
    def restore_current(self, checked=False):
        """还原当前图片为原图"""
        if 0 <= self.current_image_index < len(self.images):
            self.processed_images[self.current_image_index] = None
            self.show_image(self.current_image_index)
            
            # 更新表格状态
            self.image_table.item(self.current_image_index, 4).setText("已还原")
            self.image_table.item(self.current_image_index, 5).setText("-")
            
            logger.info(f"图片 {self.current_image_index+1} 已还原为原图")
            self.status_bar.showMessage("图片已还原")
    
    # ============= 图片处理 =============
    @handle_exceptions
    def encrypt_current(self, checked=False):
        """混淆当前图片"""
        if self.current_image_index < len(self.images):
            self.process_single_image(self.current_image_index, 'encrypt')
    
    @handle_exceptions
    def decrypt_current(self, checked=False):
        """解混淆当前图片"""
        if self.current_image_index < len(self.images):
            self.process_single_image(self.current_image_index, 'decrypt')
    
    @handle_exceptions
    def process_single_image(self, index: int, mode: str):
        """处理单张图片"""
        if index >= len(self.images):
            return
        
        logger.info(f"开始处理单张图片: 索引={index}, 模式={mode}")
        
        # 创建临时工作线程
        self.worker = ProcessWorker(self.cache_manager)
        self.worker.setup_batch([self.images[index]], mode, 
                               self.format_combo.currentText(),
                               self.quality_slider.value())
        
        # 连接信号
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.image_processed.connect(self.on_image_processed)
        self.worker.batch_finished.connect(self.on_batch_finished)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.log_message.connect(self.on_log_message)
        
        self.worker.start()
        
        # 禁用按钮
        self.set_buttons_enabled(False)
    
    @handle_exceptions
    def batch_process(self, mode: str):
        """批量处理图片"""
        if not self.images:
            QMessageBox.warning(self, "无图片", "请先上传图片")
            return
        
        # 确认对话框
        reply = QMessageBox.question(
            self, "确认批量处理",
            f"确定要批量{ '混淆' if mode == 'encrypt' else '解混淆' } {len(self.images)} 张图片吗？\n"
            f"这将使用 {self.settings.get('max_processes', cpu_count()-1)} 个CPU核心并行处理。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        logger.info(f"开始批量处理: 模式={mode}, 图片数={len(self.images)}")
        
        # 创建批处理工作线程
        self.worker = ProcessWorker(self.cache_manager)
        self.worker.setup_batch(self.images, mode, 
                               self.format_combo.currentText(),
                               self.quality_slider.value())
        
        # 连接信号
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.image_processed.connect(self.on_image_processed)
        self.worker.batch_finished.connect(self.on_batch_finished)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.log_message.connect(self.on_log_message)
        
        self.worker.start()
        
        # 禁用按钮
        self.set_buttons_enabled(False)
    
    def on_progress_updated(self, progress: int, status: str):
        """处理进度更新"""
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{status} - {progress}%")
        self.status_bar.showMessage(status)
    
    def on_image_processed(self, index: int, image_array: np.ndarray, time_cost: float, cache_hit: bool):
        """单张图片处理完成"""
        try:
            self.processed_images[index] = image_array
            
            # 更新表格状态
            mode = "混淆" if self.worker and self.worker.mode == 'encrypt' else "解混淆"
            status_text = f"{mode}完成" + (" (缓存)" if cache_hit else "")
            self.image_table.item(index, 4).setText(status_text)
            self.image_table.item(index, 5).setText(f"{time_cost:.2f}s")
            
            # 如果是当前显示的图片，更新显示
            if index == self.current_image_index:
                self.show_image(index)
            
            # 启用批量下载按钮
            if any(img is not None for img in self.processed_images):
                self.batch_download_btn.setEnabled(True)
                self.zip_download_btn.setEnabled(True)
            
        except Exception as e:
            logger.error(f"更新图片处理结果失败: {e}")
    
    def on_batch_finished(self, metrics: PerformanceMetrics):
        """批量处理完成"""
        self.set_buttons_enabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("批量处理完成")
        
        # 更新性能信息
        self.performance_metrics = metrics
        self.update_performance_info()
        
        # 显示完成消息
        success_msg = (f"批量处理完成！\n\n"
                      f"处理图片: {metrics.images_processed}张\n"
                      f"总耗时: {metrics.total_time:.2f}秒\n"
                      f"平均每张: {metrics.avg_time_per_image:.2f}秒\n"
                      f"缓存命中: {metrics.cache_hits}次\n"
                      f"使用核心: {metrics.cpu_cores_used}个")
        
        QMessageBox.information(self, "处理完成", success_msg)
        self.status_bar.showMessage("批量处理完成")
        
        logger.info(f"批量处理完成: {metrics}")
    
    def on_worker_error(self, title: str, message: str):
        """处理工作线程错误"""
        self.set_buttons_enabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("处理出错")
        
        self.show_error_dialog(title, message)
    
    def set_buttons_enabled(self, enabled: bool):
        """设置按钮启用状态"""
        has_images = bool(self.images)
        has_processed = any(img is not None for img in self.processed_images)
        
        self.encrypt_btn.setEnabled(enabled and has_images)
        self.decrypt_btn.setEnabled(enabled and has_images)
        self.restore_btn.setEnabled(enabled and has_images)
        self.batch_encrypt_btn.setEnabled(enabled and has_images)
        self.batch_decrypt_btn.setEnabled(enabled and has_images)
        self.download_btn.setEnabled(enabled and self.processed_images[self.current_image_index] is not None)
        self.batch_download_btn.setEnabled(enabled and has_processed)
        self.zip_download_btn.setEnabled(enabled and has_processed)
        self.clear_btn.setEnabled(enabled and has_images)
    
    # ============= 下载功能 =============
    @handle_exceptions
    def download_current(self, checked=False):
        """下载当前图片"""
        if self.current_image_index < len(self.processed_images):
            processed_img = self.processed_images[self.current_image_index]
            if processed_img is not None:
                self.save_image(processed_img, self.current_image_index)
            else:
                QMessageBox.warning(self, "无处理结果", "当前图片尚未处理，无法下载")
    
    @handle_exceptions
    def batch_download(self, checked=False):
        """批量下载"""
        if not any(img is not None for img in self.processed_images):
            QMessageBox.warning(self, "无处理结果", "没有已处理的图片可以下载")
            return
        
        # 选择保存目录
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not save_dir:
            return
        
        logger.info(f"开始批量下载到目录: {save_dir}")
        
        # 进度条重置
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("批量下载中...")
        
        saved_count = 0
        for i, img_array in enumerate(self.processed_images):
            if img_array is not None:
                try:
                    self.save_image_to_path(img_array, i, save_dir)
                    saved_count += 1
                    
                    # 更新进度
                    progress = int((i + 1) / len(self.processed_images) * 100)
                    self.progress_bar.setValue(progress)
                    self.progress_bar.setFormat(f"批量下载中... {i+1}/{len(self.processed_images)}")
                    
                except Exception as e:
                    logger.error(f"保存图片 {i+1} 失败: {e}")
        
        # 重置进度条
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("批量下载完成")
        
        logger.info(f"批量下载完成: 共保存 {saved_count} 张图片")
        QMessageBox.information(self, "下载完成", f"已保存 {saved_count} 张图片到:\n{save_dir}")
    
    @handle_exceptions
    def zip_download(self, checked=False):
        """打包下载"""
        if not any(img is not None for img in self.processed_images):
            QMessageBox.warning(self, "无处理结果", "没有已处理的图片可以打包")
            return
        
        # 提示需要zipfile库
        try:
            import zipfile
        except ImportError:
            QMessageBox.warning(self, "缺少依赖", "打包功能需要zipfile库，请确保已安装")
            return
        
        # 选择保存文件
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存ZIP文件", 
            f"processed_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            "ZIP文件 (*.zip)"
        )
        
        if not file_path:
            return
        
        logger.info(f"开始创建ZIP文件: {file_path}")
        
        # 显示进度
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("创建ZIP文件中...")
        
        try:
            import zipfile
            
            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for i, img_array in enumerate(self.processed_images):
                    if img_array is not None:
                        # 创建临时文件
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            # 保存图片到临时文件
                            img = Image.fromarray(img_array)
                            img.save(tmp.name, "PNG")
                            
                            # 添加到ZIP
                            zipf.write(tmp.name, f"image_{i+1}.png")
                            
                            # 删除临时文件
                            os.unlink(tmp.name)
                        
                        # 更新进度
                        progress = int((i + 1) / len(self.processed_images) * 100)
                        self.progress_bar.setValue(progress)
                        self.progress_bar.setFormat(f"创建ZIP文件中... {i+1}/{len(self.processed_images)}")
            
            # 重置进度条
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("ZIP文件创建完成")
            
            logger.info(f"ZIP文件创建成功: {file_path}")
            
            # 询问是否打开文件
            reply = QMessageBox.question(
                self, "ZIP文件创建完成",
                f"ZIP文件已创建成功！\n\n文件: {file_path}\n\n是否打开文件所在目录？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(file_path)))
                
        except Exception as e:
            logger.error(f"创建ZIP文件失败: {e}")
            self.show_error_dialog("创建ZIP文件失败", str(e))
    
    def save_image(self, image_array: np.ndarray, index: int):
        """保存图片"""
        # 获取默认文件名
        img_info = self.images[index]
        base_name = os.path.splitext(os.path.basename(img_info.path))[0]
        mode = "encrypted" if self.worker and self.worker.mode == 'encrypt' else "decrypted"
        
        # 根据格式确定扩展名
        format_text = self.format_combo.currentText()
        if "PNG" in format_text:
            ext = "png"
        elif "JPEG" in format_text:
            ext = "jpg"
        elif "WebP" in format_text:
            ext = "webp"
        else:
            ext = "png"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", 
            f"{base_name}_{mode}.{ext}",
            f"图片文件 (*.{ext})"
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
        
        try:
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
            
            logger.info(f"图片保存成功: {save_path} ({file_size/1024:.1f} KB)")
            
            # 显示成功消息
            self.status_bar.showMessage(f"图片已保存: {filename}")
            
        except Exception as e:
            logger.error(f"保存图片失败 {save_path}: {e}")
            raise
    
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
    
    # ============= 缓存管理 =============
    def update_cache_info(self):
        """更新缓存信息"""
        stats = self.cache_manager.get_stats()
        info_text = f"缓存: {stats['curve_cache_size']}曲线, {stats['result_cache_size']}结果"
        self.cache_info_label.setText(info_text)
    
    def clear_cache(self, clear_disk: bool):
        """清空缓存"""
        # 确认对话框
        cache_type = "内存和磁盘" if clear_disk else "内存"
        reply = QMessageBox.question(
            self, "确认清空缓存",
            f"确定要清空{cache_type}缓存吗？\n"
            f"这将删除所有缓存的曲线映射和处理结果。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.cache_manager.clear(clear_disk)
            self.update_cache_info()
            
            logger.info(f"已清空{cache_type}缓存")
            QMessageBox.information(self, "缓存已清空", f"{cache_type}缓存已清空")
    
    # ============= 性能信息 =============
    def update_performance_info(self):
        """更新性能信息"""
        if self.performance_metrics.images_processed > 0:
            info_text = (f"性能统计: "
                        f"处理 {self.performance_metrics.images_processed} 张图片, "
                        f"总耗时 {self.performance_metrics.total_time:.2f}s, "
                        f"平均 {self.performance_metrics.avg_time_per_image:.2f}s/张, "
                        f"缓存命中 {self.performance_metrics.cache_hits} 次")
            self.perf_label.setText(info_text)
    
    # ============= UI事件处理 =============
    def on_format_changed(self, index: int):
        """格式改变事件"""
        if index == 0:  # PNG
            self.quality_slider.setEnabled(False)
            self.quality_label.setText("100% (无损)")
        else:
            self.quality_slider.setEnabled(True)
            self.on_quality_changed(self.quality_slider.value())
    
    def on_quality_changed(self, value: int):
        """质量滑块改变事件"""
        self.quality_label.setText(f"{value}%")
        
        # 更新提示
        if value >= 90:
            hint = "高质量 (文件较大)"
        elif value >= 70:
            hint = "平衡质量与大小"
        elif value >= 50:
            hint = "中等压缩 (文件较小)"
        else:
            hint = "高压缩 (文件很小，质量较低)"
        
        self.status_bar.showMessage(f"压缩质量: {value}% - {hint}", 3000)
    
    def toggle_log_window(self):
        """显示/隐藏日志窗口"""
        if self.log_window.isVisible():
            self.log_window.hide()
        else:
            self.log_window.show()
    
    def view_log_file(self):
        """查看日志文件"""
        if os.path.exists(log_file):
            QDesktopServices.openUrl(QUrl.fromLocalFile(log_file))
        else:
            QMessageBox.information(self, "日志文件", f"日志文件路径: {log_file}")
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h2>无损图片混淆工具 - 高级版</h2>
        <p>基于空间填充曲线的图片混淆技术，支持多核并行处理和批量操作。</p>
        
        <h3>主要特性:</h3>
        <ul>
        <li>基于Gilbert空间填充曲线的无损图片混淆</li>
        <li>多核并行处理，充分利用CPU性能</li>
        <li>智能缓存系统，提高处理速度</li>
        <li>完整的日志记录和错误处理</li>
        <li>批量图片上传、处理和下载</li>
        <li>支持多种输出格式 (PNG, JPEG, WebP)</li>
        </ul>
        
        <h3>技术栈:</h3>
        <ul>
        <li>Python 3.7+</li>
        <li>PyQt5 - 图形界面</li>
        <li>Numba - JIT编译和并行计算</li>
        <li>Pillow - 图片处理</li>
        <li>NumPy - 数值计算</li>
        </ul>
        
        <p>版本: 2.0.0 | 开发者: AI Assistant</p>
        """
        
        QMessageBox.about(self, "关于", about_text)
    
    def show_documentation(self):
        """显示使用说明"""
        docs_text = """
        <h2>使用说明</h2>
        
        <h3>1. 上传图片</h3>
        <ul>
        <li>点击"选择图片"按钮选择图片文件</li>
        <li>或直接拖拽图片文件到窗口</li>
        <li>支持批量上传多张图片</li>
        <li>支持格式: PNG, JPEG, GIF, BMP, WebP, TIFF</li>
        </ul>
        
        <h3>2. 图片处理</h3>
        <ul>
        <li><b>混淆:</b> 对图片进行加密处理</li>
        <li><b>解混淆:</b> 对已加密的图片进行解密</li>
        <li><b>批量处理:</b> 一次性处理所有图片</li>
        <li><b>还原:</b> 将图片恢复为原始状态</li>
        </ul>
        
        <h3>3. 输出设置</h3>
        <ul>
        <li><b>输出格式:</b> 选择保存图片的格式</li>
        <li><b>压缩质量:</b> 调整JPEG/WebP的压缩质量</li>
        <li>PNG格式使用无损压缩</li>
        </ul>
        
        <h3>4. 下载图片</h3>
        <ul>
        <li><b>下载当前图片:</b> 保存当前显示的图片</li>
        <li><b>批量下载:</b> 保存所有已处理的图片</li>
        <li><b>打包下载:</b> 将所有图片打包为ZIP文件</li>
        </ul>
        
        <h3>5. 缓存系统</h3>
        <ul>
        <li>自动缓存曲线映射，避免重复计算</li>
        <li>缓存处理结果，提高重复处理速度</li>
        <li>可手动清空缓存释放内存</li>
        </ul>
        
        <h3>6. 多核处理</h3>
        <ul>
        <li>自动检测CPU核心数</li>
        <li>使用多进程并行处理图片</li>
        <li>可在设置中调整最大进程数</li>
        </ul>
        
        <h3>快捷键:</h3>
        <ul>
        <li>Ctrl+O: 打开图片</li>
        <li>Ctrl+S: 保存当前图片</li>
        <li>Ctrl+E: 混淆当前图片</li>
        <li>Ctrl+D: 解混淆当前图片</li>
        <li>Ctrl+L: 显示/隐藏日志窗口</li>
        <li>Ctrl+Q: 退出程序</li>
        </ul>
        """
        
        dialog = QDialog(self)
        dialog.setWindowTitle("使用说明")
        dialog.resize(600, 700)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(docs_text)
        
        layout.addWidget(text_edit)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        
        dialog.exec_()
    
    def update_ui(self):
        """定时更新UI"""
        # 更新状态栏信息
        cpu_count = mp.cpu_count()
        memory_info = self.cache_manager.get_stats()
        status_text = f"就绪 | CPU核心数: {cpu_count} | 内存缓存: {memory_info['curve_cache_size']}曲线, {memory_info['result_cache_size']}结果"
        self.status_label.setText(status_text)
    
    @handle_exceptions
    def closeEvent(self, event):
        """关闭事件"""
        # 停止工作线程
        if self.worker and self.worker.isRunning():
            logger.info("正在停止工作线程...")
            self.worker.stop()
            self.worker.wait()
        
        # 保存缓存到磁盘
        logger.info("正在保存缓存到磁盘...")
        self.cache_manager.save_to_disk()
        
        # 保存设置
        logger.info("正在保存程序设置...")
        self.save_settings()
        
        # 清理资源
        logger.info("正在清理资源...")
        
        # 记录程序结束
        logger.info("应用程序正常退出")
        
        event.accept()

# ============= 主程序入口 =============
@handle_exceptions
def main():
    """主函数"""
    # 设置多进程启动方式（Windows需要）
    mp.freeze_support()
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("无损图片混淆工具 - 高级版")
    app.setApplicationVersion("2.0.0")
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
    
    # 预热Numba JIT编译器
    logger.info("预热Numba JIT编译器...")
    try:
        test_array = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8).flatten()
        test_map = build_gilbert_curve_map_numba(100, 100)
        apply_curve_mapping_numba(test_array, test_map, 'encrypt', 1000)
        logger.info("Numba JIT预热完成")
    except Exception as e:
        logger.warning(f"Numba预热失败: {e}")
    
    # 创建并显示主窗口
    logger.info("创建主窗口...")
    window = ImageObfuscatorGUI()
    
    # 恢复窗口位置和大小
    if window.settings.get('window_geometry'):
        geo = window.settings['window_geometry']
        window.setGeometry(geo['x'], geo['y'], geo['width'], geo['height'])
    
    window.show()
    logger.info("应用程序启动完成")
    
    # 运行应用
    exit_code = app.exec_()
    logger.info(f"应用程序退出，代码: {exit_code}")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
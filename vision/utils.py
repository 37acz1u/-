"""视觉系统的工具函数"""

import cv2
import numpy as np
from typing import Tuple

def resize_image(image: np.ndarray, width: int = None, height: int = None,
                inter: int = cv2.INTER_AREA) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        image: 输入图像
        width: 目标宽度
        height: 目标高度
        inter: 插值方法
    
    Returns:
        np.ndarray: 调整大小后的图像
    """
    dim = None
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    归一化图像到 [0, 1] 范围
    
    Args:
        image: 输入图像 (像素值 0-255)
    
    Returns:
        np.ndarray: 归一化后的图像
    """
    return image.astype('float32') / 255.0

def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    直方图均衡化，增强对比度
    
    Args:
        image: 输入图像
    
    Returns:
        np.ndarray: 增强后的图像
    """
    if len(image.shape) == 3:
        # 彩色图像
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    else:
        # 灰度图像
        return cv2.equalizeHist(image)

def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                      sigma: float = 0.0) -> np.ndarray:
    """
    应用高斯模糊
    
    Args:
        image: 输入图像
        kernel_size: 核大小
        sigma: 标准差
    
    Returns:
        np.ndarray: 模糊后的图像
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def draw_emotion_on_image(image: np.ndarray, emotion: str, 
                         position: Tuple[int, int] = (10, 30),
                         font_scale: float = 1.0,
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制识别的情绪
    
    Args:
        image: 输入图像
        emotion: 情绪文本
        position: 文字位置
        font_scale: 字体大小
        color: 颜色 (BGR)
        thickness: 字体厚度
    
    Returns:
        np.ndarray: 标注后的图像
    """
    image_copy = image.copy()
    cv2.putText(image_copy, f"Emotion: {emotion}", position,
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return image_copy

def draw_landmarks(image: np.ndarray, landmarks: list,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  radius: int = 2) -> np.ndarray:
    """
    在图像上绘制面部关键点
    
    Args:
        image: 输入图像
        landmarks: 关键点列表 [(x, y), ...]
        color: 点的颜色 (BGR)
        radius: 点的半径
    
    Returns:
        np.ndarray: 标注后的图像
    """
    image_copy = image.copy()
    
    for (x, y) in landmarks:
        cv2.circle(image_copy, (int(x), int(y)), radius, color, -1)
    
    return image_copy

def crop_face_with_margin(image: np.ndarray, bbox: Tuple[int, int, int, int],
                         margin: float = 0.2) -> np.ndarray:
    """
    裁剪人脸区域（带边距）
    
    Args:
        image: 输入图像
        bbox: 边界框 (x, y, w, h)
        margin: 边距比例 (0-1)
    
    Returns:
        np.ndarray: 裁剪后的人脸
    """
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    
    # 计算边距
    x_margin = int(w * margin)
    y_margin = int(h * margin)
    
    # 计算新的坐标
    x1 = max(0, x - x_margin)
    y1 = max(0, y - y_margin)
    x2 = min(w_img, x + w + x_margin)
    y2 = min(h_img, y + h + y_margin)
    
    return image[y1:y2, x1:x2]
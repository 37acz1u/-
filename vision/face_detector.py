"""人脸检测模块"""

import cv2
import dlib
import numpy as np
from typing import List, Tuple
import os

class FaceDetector:
    """人脸检测器 - 使用 dlib 和 OpenCV 进行人脸检测"""
    
    def __init__(self, model_type: str = 'dlib', detection_threshold: float = 0.5):
        """
        初始化人脸检测器
        
        Args:
            model_type: 检测模型类型 ('dlib' 或 'opencv')
            detection_threshold: 检测阈值
        """
        self.model_type = model_type
        self.detection_threshold = detection_threshold
        
        if model_type == 'dlib':
            # 初始化 dlib 人脸检测器
            self.detector = dlib.get_frontal_face_detector()
            # 初始化 dlib 关键点检测器（68个面部关键点）
            script_dir = os.path.dirname(os.path.abspath(__file__))
            predictor_path = os.path.join(script_dir, '../models/shape_predictor_68_face_landmarks.dat')
            if os.path.exists(predictor_path):
                self.predictor = dlib.shape_predictor(predictor_path)
            else:
                self.predictor = None
                print("警告: 68点面部关键点模型未找到")
        
        elif model_type == 'opencv':
            # 初始化 OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, image_path: str = None, image: np.ndarray = None) -> List[dict]:
        """
        检测图像中的人脸
        
        Args:
            image_path: 图像文件路径
            image: numpy数组格式的图像
        
        Returns:
            List[dict]: 检测到的人脸列表，每个人脸包含:
                - 'bbox': (x, y, w, h) 人脸边界框
                - 'landmarks': 68个面部关键点 (如果可用)
                - 'confidence': 检测置信度
        """
        # 读取图像
        if image is None:
            if image_path is None:
                raise ValueError("必须提供 image_path 或 image")
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"无法读取图像: {image_path}")
        
        # 转换为RGB (如果是BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        faces = []
        
        if self.model_type == 'dlib':
            faces = self._detect_faces_dlib(image_rgb)
        elif self.model_type == 'opencv':
            faces = self._detect_faces_opencv(image_rgb)
        
        return faces
    
    def _detect_faces_dlib(self, image: np.ndarray) -> List[dict]:
        """
        使用 dlib 检测人脸
        """
        faces = []
        dlib_faces = self.detector(image, 1)  # 1表示图像上采样次数
        
        for face in dlib_faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            confidence = 1.0  # dlib 不提供置信度
            
            face_dict = {
                'bbox': (x, y, w, h),
                'confidence': confidence,
                'landmarks': None
            }
            
            # 获取面部关键点
            if self.predictor is not None:
                try:
                    landmarks = self.predictor(image, face)
                    landmarks_list = [(point.x, point.y) for point in landmarks.parts()]
                    face_dict['landmarks'] = landmarks_list
                except Exception as e:
                    print(f"关键点检测失败: {e}")
            
            faces.append(face_dict)
        
        return faces
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[dict]:
        """
        使用 OpenCV Haar Cascade 检测人脸
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        faces = []
        detected_faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in detected_faces:
            confidence = 1.0  # OpenCV Haar Cascade 也不提供置信度
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': confidence,
                'landmarks': None
            })
        
        return faces
    
    def extract_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        从图像中提取人脸区域
        
        Args:
            image: 原始图像
            bbox: (x, y, w, h) 人脸边界框
        
        Returns:
            np.ndarray: 提取的人脸图像
        """
        x, y, w, h = bbox
        face_region = image[y:y+h, x:x+w]
        return face_region
    
    def draw_faces_on_image(self, image: np.ndarray, faces: List[dict], 
                           show_landmarks: bool = False) -> np.ndarray:
        """
        在图像上绘制检测到的人脸
        
        Args:
            image: 原始图像
            faces: 检测到的人脸列表
            show_landmarks: 是否显示面部关键点
        
        Returns:
            np.ndarray: 标注后的图像
        """
        image_copy = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            # 绘制边界框
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 显示置信度
            confidence = face.get('confidence', 1.0)
            text = f"Face: {confidence:.2f}"
            cv2.putText(image_copy, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 绘制关键点
            if show_landmarks and face.get('landmarks') is not None:
                for (px, py) in face['landmarks']:
                    cv2.circle(image_copy, (px, py), 2, (255, 0, 0), -1)
        
        return image_copy
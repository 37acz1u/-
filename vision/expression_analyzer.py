"""表情分析模块 - 识别面部表情中的情绪"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import os

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
except ImportError:
    print("警告: TensorFlow/Keras 未安装")

class ExpressionAnalyzer:
    """表情分析器 - 识别面部表情和情绪"""
    
    # 情绪类别
    EMOTION_CLASSES = {
        0: 'angry',      # 愤怒
        1: 'disgusted',  # 厌恶
        2: 'afraid',     # 恐惧
        3: 'happy',      # 开心
        4: 'neutral',    # 中性
        5: 'sad',        # 悲伤
        6: 'surprised'   # 惊讶
    }
    
    # 情绪到汉语的映射
    EMOTION_CN = {
        'angry': '愤怒',
        'disgusted': '厌恶',
        'afraid': '恐惧',
        'happy': '开心',
        'neutral': '中性',
        'sad': '悲伤',
        'surprised': '惊讶'
    }
    
    def __init__(self, model_path: str = None, use_pretrained: bool = True):
        """
        初始化表情分析器
        
        Args:
            model_path: 预训练模型路径
            use_pretrained: 是否使用预训练模型
        """
        self.model = None
        self.input_size = (48, 48)
        
        if use_pretrained:
            self._load_pretrained_model(model_path)
        else:
            print("未加载预训练模型，需要自行提供模型")
    
    def _load_pretrained_model(self, model_path: str = None):
        """
        加载预训练的表情识别模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            if model_path is None:
                # 尝试从默认路径加载
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, '../models/fer2013_mini_XCEPTION.h5')
            
            if os.path.exists(model_path):
                self.model = load_model(model_path, compile=False)
                print(f"成功加载模型: {model_path}")
            else:
                print(f"模型文件未找到: {model_path}")
                print("将使用简单的表情分类器")
                self._create_simple_classifier()
        
        except Exception as e:
            print(f"加载模型失败: {e}")
            self._create_simple_classifier()
    
    def _create_simple_classifier(self):
        """
        创建一个简单的表情分类器（基于面部关键点）
        """
        self.use_simple_classifier = True
        print("使用基于面部关键点的简单分类器")
    
    def analyze_expression(self, face_image: np.ndarray, 
                          landmarks: List[Tuple[int, int]] = None) -> Dict[str, any]:
        """
        分析单个人脸的表情
        
        Args:
            face_image: 人脸图像 (numpy数组)
            landmarks: 面部关键点 (可选，用于增强分析)
        
        Returns:
            Dict: 包含以下信息的字典:
                - 'emotion': 识别的主要情绪
                - 'confidence': 置信度 (0-1)
                - 'scores': 所有情绪的得分
                - 'features': 提取的面部特征
        """
        result = {
            'emotion': None,
            'confidence': 0.0,
            'scores': {},
            'features': {}
        }
        
        # 预���理图像
        processed_image = self._preprocess_image(face_image)
        
        # 如果有模型，使用深度学习
        if self.model is not None:
            result = self._predict_with_model(processed_image, result)
        else:
            # 使用基于关键点的方法
            if landmarks is not None:
                result = self._predict_with_landmarks(landmarks, result)
            else:
                result = self._predict_with_simple_features(face_image, result)
        
        return result
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像
        
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 调整大小
        resized = cv2.resize(gray, self.input_size)
        
        # 归一化
        normalized = resized / 255.0
        
        return normalized
    
    def _predict_with_model(self, image: np.ndarray, result: Dict) -> Dict:
        """
        使用深度学习模型进行预测
        """
        try:
            # 扩展维度 (height, width) -> (height, width, 1) -> (1, height, width, 1)
            x = np.expand_dims(image, axis=-1)
            x = np.expand_dims(x, axis=0)
            
            # 预测
            prediction = self.model.predict(x, verbose=0)[0]
            
            # 获取结果
            emotion_idx = np.argmax(prediction)
            emotion = self.EMOTION_CLASSES[emotion_idx]
            confidence = float(prediction[emotion_idx])
            
            result['emotion'] = emotion
            result['confidence'] = confidence
            result['scores'] = {
                self.EMOTION_CLASSES[i]: float(prediction[i])
                for i in range(len(prediction))
            }
            
        except Exception as e:
            print(f"模型预测失败: {e}")
        
        return result
    
    def _predict_with_landmarks(self, landmarks: List[Tuple[int, int]], 
                               result: Dict) -> Dict:
        """
        基于面部关键点的表情识别
        
        Args:
            landmarks: 68个面部关键点
            result: 结果字典
        
        Returns:
            Dict: 更新后的结果字典
        """
        if len(landmarks) < 68:
            return result
        
        landmarks = np.array(landmarks)
        
        # 提取特征
        features = {}
        
        # 1. 眼睛特征
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        features['left_eye_height'] = self._calculate_eye_aspect_ratio(left_eye)
        features['right_eye_height'] = self._calculate_eye_aspect_ratio(right_eye)
        
        # 2. 嘴巴特征
        mouth = landmarks[48:68]
        features['mouth_height'] = np.max(mouth[:, 1]) - np.min(mouth[:, 1])
        features['mouth_width'] = np.max(mouth[:, 0]) - np.min(mouth[:, 0])
        mouth_aspect_ratio = features['mouth_height'] / (features['mouth_width'] + 1e-6)
        features['mouth_aspect_ratio'] = mouth_aspect_ratio
        
        # 3. 眉毛特征
        left_eyebrow = landmarks[17:22]
        right_eyebrow = landmarks[22:27]
        features['left_eyebrow_height'] = np.mean(left_eyebrow[:, 1])
        features['right_eyebrow_height'] = np.mean(right_eyebrow[:, 1])
        
        result['features'] = features
        
        # 根据特征推断情绪
        emotion, confidence = self._infer_emotion_from_features(features)
        result['emotion'] = emotion
        result['confidence'] = confidence
        
        return result
    
    def _predict_with_simple_features(self, image: np.ndarray, result: Dict) -> Dict:
        """
        基于简单特征的表情识别
        """
        # 提取简单特征
        features = {}
        
        # 计算边缘
        edges = cv2.Canny(image, 100, 200)
        features['edge_density'] = np.mean(edges)
        
        # 计算对比度
        features['contrast'] = np.std(image)
        
        # 计算亮度
        features['brightness'] = np.mean(image)
        
        result['features'] = features
        
        # 简单启发式规则
        if features['contrast'] > 0.3:
            result['emotion'] = 'surprised'
            result['confidence'] = 0.6
        elif features['brightness'] > 0.6:
            result['emotion'] = 'happy'
            result['confidence'] = 0.5
        else:
            result['emotion'] = 'neutral'
            result['confidence'] = 0.5
        
        return result
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        计算眼睛张开程度 (Eye Aspect Ratio)
        
        使用6个眼睛关键点计算
        """
        # ��算眼睛的竖直距离
        vertical_dist = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5]) + \
                       np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # 计算眼睛的水平距离
        horizontal_dist = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # 计算比例
        ear = vertical_dist / (2.0 * horizontal_dist + 1e-6)
        
        return ear
    
    def _infer_emotion_from_features(self, features: Dict) -> Tuple[str, float]:
        """
        基于特征推断情绪
        """
        eye_height = (features.get('left_eye_height', 0) + 
                     features.get('right_eye_height', 0)) / 2
        mouth_height = features.get('mouth_height', 0)
        mouth_ratio = features.get('mouth_aspect_ratio', 0)
        eyebrow_height_diff = abs(features.get('left_eyebrow_height', 0) - 
                                 features.get('right_eyebrow_height', 0))
        
        # 启发式规则
        if mouth_ratio > 0.6 and mouth_height > 20:  # 张大嘴巴
            return 'happy', 0.7
        elif eye_height < 0.1:  # 眼睛闭合
            return 'sad', 0.6
        elif eyebrow_height_diff > 5:  # 眉毛不对称
            return 'angry', 0.5
        elif mouth_ratio < 0.3:  # 嘴巴紧闭
            return 'neutral', 0.5
        else:
            return 'neutral', 0.4
    
    def analyze_faces(self, image_path: str = None, image: np.ndarray = None,
                     faces: List[Dict] = None) -> List[Dict]:
        """
        分析图像中所有人脸的表情
        
        Args:
            image_path: 图像路径
            image: 图像数组
            faces: 预检测的人脸列表
        
        Returns:
            List[Dict]: 包含表情信息的人脸列表
        """
        # 读取图像
        if image is None:
            if image_path is None:
                raise ValueError("必须提供 image_path 或 image")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 分析每个人脸
        results = []
        if faces is None:
            faces = []
        
        for face in faces:
            bbox = face.get('bbox')
            landmarks = face.get('landmarks')
            
            # 提取人脸区域
            x, y, w, h = bbox
            face_region = image[y:y+h, x:x+w]
            
            # 分析表情
            expression_result = self.analyze_expression(face_region, landmarks)
            
            # 合并结果
            face['expression'] = expression_result
            results.append(face)
        
        return results
    
    def get_emotion_description(self, emotion: str, language: str = 'en') -> str:
        """
        获取情绪的描述
        
        Args:
            emotion: 情绪类别
            language: 语言 ('en' 或 'zh')
        
        Returns:
            str: 情绪描述
        """
        descriptions = {
            'en': {
                'happy': 'The person appears happy and content',
                'sad': 'The person appears sad or unhappy',
                'angry': 'The person appears angry or irritated',
                'surprised': 'The person appears surprised or shocked',
                'afraid': 'The person appears afraid or anxious',
                'disgusted': 'The person appears disgusted or repulsed',
                'neutral': 'The person appears neutral or expressionless'
            },
            'zh': {
                'happy': '这个人看起来很开心',
                'sad': '这个人看起来很伤心',
                'angry': '这个人看起来很生气',
                'surprised': '这个人看起来很惊讶',
                'afraid': '这个人看起来很害怕',
                'disgusted': '这个人看起来很厌恶',
                'neutral': '这个人看起来很平静'
            }
        }
        
        if language in descriptions:
            return descriptions[language].get(emotion, 'Unknown emotion')
        else:
            return descriptions['en'].get(emotion, 'Unknown emotion')
"""情绪分类器 - 标准化的情绪分类"""

from typing import Dict, List
import numpy as np

class EmotionClassifier:
    """标准情绪分类器"""
    
    # 情绪分类
    EMOTIONS = {
        0: 'happy',      # 开心
        1: 'sad',        # 悲伤
        2: 'angry',      # 愤怒
        3: 'surprised',  # 惊讶
        4: 'disgusted',  # 厌恶
        5: 'afraid',     # 恐惧
        6: 'neutral'     # 中性
    }
    
    # 情绪分类的中文名称
    EMOTIONS_CN = {
        'happy': '开心',
        'sad': '悲伤',
        'angry': '愤怒',
        'surprised': '惊讶',
        'disgusted': '厌恶',
        'afraid': '恐惧',
        'neutral': '中性'
    }
    
    # 情绪的积极度（-1 消极，0 中性，1 积极）
    EMOTION_VALENCE = {
        'happy': 1,
        'sad': -1,
        'angry': -1,
        'surprised': 0,
        'disgusted': -1,
        'afraid': -1,
        'neutral': 0
    }
    
    # 情绪的能量度（0 低能量，1 高能量）
    EMOTION_AROUSAL = {
        'happy': 1,
        'sad': -1,
        'angry': 1,
        'surprised': 1,
        'disgusted': 0,
        'afraid': 1,
        'neutral': 0
    }
    
    def __init__(self):
        pass
    
    def get_emotion_name(self, emotion_id: int, language: str = 'en') -> str:
        """
        根据情绪ID获取情绪名称
        
        Args:
            emotion_id: 情绪ID (0-6)
            language: 语言 ('en' 或 'zh')
        
        Returns:
            str: 情绪名称
        """
        emotion = self.EMOTIONS.get(emotion_id, 'neutral')
        
        if language == 'zh':
            return self.EMOTIONS_CN.get(emotion, emotion)
        else:
            return emotion
    
    def get_emotion_properties(self, emotion: str) -> Dict:
        """
        获取情绪的属性（积极度和能量度）
        
        Args:
            emotion: 情绪名称
        
        Returns:
            Dict: 包含积极度和能量度
        """
        return {
            'emotion': emotion,
            'valence': self.EMOTION_VALENCE.get(emotion, 0),  # 积极度
            'arousal': self.EMOTION_AROUSAL.get(emotion, 0)   # 能量度
        }
    
    def classify_emotions(self, scores: Dict[str, float]) -> Dict:
        """
        基于得分分类情绪
        
        Args:
            scores: 各情绪的得分字典
        
        Returns:
            Dict: 分类结果
        """
        if not scores:
            return {
                'primary': 'neutral',
                'secondary': None,
                'confidence': 0.0
            }
        
        # 按得分排序
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        primary = sorted_emotions[0][0]
        primary_score = sorted_emotions[0][1]
        
        secondary = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None
        secondary_score = sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0.0
        
        return {
            'primary': primary,
            'primary_score': primary_score,
            'secondary': secondary,
            'secondary_score': secondary_score,
            'all_scores': scores
        }
    
    def is_positive_emotion(self, emotion: str) -> bool:
        """
        判断情绪是否为积极的
        
        Args:
            emotion: 情绪名称
        
        Returns:
            bool: 是否为积极情绪
        """
        return self.EMOTION_VALENCE.get(emotion, 0) > 0
    
    def is_negative_emotion(self, emotion: str) -> bool:
        """
        判断情绪是否为消极的
        
        Args:
            emotion: 情绪名称
        
        Returns:
            bool: 是否为消极情绪
        """
        return self.EMOTION_VALENCE.get(emotion, 0) < 0
    
    def is_high_energy_emotion(self, emotion: str) -> bool:
        """
        判断情绪是否为高能量
        
        Args:
            emotion: 情绪名称
        
        Returns:
            bool: 是否为高能量情绪
        """
        return self.EMOTION_AROUSAL.get(emotion, 0) > 0
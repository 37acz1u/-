"""多模态情绪融合模块 - 综合视觉、听觉和文本信息"""

from typing import Dict, List, Tuple
import numpy as np

class EmotionFusion:
    """多模态情绪融合器"""
    
    def __init__(self, weights: Dict[str, float] = None, method: str = 'weighted_average'):
        """
        初始化情绪融合器
        
        Args:
            weights: 各模态的权重 {'facial': 0.4, 'audio': 0.3, 'text': 0.3}
            method: 融合方法 ('weighted_average' 或 'voting')
        """
        if weights is None:
            self.weights = {
                'facial': 0.4,
                'audio': 0.3,
                'text': 0.3
            }
        else:
            self.weights = weights
            # 归一化权重
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        self.method = method
        self.emotion_classes = [
            'happy', 'sad', 'angry', 'surprised', 'disgusted', 'afraid', 'neutral'
        ]
    
    def fuse_emotions(self, facial_emotion: str = None, facial_scores: Dict = None,
                     audio_emotion: str = None, audio_scores: Dict = None,
                     text_emotion: str = None, text_scores: Dict = None) -> Dict:
        """
        融合多个模态的情绪识别结果
        
        Args:
            facial_emotion: 面部表情识别结果
            facial_scores: 面部表情的情绪得分
            audio_emotion: 语音情感识别结果
            audio_scores: 语音情感的得分
            text_emotion: 文本情感识别结果
            text_scores: 文本情感的得分
        
        Returns:
            Dict: 融合后的情绪识别结果
        """
        
        # 将所有得分转换为统一格式
        scores_dict = {}
        
        if facial_scores is not None:
            self._add_scores(scores_dict, facial_scores, self.weights['facial'])
        elif facial_emotion is not None:
            self._add_emotion_as_scores(scores_dict, facial_emotion, self.weights['facial'])
        
        if audio_scores is not None:
            self._add_scores(scores_dict, audio_scores, self.weights['audio'])
        elif audio_emotion is not None:
            self._add_emotion_as_scores(scores_dict, audio_emotion, self.weights['audio'])
        
        if text_scores is not None:
            self._add_scores(scores_dict, text_scores, self.weights['text'])
        elif text_emotion is not None:
            self._add_emotion_as_scores(scores_dict, text_emotion, self.weights['text'])
        
        # 选择最终情绪
        if scores_dict:
            final_emotion = max(scores_dict, key=scores_dict.get)
            confidence = scores_dict[final_emotion]
        else:
            final_emotion = 'neutral'
            confidence = 0.0
        
        return {
            'emotion': final_emotion,
            'confidence': confidence,
            'scores': scores_dict,
            'components': {
                'facial': {
                    'emotion': facial_emotion,
                    'scores': facial_scores,
                    'weight': self.weights['facial']
                },
                'audio': {
                    'emotion': audio_emotion,
                    'scores': audio_scores,
                    'weight': self.weights['audio']
                },
                'text': {
                    'emotion': text_emotion,
                    'scores': text_scores,
                    'weight': self.weights['text']
                }
            }
        }
    
    def _add_scores(self, scores_dict: Dict, emotion_scores: Dict, weight: float):
        """
        将带权重的情绪得分添加到总得分中
        """
        for emotion, score in emotion_scores.items():
            if emotion not in scores_dict:
                scores_dict[emotion] = 0.0
            scores_dict[emotion] += score * weight
    
    def _add_emotion_as_scores(self, scores_dict: Dict, emotion: str, weight: float):
        """
        将单个情绪转换为得分并添加
        """
        for emo in self.emotion_classes:
            if emo not in scores_dict:
                scores_dict[emo] = 0.0
            if emo == emotion:
                scores_dict[emo] += weight  # 该情绪得分为权重值
    
    def analyze_emotion_trend(self, emotion_history: List[str]) -> Dict:
        """
        分析情绪变化趋势
        
        Args:
            emotion_history: 历史情绪列表
        
        Returns:
            Dict: 情绪趋势分析结果
        """
        if not emotion_history:
            return {'trend': 'unknown', 'stability': 0.0}
        
        emotion_counts = {}
        for emotion in emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 计算主导情绪
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominant_count = emotion_counts[dominant_emotion]
        stability = dominant_count / len(emotion_history)
        
        # 检测趋势
        if len(emotion_history) > 1:
            first_half = emotion_history[:len(emotion_history)//2]
            second_half = emotion_history[len(emotion_history)//2:]
            
            first_emotion = max(set(first_half), key=first_half.count)
            second_emotion = max(set(second_half), key=second_half.count)
            
            if first_emotion == second_emotion:
                trend = 'stable'
            else:
                trend = 'changing'
        else:
            trend = 'unknown'
        
        return {
            'trend': trend,
            'stability': stability,
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_counts
        }
    
    def set_weights(self, weights: Dict[str, float]):
        """
        设置各模态的权重
        
        Args:
            weights: 权重字典
        """
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
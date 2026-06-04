"""文本情感分析模块"""

import numpy as np
from typing import Dict, List, Tuple
import os

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    print("警告: transformers 或 torch 未安装")

class SentimentAnalyzer:
    """文本情感分析器 - 基于 BERT 和自定义规则"""
    
    def __init__(self, model_name: str = 'bert-base-multilingual-uncased', 
                 use_transformer: bool = True, language: str = 'zh'):
        """
        初始化文本情感分析器
        
        Args:
            model_name: 使用的预训练模型名称
            use_transformer: 是否使用 Transformer 模型
            language: 语言 ('zh' 中文, 'en' 英文)
        """
        self.language = language
        self.use_transformer = use_transformer and 'pipeline' in dir()
        self.model = None
        self.tokenizer = None
        
        # 情感词典 (中文)
        self.positive_words_cn = {
            '开心', '高兴', '快乐', '喜欢', '爱', '棒', '好', '完美', '优秀', '太好了',
            '很开心', '很高兴', '非常好', '赞', '点赞', '厉害', '给力', '牛', '祝贺',
            '恭喜', '太棒了', '最好', '成功', '胜利', '赢', '幸福', '满足', '满意'
        }
        
        self.negative_words_cn = {
            '难过', '伤心', '悲伤', '生气', '愤怒', '讨厌', '厌烦', '烦', '糟糕', '差',
            '不好', '失望', '沮丧', '郁闷', '委屈', '绝望', '痛苦', '害怕', '恐惧',
            '担心', '焦虑', '紧张', '不安', '烦躁', '讨厌', '反感', '厌恶', '憎恨',
            '失败', '挫折', '不行', '太差', '糟糕透了', '崩溃', '抓狂', '要疯了'
        }
        
        self.neutral_words_cn = {
            '可能', '也许', '似乎', '感觉', '想起', '想到', '觉得', '认为', '看起来',
            '听起来', '其实', '大概', '差不多', '一般', '还好', '勉强', '算是'
        }
        
        # 情感词典 (英文)
        self.positive_words_en = {
            'love', 'like', 'good', 'great', 'awesome', 'excellent', 'amazing', 'wonderful',
            'fantastic', 'perfect', 'happy', 'joyful', 'delighted', 'pleased', 'satisfied',
            'grateful', 'proud', 'excited', 'thrilled', 'beautiful', 'nice', 'fantastic'
        }
        
        self.negative_words_en = {
            'hate', 'dislike', 'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'sad',
            'unhappy', 'disappointed', 'angry', 'furious', 'frustrated', 'worried', 'anxious',
            'scared', 'afraid', 'regret', 'sorry', 'annoyed', 'irritated', 'upset', 'failed'
        }
        
        self.neutral_words_en = {
            'maybe', 'perhaps', 'seem', 'feel', 'think', 'believe', 'consider', 'appear',
            'sound', 'look', 'could', 'might', 'probably', 'possibly'
        }
        
        if self.use_transformer:
            try:
                if language == 'zh':
                    # 中文情感分析模型
                    self.model = pipeline('text-classification', 
                                        model='uer/chinese-electra-small-sentiment',
                                        truncation=True)
                else:
                    # 英文情感分析模型
                    self.model = pipeline('sentiment-analysis',
                                        model='distilbert-base-uncased-finetuned-sst-2-english',
                                        truncation=True)
                print(f"✓ 已加载 {language} 情感分析模型")
            except Exception as e:
                print(f"⚠ 加载 Transformer 模型失败: {e}")
                self.use_transformer = False
    
    def analyze_sentiment(self, text: str, use_rules: bool = False) -> Dict:
        """
        分析文本情感
        
        Args:
            text: 输入文本
            use_rules: 是否同时使用规则匹配
        
        Returns:
            Dict: 情感分析结果
        """
        if not text or len(text.strip()) == 0:
            return {
                'text': text,
                'emotion': 'neutral',
                'confidence': 0.0,
                'scores': {},
                'method': 'empty'
            }
        
        # 方法1: 使用 Transformer 模型
        if self.use_transformer:
            try:
                result = self._analyze_with_transformer(text)
                
                # 方法2: 结合规则匹配
                if use_rules:
                    rule_result = self._analyze_with_rules(text)
                    # 融合两个结果 (60% 模型 + 40% 规则)
                    result['scores']['transformer'] = result['confidence']
                    result['scores']['rules'] = rule_result['confidence']
                    result['confidence'] = result['confidence'] * 0.6 + rule_result['confidence'] * 0.4
                    result['method'] = 'hybrid'
                else:
                    result['method'] = 'transformer'
                
                return result
            except Exception as e:
                print(f"Transformer 分析失败: {e}，使用规则匹配")
        
        # 方法2: 使用规则匹配
        return self._analyze_with_rules(text)
    
    def _analyze_with_transformer(self, text: str) -> Dict:
        """
        使用 Transformer 模型分析情感
        """
        try:
            prediction = self.model(text[0:512])[0]  # 限制文本长度
            
            label = prediction['label']
            score = prediction['score']
            
            # 转换标签到标准情感类别
            if self.language == 'zh':
                # 中文模型通常返回 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
                emotion_map = {
                    'POSITIVE': 'happy',
                    'NEGATIVE': 'sad',
                    'NEUTRAL': 'neutral'
                }
            else:
                # 英文模型返回 'POSITIVE', 'NEGATIVE'
                emotion_map = {
                    'POSITIVE': 'happy',
                    'NEGATIVE': 'sad'
                }
            
            emotion = emotion_map.get(label, 'neutral')
            
            return {
                'text': text,
                'emotion': emotion,
                'confidence': float(score),
                'scores': {
                    emotion: float(score),
                    'neutral' if emotion != 'neutral' else 'other': 1.0 - float(score)
                }
            }
        
        except Exception as e:
            raise e
    
    def _analyze_with_rules(self, text: str) -> Dict:
        """
        使用规则和情感词典分析情感
        """
        text_lower = text.lower()
        
        # 选择语言对应的词典
        if self.language == 'zh':
            positive_words = self.positive_words_cn
            negative_words = self.negative_words_cn
            neutral_words = self.neutral_words_cn
        else:
            positive_words = self.positive_words_en
            negative_words = self.negative_words_en
            neutral_words = self.neutral_words_en
        
        # 计算正负中词汇的出现次数
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        neutral_count = sum(1 for word in neutral_words if word in text_lower)
        
        total_count = positive_count + negative_count + neutral_count
        
        if total_count == 0:
            # 没有找到情感词，返回中性
            return {
                'text': text,
                'emotion': 'neutral',
                'confidence': 0.5,
                'scores': {
                    'positive': 0.33,
                    'negative': 0.33,
                    'neutral': 0.34
                }
            }
        
        # 计算比例
        positive_ratio = positive_count / total_count
        negative_ratio = negative_count / total_count
        neutral_ratio = neutral_count / total_count
        
        # 判断主要情感
        if positive_ratio > negative_ratio and positive_ratio > neutral_ratio:
            emotion = 'happy'
            confidence = positive_ratio
        elif negative_ratio > positive_ratio and negative_ratio > neutral_ratio:
            emotion = 'sad'
            confidence = negative_ratio
        else:
            emotion = 'neutral'
            confidence = neutral_ratio
        
        # 添加强度判断 (感叹号、重复词等)
        if '!!' in text or '!!!' in text:
            confidence = min(1.0, confidence + 0.2)
        if '...' in text or '。。。' in text:
            confidence = min(1.0, confidence + 0.1)
        
        return {
            'text': text,
            'emotion': emotion,
            'confidence': min(1.0, confidence),
            'scores': {
                'positive': positive_ratio,
                'negative': negative_ratio,
                'neutral': neutral_ratio
            }
        }
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        提取文本中的关键词
        
        Args:
            text: 输入文本
            top_k: 返回的关键词个数
        
        Returns:
            List[str]: 关键词列表
        """
        try:
            if self.language == 'zh':
                import jieba
                words = jieba.cut(text)
                # 过滤停用词
                stop_words = {'的', '是', '在', '和', '了', '不', '很', '有', '个', '从'}
                keywords = [w for w in words if w not in stop_words and len(w) > 1]
            else:
                import nltk
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt')
                    nltk.download('stopwords')
                
                tokens = word_tokenize(text.lower())
                stop_words = set(stopwords.words('english'))
                keywords = [w for w in tokens if w.isalpha() and w not in stop_words]
            
            # 返回前 k 个
            return list(set(keywords))[:top_k]
        
        except Exception as e:
            print(f"关键词提取失败: {e}")
            return []
    
    def analyze_emotion_intensity(self, text: str) -> Tuple[str, float]:
        """
        分析文本的情感强度
        
        Args:
            text: 输入文本
        
        Returns:
            Tuple: (情感类型, 强度)
        """
        # 基于特殊字符和重复词的强度判断
        intensity = 0.5
        
        # 感叹号 (+0.2 per !)
        intensity += text.count('!') * 0.2
        intensity += text.count('！') * 0.2
        
        # 问号 (+0.1 per ?)
        intensity += text.count('?') * 0.1
        intensity += text.count('？') * 0.1
        
        # 省略号 (+0.1)
        if '...' in text or '。。。' in text:
            intensity += 0.1
        
        # 重复字符
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2]:
                intensity += 0.1
        
        # 分析基础情感
        result = self.analyze_sentiment(text)
        emotion = result['emotion']
        
        # 归一化强度 (0-1)
        intensity = min(1.0, max(0.0, intensity))
        
        return emotion, intensity
    
    def compare_sentiments(self, texts: List[str]) -> Dict:
        """
        比较多个文本的情感
        
        Args:
            texts: 文本列表
        
        Returns:
            Dict: 比较结果
        """
        results = []
        emotion_counts = {}
        
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
            
            emotion = result['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 找出主要情感
        if emotion_counts:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        else:
            dominant_emotion = 'neutral'
        
        return {
            'total_texts': len(texts),
            'results': results,
            'emotion_distribution': emotion_counts,
            'dominant_emotion': dominant_emotion
        }
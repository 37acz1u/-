"""机器人集成模块 - 实时情绪识别"""

import cv2
import threading
import time
from queue import Queue
from typing import Dict, Callable
import numpy as np

from vision.face_detector import FaceDetector
from vision.expression_analyzer import ExpressionAnalyzer
from audio.tone_analyzer import ToneAnalyzer
from text.sentiment_analyzer import SentimentAnalyzer
from emotion_engine.emotion_fusion import EmotionFusion

class RobotEmotionEngine:
    """机器人情绪识别引擎 - 支持实时多模态分析"""
    
    def __init__(self, enable_facial=True, enable_audio=True, enable_text=True):
        """
        初始化机器人情绪引擎
        
        Args:
            enable_facial: 启用面部表情识别
            enable_audio: 启用语音情感分析
            enable_text: 启用文本情感分析
        """
        print("[RobotEmotionEngine] 初始化中...")
        
        self.enable_facial = enable_facial
        self.enable_audio = enable_audio
        self.enable_text = enable_text
        
        # 初始化分析模块
        if self.enable_facial:
            print("  - 初始化面部识别模块...")
            self.face_detector = FaceDetector(model_type='opencv')
            self.expression_analyzer = ExpressionAnalyzer(use_pretrained=True)
        
        if self.enable_audio:
            print("  - 初始化语音分析模块...")
            self.tone_analyzer = ToneAnalyzer()
        
        if self.enable_text:
            print("  - 初始化文本分析模块...")
            self.sentiment_analyzer = SentimentAnalyzer(language='zh', use_transformer=False)
        
        # 初始化融合引擎
        self.fusion_engine = EmotionFusion()
        
        # 实时数据缓存
        self.latest_results = {
            'facial': None,
            'audio': None,
            'text': None,
            'fused': None,
            'timestamp': None
        }
        
        # 回调函数
        self.callbacks = []
        
        # 线程控制
        self.running = False
        self.threads = []
        
        print("✓ RobotEmotionEngine 初始化完成！\n")
    
    def register_callback(self, callback: Callable):
        """
        注册回调函数，当有新的情绪识别结果时调用
        
        Args:
            callback: 回调函数，接收 emotion_result 字典
        """
        self.callbacks.append(callback)
    
    def process_camera_frame(self, frame: np.ndarray, face_ids: Dict = None) -> Dict:
        """
        处理摄像头帧 - 面部表情识别
        
        Args:
            frame: 摄像头帧 (BGR 格式)
            face_ids: 已知的人脸ID列表 (可选)
        
        Returns:
            Dict: 识别结果
        """
        try:
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            faces = self.face_detector.detect_faces(image=frame_rgb)
            
            results = []
            for i, face in enumerate(faces):
                bbox = face['bbox']
                x, y, w, h = bbox
                face_region = frame_rgb[y:y+h, x:x+w]
                
                # 分析表情
                expression = self.expression_analyzer.analyze_expression(
                    face_region, 
                    face.get('landmarks')
                )
                
                result = {
                    'face_id': i + 1,
                    'bbox': bbox,
                    'emotion': expression['emotion'],
                    'confidence': float(expression['confidence']),
                    'scores': expression['scores']
                }
                
                results.append(result)
            
            # 更新缓存
            self.latest_results['facial'] = {
                'faces': results,
                'count': len(results),
                'timestamp': time.time()
            }
            
            return self.latest_results['facial']
        
        except Exception as e:
            print(f"[错误] 处理摄像头帧失败: {e}")
            return None
    
    def process_audio_stream(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        处理音频流 - 实时语音情感分析
        
        Args:
            audio_data: 音频数据 (numpy 数组)
            sample_rate: 采样率
        
        Returns:
            Dict: 分析结果
        """
        try:
            # 保存为临时文件
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio_data, sample_rate)
                tmp_path = tmp.name
            
            # 分析语调
            result = self.tone_analyzer.analyze_tone(tmp_path)
            
            if result.get('success'):
                self.latest_results['audio'] = {
                    'emotion': result['emotion'],
                    'confidence': float(result['confidence']),
                    'scores': result['scores'],
                    'features': result.get('features'),
                    'timestamp': time.time()
                }
            
            # 清理临时文件
            import os
            os.remove(tmp_path)
            
            return self.latest_results['audio']
        
        except Exception as e:
            print(f"[错误] 处理音频失败: {e}")
            return None
    
    def process_text(self, text: str) -> Dict:
        """
        处理文本 - 文本情感分析
        
        Args:
            text: 输入文本
        
        Returns:
            Dict: 分析结果
        """
        try:
            result = self.sentiment_analyzer.analyze_sentiment(text, use_rules=True)
            emotion, intensity = self.sentiment_analyzer.analyze_emotion_intensity(text)
            
            self.latest_results['text'] = {
                'text': text,
                'emotion': result['emotion'],
                'confidence': float(result['confidence']),
                'intensity': float(intensity),
                'scores': result['scores'],
                'timestamp': time.time()
            }
            
            return self.latest_results['text']
        
        except Exception as e:
            print(f"[错误] 处理文本失败: {e}")
            return None
    
    def fuse_emotions(self) -> Dict:
        """
        融合多模态结果
        
        Returns:
            Dict: 融合后的情感结果
        """
        try:
            fused = self.fusion_engine.fuse_emotions(
                facial_emotion=self.latest_results['facial']['faces'][0]['emotion'] 
                    if self.latest_results['facial'] and self.latest_results['facial']['faces'] else None,
                facial_scores=self.latest_results['facial']['faces'][0]['scores']
                    if self.latest_results['facial'] and self.latest_results['facial']['faces'] else None,
                audio_emotion=self.latest_results['audio']['emotion'] 
                    if self.latest_results['audio'] else None,
                audio_scores=self.latest_results['audio']['scores']
                    if self.latest_results['audio'] else None,
                text_emotion=self.latest_results['text']['emotion'] 
                    if self.latest_results['text'] else None,
                text_scores=self.latest_results['text']['scores']
                    if self.latest_results['text'] else None
            )
            
            self.latest_results['fused'] = {
                'emotion': fused['emotion'],
                'confidence': float(fused['confidence']),
                'scores': fused['scores'],
                'timestamp': time.time()
            }
            
            # 触发回调
            self._trigger_callbacks(self.latest_results['fused'])
            
            return self.latest_results['fused']
        
        except Exception as e:
            print(f"[错误] 融合情感失败: {e}")
            return None
    
    def _trigger_callbacks(self, result: Dict):
        """触发所有注册的回调函数"""
        for callback in self.callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"[警告] 回调函数执行失败: {e}")
    
    def get_latest_result(self) -> Dict:
        """获取最新的融合结果"""
        return self.latest_results['fused']
    
    def get_all_results(self) -> Dict:
        """获取所有分析结果"""
        return self.latest_results


class RealtimeCameraProcessor:
    """实时摄像头处理器"""
    
    def __init__(self, camera_id: int = 0, fps: int = 30):
        """
        初始化摄像头处理器
        
        Args:
            camera_id: 摄像头 ID (0 = 默认摄像头)
            fps: 处理帧率
        """
        self.camera_id = camera_id
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        self.engine = RobotEmotionEngine(
            enable_facial=True,
            enable_audio=False,  # 实时处理不包含音频
            enable_text=False
        )
        
        self.cap = None
        self.running = False
        self.latest_frame = None
        self.latest_result = None
    
    def start(self):
        """启动实时处理"""
        print("[RealtimeCameraProcessor] 正在启动摄像头...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("✗ 错误：无法打开摄像头")
            return False
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.running = True
        
        # 启动处理线程
        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()
        
        print("✓ 摄像头已启动，按 'q' 退出")
        return True
    
    def _process_loop(self):
        """处理循环"""
        last_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("✗ 读取摄像头帧失败")
                break
            
            self.latest_frame = frame
            
            # 控制帧率
            current_time = time.time()
            if current_time - last_time >= self.frame_interval:
                # 处理帧
                result = self.engine.process_camera_frame(frame)
                self.latest_result = result
                last_time = current_time
    
    def display(self):
        """显示实时处理结果"""
        if not self.running or self.cap is None:
            return False
        
        while self.running:
            if self.latest_frame is None:
                continue
            
            frame = self.latest_frame.copy()
            
            # 绘制结果
            if self.latest_result and self.latest_result.get('faces'):
                for face in self.latest_result['faces']:
                    bbox = face['bbox']
                    x, y, w, h = bbox
                    
                    # 绘制边框
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # 绘制情绪标签
                    emotion = face['emotion']
                    confidence = face['confidence']
                    text = f"{emotion} ({confidence:.1%})"
                    cv2.putText(frame, text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 显示帧
            cv2.imshow('Real-time Emotion Recognition', frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break
        
        cv2.destroyAllWindows()
        return True
    
    def stop(self):
        """停止处理"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("✓ 摄像头已关闭")
    
    def register_callback(self, callback: Callable):
        """注册回调函数"""
        self.engine.register_callback(callback)


class RobotIntegration:
    """机器人集成接口 - 与机器人平台集成"""
    
    def __init__(self):
        """初始化机器人集成"""
        self.emotion_engine = RobotEmotionEngine()
        self.camera_processor = None
        self.emotion_history = []
    
    def setup_camera(self, camera_id: int = 0, fps: int = 30):
        """设置摄像头"""
        self.camera_processor = RealtimeCameraProcessor(camera_id, fps)
        return self.camera_processor
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        分析单张图片
        
        Args:
            image_path: 图片路径
        
        Returns:
            Dict: 分析结果
        """
        import cv2
        frame = cv2.imread(image_path)
        return self.emotion_engine.process_camera_frame(frame)
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        分析音频文件
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            Dict: 分析结果
        """
        import librosa
        audio_data, sr = librosa.load(audio_path, sr=16000)
        return self.emotion_engine.process_audio_stream(audio_data, sr)
    
    def analyze_text(self, text: str) -> Dict:
        """
        分析文本
        
        Args:
            text: 输入文本
        
        Returns:
            Dict: 分析结果
        """
        return self.emotion_engine.process_text(text)
    
    def get_emotion_command(self, emotion: str) -> str:
        """
        根据识别的情绪返回机器人应该执行的命令
        
        Args:
            emotion: 识别的情绪
        
        Returns:
            str: 机器人命令
        """
        emotion_commands = {
            'happy': 'smile',      # 微笑
            'sad': 'comfort',      # 安慰
            'angry': 'retreat',    # 后退/冷静
            'surprised': 'attention',  # 注意/警惕
            'afraid': 'protect',   # 保护/靠近
            'disgusted': 'avoid',  # 避开
            'neutral': 'idle'      # 待机
        }
        return emotion_commands.get(emotion, 'idle')
    
    def execute_emotion_response(self, emotion: str, robot_interface):
        """
        执行情感反应
        
        Args:
            emotion: 识别的情绪
            robot_interface: 机器人接口对象
        """
        command = self.get_emotion_command(emotion)
        
        # 这里调用机器人特定的 API
        # 例如：robot_interface.execute_command(command)
        print(f"[机器人响应] 情绪: {emotion} -> 命令: {command}")
        
        # 记录情绪历史
        self.emotion_history.append({
            'emotion': emotion,
            'command': command,
            'timestamp': time.time()
        })
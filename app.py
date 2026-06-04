"""多模态情绪识别机器人 - Flask Web 应用"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# 导入模块
from vision.face_detector import FaceDetector
from vision.expression_analyzer import ExpressionAnalyzer
from audio.speech_recognizer import SpeechRecognizer
from audio.tone_analyzer import ToneAnalyzer
from text.sentiment_analyzer import SentimentAnalyzer
from emotion_engine.emotion_fusion import EmotionFusion
from emotion_engine.emotion_classifier import EmotionClassifier

# 初始化 Flask 应用
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = 'uploads'

# 创建文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# 初始化模型
print("[系统初始化中...]")
print("- 初始化人脸检测...")
face_detector = FaceDetector(model_type='opencv')
expression_analyzer = ExpressionAnalyzer(use_pretrained=True)

print("- 初始化语音分析...")
speech_recognizer = SpeechRecognizer(language='zh-CN')
tone_analyzer = ToneAnalyzer()

print("- 初始化文本分析...")
sentiment_analyzer = SentimentAnalyzer(language='zh', use_transformer=False)

print("- 初始化情绪融合...")
fusion_engine = EmotionFusion()
emotion_classifier = EmotionClassifier()

print("✓ 系统初始化完成！\n")

# 允许的文件类型
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """分析图片中的面部表情"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未提供图片文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': '不支持的图片格式'}), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 读取和分析图片
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': '无法读取图片'}), 400
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测人脸
        faces = face_detector.detect_faces(image=image_rgb)
        
        if len(faces) == 0:
            return jsonify({
                'success': True,
                'faces_count': 0,
                'message': '未检测到人脸'
            })
        
        # 分析每个人脸
        results = []
        for i, face in enumerate(faces):
            bbox = face['bbox']
            x, y, w, h = bbox
            face_region = image_rgb[y:y+h, x:x+w]
            
            # 分析表情
            expression = expression_analyzer.analyze_expression(face_region, face.get('landmarks'))
            
            results.append({
                'face_id': i + 1,
                'bbox': list(bbox),
                'emotion': expression['emotion'],
                'confidence': float(expression['confidence']),
                'scores': {k: float(v) for k, v in expression['scores'].items()}
            })
        
        # 生成标注图片
        annotated_image = face_detector.draw_faces_on_image(image_rgb, faces)
        for i, result in enumerate(results):
            bbox = result['bbox']
            x, y, w, h = bbox
            emotion = result['emotion']
            confidence = result['confidence']
            text = f"{emotion} ({confidence:.1%})"
            cv2.putText(annotated_image, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 保存结果图片
        result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        result_filepath = os.path.join('static/results', result_filename)
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_filepath, annotated_image_bgr)
        
        return jsonify({
            'success': True,
            'faces_count': len(results),
            'results': results,
            'result_image': f'/static/results/{result_filename}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/audio', methods=['POST'])
def analyze_audio():
    """分析音频的语音情感"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未提供音频文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({'error': '不支持的音频格式'}), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 分析语调
        tone_result = tone_analyzer.analyze_tone(filepath)
        
        if not tone_result.get('success'):
            return jsonify({'error': tone_result.get('error', '分析失败')}), 400
        
        return jsonify({
            'success': True,
            'emotion': tone_result['emotion'],
            'confidence': float(tone_result['confidence']),
            'scores': {k: float(v) for k, v in tone_result['scores'].items()}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """分析文本情感"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '文本不能为空'}), 400
        
        # 分析情感
        result = sentiment_analyzer.analyze_sentiment(text, use_rules=True)
        
        # 分析强度
        emotion, intensity = sentiment_analyzer.analyze_emotion_intensity(text)
        
        return jsonify({
            'success': True,
            'text': text,
            'emotion': result['emotion'],
            'confidence': float(result['confidence']),
            'intensity': float(intensity),
            'scores': {k: float(v) for k, v in result['scores'].items()}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/multimodal', methods=['POST'])
def analyze_multimodal():
    """多模态情感分析 - 融合三个系统"""
    try:
        data = request.get_json()
        
        facial_emotion = data.get('facial_emotion')
        facial_scores = data.get('facial_scores')
        audio_emotion = data.get('audio_emotion')
        audio_scores = data.get('audio_scores')
        text_emotion = data.get('text_emotion')
        text_scores = data.get('text_scores')
        
        # 融合情感
        fused_result = fusion_engine.fuse_emotions(
            facial_emotion=facial_emotion,
            facial_scores=facial_scores,
            audio_emotion=audio_emotion,
            audio_scores=audio_scores,
            text_emotion=text_emotion,
            text_scores=text_scores
        )
        
        # 获取情感属性
        emotion_props = emotion_classifier.get_emotion_properties(fused_result['emotion'])
        
        return jsonify({
            'success': True,
            'emotion': fused_result['emotion'],
            'confidence': float(fused_result['confidence']),
            'scores': {k: float(v) for k, v in fused_result['scores'].items()},
            'properties': emotion_props
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotions')
def get_emotions():
    """获取支持的情感列表"""
    emotions = {
        'happy': {'cn': '开心', 'description': '积极、快乐'},
        'sad': {'cn': '悲伤', 'description': '消极、低落'},
        'angry': {'cn': '愤怒', 'description': '激怒、生气'},
        'surprised': {'cn': '惊讶', 'description': '意外、惊讶'},
        'afraid': {'cn': '害怕', 'description': '恐惧、害怕'},
        'disgusted': {'cn': '厌恶', 'description': '厌恶、反感'},
        'neutral': {'cn': '中性', 'description': '平静、无表情'}
    }
    return jsonify(emotions)

@app.route('/api/info')
def get_info():
    """获取系统信息"""
    return jsonify({
        'name': '多模态情绪识别机器人',
        'version': '1.0.0',
        'features': [
            '面部表情识别 (视觉系统)',
            '语音情感分析 (听觉系统)',
            '文本情感分析 (文本系统)',
            '多模态情感融合'
        ],
        'supported_formats': {
            'image': ['jpg', 'jpeg', 'png', 'gif', 'bmp'],
            'audio': ['wav', 'mp3', 'ogg', 'flac']
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
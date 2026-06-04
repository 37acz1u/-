# 多模态情绪识别机器人

这是��个能够通过视觉系统、听觉系统和文本系统识别用户情绪的智能机器人。

## 项目结构

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── vision/                    # 视觉系统 - 面部表情识别
│   ├── __init__.py
│   ├── face_detector.py      # 人脸检测
│   ├── expression_analyzer.py # 表情分析
│   └── utils.py              # 工具函数
├── audio/                     # 听觉系统 - 语音情感分析
│   ├── __init__.py
│   ├── speech_recognizer.py
│   └── tone_analyzer.py
├── text/                      # 文本系统 - 文本情感分析
│   ├── __init__.py
│   └── sentiment_analyzer.py
├── emotion_engine/            # 情绪融合引擎
│   ├── __init__.py
│   ├── emotion_fusion.py     # 多模态融合
│   └── emotion_classifier.py
└── examples/                  # 示例脚本
    ├── facial_recognition_demo.py
    └── multi_modal_demo.py
```

## 功能特性

- ✅ **面部表情识别** - 识别开心、悲伤、惊讶、愤怒、厌恶、恐惧、中性等情绪
- 🎤 **语音情感分析** - 分析语调和语速中的情绪信息
- 💬 **文本情感分析** - 理解文本内容中的情感
- 🧠 **多模态融合** - 综合三个系统的信息进行最终情绪判断

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 面部表情识别示例

```python
from vision.face_detector import FaceDetector
from vision.expression_analyzer import ExpressionAnalyzer

# 初始化
detector = FaceDetector()
analyzer = ExpressionAnalyzer()

# 处理图像
faces = detector.detect_faces('image.jpg')
emotions = analyzer.analyze(faces)

print(emotions)
```

### 多模态情绪识别

```python
from emotion_engine.emotion_fusion import EmotionFusion

fusion = EmotionFusion()
result = fusion.fuse_emotions(
    facial_emotion='happy',
    voice_emotion='neutral',
    text_emotion='positive'
)
print(result)
```

## 依赖库

- OpenCV - 计算机视觉
- TensorFlow/Keras - 深度学习
- dlib - 人脸检测和关键点
- librosa - 音频处理
- transformers - NLP模型

## 许可证

MIT
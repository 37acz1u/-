#!/usr/bin/env python3
"""
多模态情绪识别机器人 - 快速启动指南

本脚本演示了系统的各种使用方式
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main_menu():
    """主菜单"""
    print("\n" + "="*60)
    print("多模态情绪识别机器人 - 主菜单")
    print("="*60)
    print("\n请选择运行模式:\n")
    print("1. 🌐 Web 界面 (推荐)")
    print("2. 📸 实时摄像头识别")
    print("3. 📁 分析本地文件")
    print("4. 🤖 机器人集成模式")
    print("5. ℹ️  系统信息")
    print("6. ❌ 退出\n")
    
    choice = input("请输入选择 (1-6): ").strip()
    return choice

def start_web_app():
    """启动 Web 应用"""
    print("\n[Web 应用] 正在启动...")
    print("="*60)
    try:
        from app import app
        print("✓ 应用已启动!")
        print("\n访问地址: http://localhost:5000")
        print("按 Ctrl+C 停止应用\n")
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"✗ 错误: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")

def start_realtime_camera():
    """启动实时摄像头识别"""
    print("\n[实时摄像头] 正在初始化...")
    print("="*60)
    try:
        from robot_integration import RealtimeCameraProcessor
        
        processor = RealtimeCameraProcessor(camera_id=0, fps=30)
        
        if not processor.start():
            print("✗ 无法启动摄像头")
            return
        
        # 注册回调函数
        def emotion_callback(result):
            if result:
                emotion = result.get('emotion', 'unknown')
                confidence = result.get('confidence', 0)
                print(f"[识别结果] 情绪: {emotion}, 置信度: {confidence:.1%}")
        
        processor.register_callback(emotion_callback)
        
        # 显示结果
        processor.display()
    
    except Exception as e:
        print(f"✗ 错误: {e}")

def analyze_local_files():
    """分析本地文件"""
    print("\n[本地文件分析]")
    print("="*60)
    
    while True:
        print("\n请选择文件类型:\n")
        print("1. 📸 分析图片")
        print("2. 🎤 分析音频")
        print("3. 📝 分析文本")
        print("4. 🔙 返回主菜单\n")
        
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == '1':
            analyze_image()
        elif choice == '2':
            analyze_audio()
        elif choice == '3':
            analyze_text()
        elif choice == '4':
            break
        else:
            print("✗ 无效的选择")

def analyze_image():
    """分析图片"""
    try:
        from robot_integration import RobotIntegration
        
        image_path = input("\n请输入图片路径: ").strip()
        
        if not Path(image_path).exists():
            print(f"✗ 文件不存在: {image_path}")
            return
        
        print("\n正在分析图片...")
        robot = RobotIntegration()
        result = robot.analyze_image(image_path)
        
        if result and result.get('faces'):
            print(f"\n检测到 {result['count']} 个人脸:\n")
            for face in result['faces']:
                print(f"  人脸 {face['face_id']}:")
                print(f"    - 情绪: {face['emotion']}")
                print(f"    - 置信度: {face['confidence']:.1%}")
                print(f"    - 位置: {face['bbox']}")
        else:
            print("\n✗ 未检测到人脸")
    
    except Exception as e:
        print(f"✗ 错误: {e}")

def analyze_audio():
    """分析音频"""
    try:
        from robot_integration import RobotIntegration
        
        audio_path = input("\n请输入音频文件路径: ").strip()
        
        if not Path(audio_path).exists():
            print(f"✗ 文件不存在: {audio_path}")
            return
        
        print("\n正在分析音频...")
        robot = RobotIntegration()
        result = robot.analyze_audio(audio_path)
        
        if result:
            print(f"\n分析结果:")
            print(f"  - 情绪: {result.get('emotion')}")
            print(f"  - 置信度: {result.get('confidence', 0):.1%}")
            print(f"  - 强度: {result.get('intensity', 0):.1%}")
        else:
            print("\n✗ 分析失败")
    
    except Exception as e:
        print(f"✗ 错误: {e}")

def analyze_text():
    """分析文本"""
    try:
        from robot_integration import RobotIntegration
        
        text = input("\n请输入文本 (或输入 'quit' 返回): ").strip()
        
        if text.lower() == 'quit':
            return
        
        if not text:
            print("✗ 文本不能为空")
            return
        
        print("\n正在分析文本...")
        robot = RobotIntegration()
        result = robot.analyze_text(text)
        
        if result:
            print(f"\n分析结果:")
            print(f"  - 情绪: {result.get('emotion')}")
            print(f"  - 置信度: {result.get('confidence', 0):.1%}")
            print(f"  - 强度: {result.get('intensity', 0):.1%}")
        else:
            print("\n✗ 分析失败")
    
    except Exception as e:
        print(f"✗ 错误: {e}")

def start_robot_mode():
    """启动机器人集成模式"""
    print("\n[机器人集成模式]")
    print("="*60)
    print("\n此模式允许你集成自己的机器人平台")
    print("\n示例代码:\n")
    
    example_code = '''
from robot_integration import RobotIntegration

# 初始化机器人集成
robot = RobotIntegration()

# 方法 1: 使用摄像头进行实时识别
camera = robot.setup_camera(camera_id=0, fps=30)
camera.register_callback(lambda result: 
    robot.execute_emotion_response(result['emotion'], your_robot_interface)
)
camera.start()
camera.display()

# 方法 2: 分析单张图片
result = robot.analyze_image('image.jpg')
emotion = result['faces'][0]['emotion']
robot.execute_emotion_response(emotion, your_robot_interface)

# 方法 3: 处理音频流
import librosa
audio, sr = librosa.load('audio.wav', sr=16000)
result = robot.emotion_engine.process_audio_stream(audio, sr)

# 方法 4: 处理文本
result = robot.emotion_engine.process_text("我很开心!")
    '''
    
    print(example_code)
    print("\n更详细的集成指南，请参考 DEPLOYMENT_GUIDE.md")
    print("\n按 Enter 返回主菜单")
    input()

def show_system_info():
    """显示系统信息"""
    print("\n[系统信息]")
    print("="*60)
    
    info = """
📊 项目统计
  - 视觉系统: 面部表情识别 (7种情绪)
  - 听觉系统: 语音情感分析 (基于音高、能量、语速)
  - 文本系统: 情感分析 (Transformer + 规则匹配)
  - 融合引擎: 加权平均融合

🔧 技术栈
  - 深度学习: TensorFlow, Keras, PyTorch
  - 计算机视觉: OpenCV, dlib
  - 音频处理: Librosa, SpeechRecognition
  - NLP: Transformers (BERT)
  - Web: Flask
  - 容器: Docker

📱 支持的情感类别
  1. 😊 Happy (开心)
  2. 😢 Sad (悲伤)
  3. 😠 Angry (愤怒)
  4. 😲 Surprised (惊讶)
  5. 🤢 Disgusted (厌恶)
  6. 😨 Afraid (恐惧)
  7. 😐 Neutral (中性)

🚀 部署方式
  - 本地运行 (Web + 命令行)
  - Docker 容器化
  - Heroku 云端部署
  - 机器人集成

💾 数据处理
  ✓ 无需预先提供数据 - 使用预训练模型
  ✓ 开箱即用 - 无需训练
  ✓ 支持实时处理
  ✓ 支持批量处理

🔌 机器人集成
  ✓ 支持 USB 摄像头
  ✓ 支持实时视频流
  ✓ 支持音频输入
  ✓ 支持文本输入
  ✓ 可自定义情绪反应
    """
    
    print(info)
    print("="*60)
    print("\n按 Enter 返回主菜单")
    input()

def main():
    """主程序"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  🤖 多模态情绪识别机器人系统".center(58) + "║")
    print("║" + "  Multimodal Emotion Recognition Robot".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    while True:
        try:
            choice = main_menu()
            
            if choice == '1':
                start_web_app()
            elif choice == '2':
                start_realtime_camera()
            elif choice == '3':
                analyze_local_files()
            elif choice == '4':
                start_robot_mode()
            elif choice == '5':
                show_system_info()
            elif choice == '6':
                print("\n👋 感谢使用，再见！\n")
                sys.exit(0)
            else:
                print("\n✗ 无效的选择，请重试\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 已退出程序")
            sys.exit(0)
        except Exception as e:
            print(f"\n✗ 发生错误: {e}\n")

if __name__ == '__main__':
    main()
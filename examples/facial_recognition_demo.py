#!/usr/bin/env python3
"""面部表情识别演示脚本"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vision.face_detector import FaceDetector
from vision.expression_analyzer import ExpressionAnalyzer
from vision import utils

def demo_image_recognition(image_path: str):
    """
    演示：从图像文件进行面部表情识别
    
    Args:
        image_path: 图像文件路径
    """
    print("="*60)
    print("面部表情识别演示 - 图像模式")
    print("="*60)
    
    # 初始化检测器和分析器
    print("\n[1] 初始化模型...")
    detector = FaceDetector(model_type='opencv')  # 使用 OpenCV 用于快速演示
    analyzer = ExpressionAnalyzer(use_pretrained=True)
    print("✓ 模型初始化完成")
    
    # 读取图像
    print(f"\n[2] 读取图像: {image_path}")
    if not os.path.exists(image_path):
        print(f"✗ 图像文件不存在: {image_path}")
        print("\n使用演示模式生成测试图像...")
        image = create_demo_image()
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"✓ 图像加载成功，大小: {image.shape}")
    
    # 检测人脸
    print("\n[3] 检测人脸...")
    faces = detector.detect_faces(image=image)
    print(f"✓ 检测到 {len(faces)} 个人脸")
    
    if len(faces) == 0:
        print("未检测到人脸，演示结束")
        return
    
    # 分析每个人脸的表情
    print("\n[4] 分析人脸表情...")
    results = []
    for i, face in enumerate(faces, 1):
        bbox = face['bbox']
        print(f"  人脸 {i}:")
        print(f"    - 位置: {bbox}")
        
        # 提取人脸区域
        x, y, w, h = bbox
        face_region = image[y:y+h, x:x+w]
        
        # 分析表情
        expression = analyzer.analyze_expression(face_region, face.get('landmarks'))
        face['expression'] = expression
        results.append(face)
        
        print(f"    - 情绪: {expression['emotion']}")
        print(f"    - 置信度: {expression['confidence']:.2%}")
        print(f"    - 详细得分:")
        for emotion, score in expression['scores'].items():
            print(f"        {emotion:12s}: {score:.2%}")
    
    # 绘制结果
    print("\n[5] 生成标注图像...")
    annotated_image = detector.draw_faces_on_image(image, results)
    
    # 为每个人脸添加表情标注
    for face in results:
        bbox = face['bbox']
        x, y, w, h = bbox
        emotion = face['expression']['emotion']
        confidence = face['expression']['confidence']
        
        # 在人脸上方显示情绪
        text = f"{emotion} ({confidence:.1%})"
        cv2.putText(annotated_image, text, (x, y-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 保存结果
    output_path = "facial_recognition_result.jpg"
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, annotated_image_bgr)
    print(f"✓ 结果已保存到: {output_path}")
    
    # 显示统计信息
    print("\n" + "="*60)
    print("分析结果统计")
    print("="*60)
    emotion_counts = {}
    for face in results:
        emotion = face['expression']['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print(f"总检测人数: {len(results)}")
    print(f"情绪分布:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"  - {emotion:12s}: {count} 人 ({percentage:.1f}%)")
    
    return results

def demo_video_recognition(video_path: str = None, use_webcam: bool = True):
    """
    演示：从视频或摄像头进行实时面部表情识别
    
    Args:
        video_path: 视频文件路径
        use_webcam: 是否使用摄像头
    """
    print("="*60)
    print("面部表情识别演示 - 视频/摄像头模式")
    print("="*60)
    
    # 初始化
    print("\n[1] 初始化模型...")
    detector = FaceDetector(model_type='opencv')
    analyzer = ExpressionAnalyzer(use_pretrained=True)
    print("✓ 模型初始化完成")
    
    # 打开视频源
    print("\n[2] 打开视频源...")
    if use_webcam:
        cap = cv2.VideoCapture(0)
        print("✓ 摄像头已打开 (按 'q' 退出)")
    else:
        if not os.path.exists(video_path):
            print(f"✗ 视频文件不存在: {video_path}")
            return
        cap = cv2.VideoCapture(video_path)
        print(f"✓ 视频文件已打开: {video_path}")
    
    frame_count = 0
    emotion_history = []
    
    print("\n[3] 开始处理视频帧...")
    print("按 'q' 键退出\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测人脸
        faces = detector.detect_faces(image=frame_rgb)
        
        # 分析表情
        for face in faces:
            bbox = face['bbox']
            x, y, w, h = bbox
            face_region = frame_rgb[y:y+h, x:x+w]
            
            try:
                expression = analyzer.analyze_expression(face_region)
                emotion = expression['emotion']
                confidence = expression['confidence']
                emotion_history.append(emotion)
                
                # 绘制边界框
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 绘制情绪标签
                text = f"{emotion} ({confidence:.1%})"
                cv2.putText(frame, text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"处理人脸失败: {e}")
        
        # 显示帧数
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示人脸数
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示视频
        cv2.imshow('Facial Expression Recognition', frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 统计
    if emotion_history:
        print("\n" + "="*60)
        print("统计信息")
        print("="*60)
        print(f"总处理帧数: {frame_count}")
        print(f"检测到表情的帧数: {len(emotion_history)}")
        
        emotion_counts = {}
        for emotion in emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"\n情绪分布:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(emotion_history)) * 100
            print(f"  - {emotion:12s}: {count} 帧 ({percentage:.1f}%)")

def create_demo_image(width: int = 400, height: int = 400):
    """
    创建一个演示图像
    """
    # 创建白色背景
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 绘制简单的人脸
    center_x, center_y = width // 2, height // 2
    
    # 脸部圆形
    cv2.circle(image, (center_x, center_y), 100, (200, 150, 100), -1)
    
    # 眼睛
    cv2.circle(image, (center_x - 40, center_y - 30), 15, (0, 0, 0), -1)
    cv2.circle(image, (center_x + 40, center_y - 30), 15, (0, 0, 0), -1)
    
    # 笑脸
    cv2.ellipse(image, (center_x, center_y + 30), (50, 40), 0, 0, 180, (0, 0, 0), 3)
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='面部表情识别演示')
    parser.add_argument('--mode', choices=['image', 'video', 'webcam'], 
                       default='webcam',
                       help='运行模式')
    parser.add_argument('--input', type=str, help='输入文件路径')
    
    args = parser.parse_args()
    
    if args.mode == 'image':
        input_path = args.input or 'test_image.jpg'
        demo_image_recognition(input_path)
    elif args.mode == 'video':
        input_path = args.input or 'test_video.mp4'
        demo_video_recognition(video_path=input_path, use_webcam=False)
    else:  # webcam
        demo_video_recognition(use_webcam=True)
    
    print("\n演示完成！")
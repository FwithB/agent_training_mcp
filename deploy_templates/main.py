import os
import glob
import io
import json
import base64
import traceback

import numpy as np
from PIL import Image
from ultralytics import YOLO

def find_latest_model(model_dir):
    pt_files = glob.glob(os.path.join(model_dir, "*.pt"))
    if not pt_files:
        return None
    return max(pt_files, key=os.path.getmtime)

# 假设在当前目录下有一个 models 子目录，用来放自定义 .pt 模型
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "models")
latest_model_path = find_latest_model(model_dir)

if latest_model_path:
    model = YOLO(latest_model_path)
else:
    # 如果找不到自定义模型，使用官方默认的 yolov8n.pt
    model = YOLO('yolov8n.pt')

def handler(context, event):
    try:
        # 记录请求开始
        context.logger.info("开始处理检测请求")
        
        # 解析请求数据
        data = event.body
        if isinstance(data, bytes):
            try:
                data = json.loads(data.decode('utf-8'))
            except UnicodeDecodeError:
                context.logger.warning("UTF-8解码失败，尝试使用latin-1编码")
                data = json.loads(data.decode('latin-1'))
        
        if 'image' not in data:
            raise ValueError("Missing 'image' field in request data.")
            
        # 记录图像处理开始
        context.logger.info("开始解码图像")
        
        # 解码图像，增加错误处理
        try:
            img_bytes = base64.b64decode(data["image"])
        except Exception as decode_err:
            context.logger.error(f"Base64解码失败: {str(decode_err)}")
            raise ValueError(f"无法解码图像: {str(decode_err)}")
            
        threshold = float(data.get("threshold", 0.5))
        
        # 图像处理和推理，增加错误处理
        try:
            img = Image.open(io.BytesIO(img_bytes))
            
            # 确保图像格式正确
            if img.mode != 'RGB':
                context.logger.info(f"转换图像格式从 {img.mode} 到 RGB")
                img = img.convert('RGB')
                
            img_array = np.array(img)
            
            # 记录图像尺寸
            context.logger.info(f"图像尺寸: {img.size}, 数组形状: {img_array.shape}")
            
            # 运行推理
            results = model(img_array)
        except Exception as img_err:
            context.logger.error(f"图像处理或推理失败: {str(img_err)}")
            context.logger.error(traceback.format_exc())
            raise ValueError(f"图像处理错误: {str(img_err)}")
            
        # 解析结果
        detections = []
        for r in results:
            for box in r.boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    if conf < threshold:
                        continue
                    detections.append({
                        "confidence": float(conf),
                        "label": model.names[cls_id],
                        "points": [float(x1), float(y1), float(x2), float(y2)],
                        "type": "rectangle"
                    })
                except Exception as box_err:
                    context.logger.error(f"处理检测框时出错: {str(box_err)}")
                    # 继续处理下一个框
                
        context.logger.info(f"检测完成，找到 {len(detections)} 个目标")
        
        return context.Response(
            body=json.dumps(detections),
            headers={},
            content_type='application/json',
            status_code=200
        )
    except Exception as e:
        context.logger.error(f"推理错误: {str(e)}")
        context.logger.error(traceback.format_exc())
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type='application/json',
            status_code=500
        )
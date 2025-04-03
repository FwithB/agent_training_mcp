# train.py
from ultralytics import YOLO
import os

# 设置为UTF-8编码
os.environ["PYTHONIOENCODING"] = "utf-8"

def train_yolo(model_type='yolov8n', epochs=1, data='coco128.yaml'):
    """
    训练YOLOv8模型的简易函数
    
    :param model_type: YOLO模型类型，如yolov8n.pt, yolov8m.pt
    :param epochs: 训练轮数
    :param data: 数据集配置文件
    :return: dict，包含训练状态和信息
    """
    try:
        # 加载预训练模型
        model_path = f"{model_type}.pt" if not model_type.endswith(".pt") else model_type
        model = YOLO(model_path)

        # 训练
        results = model.train(
            data=data,         # 使用内置配置或自定义配置路径
            epochs=epochs,
            imgsz=640,         # 可调
            name=f'{model_type}_custom'
        )

        return {
            "status": "success",
            "message": f"开始训练 {model_type}，共 {epochs} 轮，使用 {data} 数据集。",
            "details": str(results)  # 这里可以返回更多信息，比如日志路径、权重保存路径等
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"训练过程出现错误: {str(e)}"
        }

if __name__ == '__main__':
    # 如果需要在命令行直接调用该脚本
    # 这部分可根据需要保留或移除
    result = train_yolo('yolov8n', 1, 'coco128.yaml')
    print(result)

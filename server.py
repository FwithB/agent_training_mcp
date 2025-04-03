# server.py
from flask import Flask, request, jsonify
import json
import multiprocessing
import subprocess
import os
import sys

# 引入我们原先的训练函数（在train.py中定义）
# 假设我们在同一目录下，所以直接import
from train import train_yolo

app = Flask(__name__)

# 示例：定义一个 /train 的 POST 接口，接收训练请求
@app.route('/train', methods=['POST'])
def train_endpoint():
    try:
        # 解析JSON请求体
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "缺少JSON请求体"}), 400
        
        # 从JSON中获取训练参数（如果没有，设置默认值）
        model_type = data.get('model_type', 'yolov8n')
        epochs = data.get('epochs', 1)
        dataset = data.get('data', 'coco128.yaml')
        
        # 调用训练函数
        result = train_yolo(model_type, epochs, dataset)
        
        # 将训练结果以JSON形式返回
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # 在0.0.0.0上监听，端口可自由指定，示例为5000
    # debug=True仅为开发调试使用
    app.run(host='0.0.0.0', port=5000, debug=True)

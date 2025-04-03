# main.py - MCP客户端
import requests
import json
import os
import sys
import logging
from openai import OpenAI  # 需要安装: pip install openai

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_client.log')
    ]
)
logger = logging.getLogger('mcp_client')

# 设置API密钥 - 从环境变量获取
API_KEY = "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"
if not API_KEY:
    logger.warning("未设置OPENROUTER_API_KEY环境变量，LLM解析功能可能无法正常工作")
    logger.warning("请设置环境变量: export OPENROUTER_API_KEY=你的密钥")

# OpenRouter基础URL
BASE_URL = "https://openrouter.ai/api/v1"

# 初始化OpenAI客户端 (OpenRouter兼容OpenAI API)
client = None
if API_KEY:
    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY
        )
    except Exception as e:
        logger.error(f"初始化OpenAI客户端失败: {str(e)}")

def parse_instruction_with_llm(user_query):
    """
    使用LLM解析用户的自然语言指令，转换为结构化参数
    
    Args:
        user_query: 用户输入的自然语言指令，例如"训练一个yolov8n模型，用coco数据集跑3轮"
        
    Returns:
        dict: 包含解析后的训练参数，如model_type, epochs, data
    """
    # 如果客户端未初始化，返回默认参数
    if client is None:
        logger.error("LLM客户端未初始化，无法解析指令")
        return {"model_type": "yolov8n", "epochs": 1, "data": "coco128.yaml"}
    
    try:
        # 定义函数调用格式
        functions = [
            {
                "name": "train_yolo",
                "description": "训练YOLOv8模型的函数",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "YOLO模型类型，例如yolov8n、yolov8s、yolov8m、yolov8l、yolov8x等"
                        },
                        "epochs": {
                            "type": "integer",
                            "description": "训练轮数，默认为1"
                        },
                        "data": {
                            "type": "string",
                            "description": "数据集配置文件路径，如coco128.yaml"
                        }
                    },
                    "required": []
                }
            }
        ]
        
        logger.info(f"发送指令到LLM进行解析: '{user_query}'")
        # 调用LLM API进行解析
        response = client.chat.completions.create(
            model="openrouter/anthropic/claude-3-haiku-20240307",  # 可根据需要替换为其他模型
            messages=[
                {
                    "role": "system", 
                    "content": "你是一个专业的计算机视觉助手，负责将用户的自然语言指令解析为YOLO训练参数。"
                },
                {
                    "role": "user", 
                    "content": user_query
                }
            ],
            tools=[{"type": "function", "function": f} for f in functions],
            tool_choice={"type": "function", "function": {"name": "train_yolo"}}
        )
        
        # 安全地提取函数调用参数
        tool_calls = getattr(response.choices[0].message, 'tool_calls', None)
        
        # 检查tool_calls是否存在且非空
        if not tool_calls:
            logger.warning("LLM响应中没有tool_calls字段")
            return {"model_type": "yolov8n", "epochs": 1, "data": "coco128.yaml"}
            
        # 检查是否有函数调用
        if len(tool_calls) == 0:
            logger.warning("LLM响应中tool_calls为空列表")
            return {"model_type": "yolov8n", "epochs": 1, "data": "coco128.yaml"}
            
        # 检查是否为预期的函数名
        if tool_calls[0].function.name != "train_yolo":
            logger.warning(f"预期函数'train_yolo'，但收到'{tool_calls[0].function.name}'")
            return {"model_type": "yolov8n", "epochs": 1, "data": "coco128.yaml"}
        
        # 尝试解析JSON参数
        try:
            parameters = json.loads(tool_calls[0].function.arguments)
            logger.info(f"成功解析LLM响应: {parameters}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}")
            logger.error(f"原始参数字符串: {tool_calls[0].function.arguments}")
            return {"model_type": "yolov8n", "epochs": 1, "data": "coco128.yaml"}
        
        # 确保所有必要参数都存在，设置默认值
        default_params = {"model_type": "yolov8n", "epochs": 1, "data": "coco128.yaml"}
        for key, default_value in default_params.items():
            if key not in parameters or parameters[key] is None:
                logger.info(f"参数 '{key}' 未提供，使用默认值: {default_value}")
                parameters[key] = default_value
        
        # 确保epochs是整数
        try:
            parameters["epochs"] = int(parameters["epochs"])
        except (ValueError, TypeError):
            logger.warning(f"epochs参数无效: {parameters.get('epochs')}，使用默认值: 1")
            parameters["epochs"] = 1
            
        return parameters
        
    except Exception as e:
        logger.exception(f"解析指令时出现异常: {str(e)}")
        # 出错时返回默认值
        return {"model_type": "yolov8n", "epochs": 1, "data": "coco128.yaml"}

def send_train_request(model_type='yolov8n', epochs=1, data='coco128.yaml', max_retries=2):
    """
    向服务器发送训练请求，并返回结果
    
    Args:
        model_type: YOLO模型类型
        epochs: 训练轮数
        data: 数据集配置文件
        max_retries: 最大重试次数
        
    Returns:
        dict: 服务器返回的结果
    """
    url = 'http://localhost:5000/train'  # 假设服务器跑在本机5000端口
    payload = {
        "model_type": model_type,
        "epochs": epochs,
        "data": data
    }
    
    logger.info(f"准备向服务器 {url} 发送训练请求: {payload}")
    
    retries = 0
    while retries <= max_retries:
        try:
            logger.info(f"发送请求 (尝试 {retries+1}/{max_retries+1})...")
            response = requests.post(url, json=payload, timeout=180)  # 增加超时时间，因为训练可能需要较长时间
            
            # 检查响应状态码
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info("请求成功，服务器返回结果")
                    return result
                except json.JSONDecodeError:
                    logger.error(f"无法解析服务器响应为JSON: {response.text}")
                    return {
                        "status": "error",
                        "message": "无法解析服务器响应",
                        "details": response.text[:200] + "..." if len(response.text) > 200 else response.text
                    }
            else:
                logger.error(f"服务器返回错误状态码: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"服务端错误，状态码: {response.status_code}",
                    "details": response.text[:200] + "..." if len(response.text) > 200 else response.text
                }
                
        except requests.exceptions.Timeout:
            retries += 1
            if retries <= max_retries:
                logger.warning(f"请求超时，将在5秒后重试...")
                import time
                time.sleep(5)
            else:
                logger.error("请求超时，已达到最大重试次数")
                return {
                    "status": "error",
                    "message": "请求超时",
                    "details": f"服务器在{180}秒内没有响应"
                }
                
        except requests.exceptions.ConnectionError:
            logger.error(f"连接错误：无法连接到服务器 {url}")
            return {
                "status": "error",
                "message": "无法连接到服务器",
                "details": f"请确保服务器正在运行并且可以通过 {url} 访问"
            }
            
        except Exception as e:
            logger.exception(f"请求过程中发生未预期的错误: {str(e)}")
            return {
                "status": "error",
                "message": f"请求异常: {str(e)}",
                "details": "查看日志文件获取更多信息"
            }

def validate_input(prompt, validator=None, default=None):
    """
    通用输入验证函数
    
    Args:
        prompt: 提示用户的文本
        validator: 验证函数，返回True/False
        default: 默认值
        
    Returns:
        验证通过的用户输入或默认值
    """
    while True:
        value = input(prompt).strip()
        
        # 如果用户未输入且有默认值，返回默认值
        if not value and default is not None:
            return default
            
        # 如果没有验证函数或验证通过，返回值
        if validator is None or validator(value):
            return value
            
        # 否则提示错误并重新请求输入
        print("输入无效，请重新输入")

def main():
    """主函数，处理用户交互和请求发送"""
    print("="*50)
    print("YOLO训练系统 - MCP客户端")
    print("="*50)
    
    # 检查API密钥
    if not API_KEY:
        print("\n警告: 未设置OPENROUTER_API_KEY环境变量")
        print("自然语言指令解析功能将不可用，只能使用手动参数模式")
    
    while True:
        print("\n请选择操作：")
        if API_KEY:
            print("1. 使用自然语言指令训练模型")
        print("2. 手动指定训练参数")
        print("3. 退出程序")
        
        valid_choices = ['2', '3']
        if API_KEY:
            valid_choices.append('1')
        
        choice = validate_input(
            "请输入选项 " + (
                "[1-3]: " if API_KEY else "[2-3]: "
            ), 
            lambda x: x in valid_choices
        )
        
        if choice == '1' and API_KEY:
            # 自然语言指令模式
            user_query = validate_input("\n请输入您的训练指令 (例如：'训练一个yolov8n模型识别猫狗，训练5轮'): ", 
                                      lambda x: len(x) > 0, 
                                      None)
            
            print("\n正在解析您的指令...")
            params = parse_instruction_with_llm(user_query)
            
            print("\n解析结果:")
            print(f"- 模型类型: {params['model_type']}")
            print(f"- 训练轮数: {params['epochs']}")
            print(f"- 数据集: {params['data']}")
            
            confirm = validate_input("\n确认开始训练? (y/n): ", 
                                  lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                                  'n')
                                  
            if confirm.lower() in ['y', 'yes']:
                result = send_train_request(
                    params['model_type'], 
                    params['epochs'], 
                    params['data']
                )
                print("\n服务器返回结果:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("已取消训练请求")
                
        elif choice == '2':
            # 手动参数模式
            print("\n请输入训练参数:")
            model_type = validate_input("模型类型 (默认 yolov8n): ", None, "yolov8n")
            
            epochs = validate_input("训练轮数 (默认 1): ", 
                                 lambda x: x.isdigit() and int(x) > 0, 
                                 "1")
            epochs = int(epochs)
            
            data = validate_input("数据集配置 (默认 coco128.yaml): ", None, "coco128.yaml")
            
            print(f"\n参数确认: 模型={model_type}, 轮数={epochs}, 数据集={data}")
            confirm = validate_input("确认开始训练? (y/n): ", 
                                  lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                                  'n')
            
            if confirm.lower() in ['y', 'yes']:
                result = send_train_request(model_type, epochs, data)
                print("\n服务器返回结果:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("已取消训练请求")
                
        elif choice == '3':
            # 退出程序
            print("感谢使用YOLO训练系统，再见！")
            logger.info("用户选择退出程序")
            return
            
        else:
            print("无效选项，请重新选择")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        logger.info("程序被用户中断(KeyboardInterrupt)")
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        logger.exception("程序意外终止")
        sys.exit(1)
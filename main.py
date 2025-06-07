""" MCP Client - Handle user interaction, parse commands and send requests """
import requests
import json
import os
import sys
import logging
import re
from openai import OpenAI  # Added
import cvat_api
from deploy_to_cvat import get_latest_model_from_runs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_client.log')
    ]
)
logger = logging.getLogger('mcp_client')

logger = logging.getLogger(__name__)

# Assume TRAIN_ENDPOINT is already defined in configuration, for example:
TRAIN_ENDPOINT = "http://localhost:5000/train"

# Replace API configuration
API_KEY = "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"
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")  # Added

if not API_KEY:
    logger.warning("API key not set, LLM parsing functionality may not work properly")

# Server configuration
SERVER_URL = "http://localhost:5000"  # Can be modified to the actual server address
TRAIN_ENDPOINT = f"{SERVER_URL}/train"
BROWSER_ENDPOINT = f"{SERVER_URL}/browser"


def list_available_datasets():
    """
    List all available datasets (containing data.yaml) in the current datasets directory,
    sorted by modification time in descending order.
    Returns a list, each element is:
    {
      "name": <dataset directory name>,
      "path": <absolute path to data.yaml>,
      "modified_time": <ISO format string>
    }
    """
    from datetime import datetime
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    if not os.path.exists(datasets_dir):
        return []
    
    dataset_list = []
    for d in os.listdir(datasets_dir):
        subdir = os.path.join(datasets_dir, d)
        if os.path.isdir(subdir):
            yaml_file = os.path.join(subdir, "data.yaml")
            if os.path.exists(yaml_file):
                mtime = os.path.getmtime(yaml_file)
                dataset_list.append({
                    "name": d,
                    "path": os.path.abspath(yaml_file),
                    "modified_time": datetime.fromtimestamp(mtime).isoformat()
                })
    dataset_list.sort(key=lambda x: x["modified_time"], reverse=True)
    return dataset_list

def fallback_keyword_match(user_query):
    """
    When LLM response parsing fails, use keyword matching as a fallback to return default commands
    """
    if any(kw in user_query.lower() for kw in ["train", "yolo", "model"]):
        return {"operation": "train", "params": {"model_type": "yolov8n", "epochs": 1, "data": "coco128.yaml"}}
    elif any(kw in user_query.lower() for kw in ["browser", "web", "cvat"]):
        return {"operation": "browser", "params": {"action": "open_cvat", "browser_type": "msedge"}}
    elif any(kw in user_query.lower() for kw in ["deploy", "upload model"]):
        return {"operation": "deploy", "params": {"force": False}}
    elif any(kw in user_query.lower() for kw in ["upload data", "upload image", "upload training set"]):
        return {"operation": "upload", "params": {}}
    elif any(kw in user_query.lower() for kw in ["auto annotation", "inference", "mark data"]):
        return {"operation": "inference", "params": {}}
    elif any(kw in user_query.lower() for kw in ["view annotation", "open annotation", "view task", "annotation interface"]):
        return {"operation": "view_annotation", "params": {}}
    elif any(kw in user_query.lower() for kw in ["download data", "download dataset", "export data"]):
        return {"operation": "download_dataset", "params": {}}
    elif any(kw in user_query.lower() for kw in ["convert dataset", "convert format", "cvat to yolo"]):
        return {"operation": "convert_dataset", "params": {}}
    else:
        return {"operation": "unknown", "params": {}}

def parse_llm_response(response, user_query):
    """
    Try to parse structured JSON from the LLM response.
    Prioritize checking function_call and tool_calls, using attribute access instead of dictionary indexing.
    """
    try:
        # Use attribute access
        msg = response.choices[0].message  
        if not msg:
            return None

        # If function_call exists, parse its arguments
        if hasattr(msg, "function_call") and msg.function_call:
            if hasattr(msg.function_call, "arguments") and msg.function_call.arguments:
                raw_args = msg.function_call.arguments
                parsed = json.loads(raw_args)
                return parsed

        # If there's no function_call but has tool_calls, use the first tool_call's arguments
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            raw_args = msg.tool_calls[0].function.arguments
            parsed = json.loads(raw_args)
            return parsed

        # Otherwise try to directly parse message.content
        content = msg.content or ""
        content = content.strip()
        if content:
            parsed = json.loads(content)
            return parsed

    except Exception as e:
        logger.warning(f"Error parsing LLM response: {e}")
    return None

def parse_instruction_with_llm(user_query):
    """
    Use OpenRouter API to parse user's natural language instructions, and return structured JSON instructions.
    If the data field for train operation in the parsing result is invalid,
    automatically call list_available_datasets to supplement the latest dataset path.
    """
    system_prompt = (
        "You are a professional computer vision assistant responsible for parsing user instructions.\n"
        "When the user requests to train a model, return an operation of 'train' with params containing model_type, epochs, and data fields.\n"
        "If the user requests to use the latest dataset, first call the 'list_datasets' tool to get information about currently available datasets, then set data to the absolute path of the latest dataset configuration file when returning the train instruction.\n"
        "When the user requests to view the annotation interface, return an operation of 'view_annotation' with params containing task_id and job_id fields.\n"
        "When the user requests to convert dataset formats, return an operation of 'convert_dataset' with params that can be empty.\n"
        "For example, input: 'Train model with yolov8n using our latest data' should return:\n"
        "{\"operation\": \"train\", \"params\": {\"model_type\": \"yolov8n\", \"epochs\": 1, \"data\": \"C:\\\\...\\\\datasets\\\\19mission\\\\data.yaml\"}}\n"
        "For example, input: 'View annotation interface for task 14, job is 3' should return:\n"
        "{\"operation\": \"view_annotation\", \"params\": {\"task_id\": \"14\", \"job_id\": \"3\"}}\n"
        "For example, input: 'Convert dataset format' should return:\n"
        "{\"operation\": \"convert_dataset\", \"params\": {}}\n"
        "Make sure the data parameter includes the .yaml suffix, and ensure all relevant parameters are parsed."
        "When the user requests to download and convert a dataset, return an operation of 'download_and_convert' with params containing the task_id field.\n"
        "For example, input: 'Download and convert data from task 14 in one click' should return:\n"
        "{\"operation\": \"download_and_convert\", \"params\": {\"task_id\": \"14\"}}\n"
    )
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "parse_command",
                "description": "Parse user instructions into operations and parameters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": [
                                "train", "browser", "deploy", "upload",
                                "inference", "view_annotation",
                                "download_dataset", "convert_dataset",
                                "download_and_convert"  # Added new operation
                            ],
                            "description": "Operation type"
                        },
                        "params": {
                            "type": "object",
                            "properties": {
                                "model_type": {"type": "string"},
                                "epochs": {"type": "integer"},
                                # Upload data
                                "data_path": {
                                    "type": "string",
                                    "description": "Local file or directory path to upload, or training dataset configuration file path (.yaml suffix)"
                                },
                                "task_name": {
                                    "type": "string",
                                    "description": "Name to use when creating CVAT task"
                                },
                                # View annotation
                                "task_id": {"type": "string"},
                                "job_id": {"type": "string"},
                                # Download dataset
                                "format": {
                                    "type": "string",
                                    "description": "Download format, such as CVAT, COCO, etc."
                                },
                                # Dataset conversion
                                "dataset_name": {
                                    "type": "string",
                                    "description": "Name of the converted dataset"
                                }
                            },
                            "required": []
                        }
                    },
                    "required": ["operation", "params"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "parse_command"}}
    )
    
    logger.info(f"GPT-3.5 complete response:\n{json.dumps(response, default=str, ensure_ascii=False, indent=2)}")
    
    parsed = parse_llm_response(response, user_query)
    if parsed is None:
        logger.warning("Unable to parse LLM response, using keyword matching")
        parsed = fallback_keyword_match(user_query)
    
    # If operation is train, check if the data field is valid
    if parsed.get("operation") == "train":
        params_out = parsed.get("params", {})
        data_field = params_out.get("data", "").strip()
        # If the data field doesn't exist or the file doesn't exist, automatically call list_available_datasets
        if not data_field or not os.path.exists(data_field):
            try:
                from tools import list_available_datasets
            except ImportError:
                logger.warning("Cannot import list_available_datasets, please ensure the function is correctly defined in the tools module")
                list_available_datasets = globals().get("list_available_datasets")
            if list_available_datasets:
                datasets = list_available_datasets()
                if datasets and len(datasets) > 0:
                    params_out["data"] = datasets[0]["path"]
                    logger.info(f"Automatically selected latest dataset: {params_out['data']}")
                else:
                    logger.warning("No available datasets found, please convert a dataset first.")
            else:
                logger.warning("list_available_datasets function was not imported successfully.")
        parsed["params"] = params_out
    
    return parsed


def process_upload_operation(params):
    """Process upload data operation (only creates a new task, compatible with data_path and task_name)"""
    # Read parameters
    data_path = params.get('data_path')
    task_name = params.get('task_name')

    # If the specified path doesn't exist, ignore and use automatic search
    if data_path and not os.path.exists(data_path):
        logger.warning(f"Specified path does not exist: {data_path}, will use automatic search instead")
        data_path = None

    # Print results
    print("\nParsing result (upload data operation):")
    print(f"- Data path: {data_path or 'automatic search'}")
    print(f"- Task name: {task_name or 'default name'}")

    confirm = validate_input(
        "\nConfirm upload data to CVAT? (y/n): ",
        lambda x: x.lower() in ['y','n','yes','no'],
        'y'
    )

    if confirm.lower() in ['y','yes']:
        print_info("Uploading data to CVAT...")
        result = cvat_api.upload_data_to_cvat(
            data_path = data_path,
            task_name = task_name
        )
        if result["status"] == "success":
            print_success(f"Upload successful: {result['message']}")
            print_info(f"Task ID: {result['task_id']}")
        else:
            print_error(f"Upload failed: {result['message']}")
        return result
    else:
        print("Upload request canceled")
        return {"status": "canceled", "message": "User canceled the operation"}

def send_train_request(params, max_retries=2):
    """
    Send training request to the server. Note that this code only passes parameters to the server,
    actual training is executed by the Ultralytics YOLO library.
    """
    # Construct weight filename based on model type, e.g., "yolov8n" corresponds to "yolov8n.pt"
    model_type = params.get("model_type", "yolov8n")
    data_file = params.get("data", "coco128.yaml")
    weight_file = f"{model_type}.pt"

    # Smart dataset path handling: if data_file is neither an absolute path nor exists, automatically look in the datasets directory
    if not os.path.isabs(data_file) and not os.path.exists(data_file):
        datasets_path = os.path.join(os.getcwd(), "datasets")
        
        # 1) First check if there's a data.yaml in a directory with the same name
        dataset_name = os.path.splitext(os.path.basename(data_file))[0]
        dataset_yaml = os.path.join(datasets_path, dataset_name, "data.yaml")
        if os.path.exists(dataset_yaml):
            data_file = dataset_yaml
            logger.info(f"Found and using converted dataset configuration: {data_file}")
            params["data"] = data_file  # Update data in params
        else:
            # 2) If data.yaml doesn't exist in a directory with the same name, check if data_file exists directly in the datasets directory
            potential_path = os.path.join(datasets_path, data_file)
            if os.path.exists(potential_path):
                data_file = potential_path
                logger.info(f"Using data configuration from datasets directory: {data_file}")
                params["data"] = data_file

    logger.info(f"Final data configuration file used: {data_file}")

    # Check if pre-trained weight file exists in the current working directory
    if os.path.exists(weight_file):
        logger.info(f"Pre-trained weight file found locally: {weight_file}")
    else:
        logger.info(f"Local weight file not found: {weight_file}, will download from official site.")

    # Check if data file exists
    if os.path.exists(data_file):
        logger.info(f"Data configuration file found locally: {data_file}")
    else:
        logger.error(f"Data configuration file not found: {data_file}. Please verify the data file location.")
        return {"status": "error", "message": f"Data configuration file does not exist: {data_file}"}
    
    # Print parameters before sending training request
    logger.info(f"Preparing to send training request to server {TRAIN_ENDPOINT}: {params}")
    
    retries = 0
    while retries <= max_retries:
        try:
            logger.info(f"Sending request (attempt {retries+1}/{max_retries+1})...")
            response = requests.post(TRAIN_ENDPOINT, json=params, timeout=180)
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info("Request successful, server returned results")
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Unable to parse server response as JSON: {response.text}")
                    return {"status": "error", "message": "Unable to parse server response", "details": response.text[:200]}
            else:
                logger.error(f"Server returned error status code: {response.status_code}")
                return {"status": "error", "message": f"Server error, status code: {response.status_code}", "details": response.text[:200]}
        except requests.exceptions.Timeout:
            retries += 1
            if retries <= max_retries:
                logger.warning("Request timed out, will retry in 5 seconds...")
                import time
                time.sleep(5)
            else:
                logger.error("Request timed out, maximum retry attempts reached")
                return {"status": "error", "message": "Request timed out", "details": f"Server did not respond within {180} seconds"}
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error: Unable to connect to server {TRAIN_ENDPOINT}")
            return {"status": "error", "message": "Unable to connect to server", "details": f"Please ensure the server is running and accessible via {TRAIN_ENDPOINT}"}
        except Exception as e:
            logger.exception(f"Exception occurred during request: {str(e)}")
            return {"status": "error", "message": f"Request exception: {str(e)}", "details": "Check log file for more information"}
        
def send_browser_request(params, max_retries=2):
    """
    Send browser control requests to the server
    
    Args:
        params: Dictionary of browser control parameters
        max_retries: Maximum number of retry attempts
        
    Returns:
        dict: Results returned from the server
    """
    logger.info(f"Preparing to send browser control request to server {BROWSER_ENDPOINT}: {params}")
    
    retries = 0
    while retries <= max_retries:
        try:
            logger.info(f"Sending request (attempt {retries+1}/{max_retries+1})...")
            response = requests.post(BROWSER_ENDPOINT, json=params, timeout=30)
            
            # Check response status code
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info("Request successful, server returned results")
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Unable to parse server response as JSON: {response.text}")
                    return {
                        "status": "error",
                        "message": "Unable to parse server response",
                        "details": response.text[:200] + "..." if len(response.text) > 200 else response.text
                    }
            else:
                logger.error(f"Server returned error status code: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"Server error, status code: {response.status_code}",
                    "details": response.text[:200] + "..." if len(response.text) > 200 else response.text
                }
                
        except requests.exceptions.Timeout:
            retries += 1
            if retries <= max_retries:
                logger.warning(f"Request timed out, will retry in 5 seconds...")
                import time
                time.sleep(5)
            else:
                logger.error("Request timed out, maximum retry attempts reached")
                return {
                    "status": "error",
                    "message": "Request timed out",
                    "details": f"Server did not respond within 30 seconds"
                }
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error: Unable to connect to server {BROWSER_ENDPOINT}")
            return {
                "status": "error",
                "message": "Unable to connect to server",
                "details": f"Please ensure the server is running and accessible via {BROWSER_ENDPOINT}"
            }
            
        except Exception as e:
            logger.exception(f"Unexpected error occurred during request: {str(e)}")
            return {
                "status": "error",
                "message": f"Request exception: {str(e)}",
                "details": "Check log file for more information"
            }


def send_deploy_request(params, max_retries=2):
    """Send model deployment request to the server"""
    import time  # Ensure time module is imported at the beginning of the function
    logger.info(f"Preparing to send model deployment request to server: {params}")
    
    # If it's a Windows format absolute path, copy to temporary directory
    model_path = params.get('model_path')
    if model_path and os.path.isabs(model_path) and os.name == 'nt':
        import shutil
        temp_model_name = f"Model_{int(time.time())}.pt"
        temp_model_path = os.path.join(os.getcwd(), temp_model_name)
        try:
            shutil.copy2(model_path, temp_model_path)
            logger.info(f"Model copied to temporary location: {temp_model_path}")
            # Update request parameters path to relative path
            params = params.copy()  # Create a copy to avoid modifying original parameters
            params['model_path'] = temp_model_name
        except Exception as e:
            logger.error(f"Failed to copy model file to temporary location: {str(e)}")
            # Continue using original path
    
    retries = 0
    while retries <= max_retries:
        try:
            response = requests.post(f"{SERVER_URL}/deploy", json=params, timeout=6000)
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Deployment request successful, server returned results")
                
                # Clean up temporary files
                if 'temp_model_path' in locals() and os.path.exists(temp_model_path):
                    try:
                        os.remove(temp_model_path)
                        logger.info(f"Temporary model file cleaned up: {temp_model_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary file: {str(e)}")
                        
                return result
            else:
                logger.error(f"Server returned error status code: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"Server error, status code: {response.status_code}",
                    "details": response.text[:200]
                }
                
        except Exception as e:
            retries += 1
            if retries <= max_retries:
                logger.warning(f"Request exception, will retry in 5 seconds: {str(e)}")
                time.sleep(5)
            else:
                logger.error(f"Deployment request failed: {str(e)}")
                # Clean up temporary files
                if 'temp_model_path' in locals() and os.path.exists(temp_model_path):
                    try:
                        os.remove(temp_model_path)
                    except:
                        pass
                return {"status": "error", "message": f"Request exception: {str(e)}"}

def send_open_annotation_request(params, max_retries=2):
    """Send open annotation interface request to the server"""
    logger.info(f"Preparing to send open annotation interface request to server: {params}")
    
    retries = 0
    while retries <= max_retries:
        try:
            response = requests.post(f"{SERVER_URL}/open_annotation", json=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Open annotation interface request successful, server returned results")
                return result
            else:
                logger.error(f"Server returned error status code: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"Server error, status code: {response.status_code}",
                    "details": response.text[:200]
                }
                
        except Exception as e:
            retries += 1
            if retries <= max_retries:
                logger.warning(f"Request exception, will retry in 5 seconds: {str(e)}")
                import time
                time.sleep(5)
            else:
                logger.error(f"Open annotation interface request failed: {str(e)}")
                return {"status": "error", "message": f"Request exception: {str(e)}"}



def process_deploy_operation(params):
    """Process model deployment operation"""
    # If model_path is not provided, try to get the latest model from the training directory
    if params.get('model_path') is None:
        latest_model = get_latest_model_from_runs()
        if latest_model:
            params['model_path'] = latest_model
            print_info(f"Found latest trained model: {latest_model}")
        else:
            print_info("No model found in training directory, will use default location model")
    
    model_path_display = params.get('model_path') if params.get('model_path') is not None else "Automatically select latest model"
    print("\nParsing result (deployment operation):")
    print(f"- Model path: {model_path_display}")
    print(f"- Force deployment: {params.get('force', False)}")
    
    confirm = validate_input("\nConfirm deploy model to CVAT? (y/n): ", 
                          lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                          'y')
                          
    if confirm.lower() in ['y', 'yes']:
        print_info("Deploying model to CVAT...")
        result = send_deploy_request(params)
        
        print("\nServer returned result:")
        if result["status"] == "success":
            print_success(f"Model deployment successful: {result.get('function_name', '')}")
            if "details" in result:
                print_info("Deployment details:")
                print(result["details"])
        else:
            print_error(f"Model deployment failed: {result.get('message', '')}")
            
        return result
    else:
        print("Deployment request canceled")
        return {"status": "canceled", "message": "User canceled the operation"}


def validate_input(prompt, validator=None, default=None):
    """
    Generic input validation function
    
    Args:
        prompt: Text to prompt the user
        validator: Validation function, returns True/False
        default: Default value
        
    Returns:
        Validated user input or default value
    """
    while True:
        value = input(prompt).strip()
        
        # If user didn't input anything and there's a default value, return the default
        if not value and default is not None:
            return default
            
        # If there's no validation function or validation passes, return the value
        if validator is None or validator(value):
            return value
            
        # Otherwise prompt error and request input again
        print("Invalid input, please try again")

def process_train_operation(params):
    """
    Collect training parameters. If params already contains model_type, epochs, and data
    and the data file actually exists, use them directly; otherwise enter interactive
    input flow (including selecting recently converted datasets).
    """
    import os

    # ---------- 1. Quick check if LLM parameters are sufficient ----------
    # Model type
    if "model_type" in params and str(params["model_type"]).strip():
        model_type = str(params["model_type"]).strip()
    else:
        model_type = None

    # Training epochs
    if "epochs" in params:
        try:
            epochs = int(params["epochs"])
        except:
            epochs = None
    else:
        epochs = None

    # Dataset configuration file
    data_file = params.get("data", "").strip()
    data_valid = False
    if data_file:
        if os.path.exists(data_file):
            data_valid = True
        else:
            # Try using helper function to find converted dataset configuration
            candidate = find_dataset_config(data_file)
            if candidate is not None:
                data_file = candidate
                data_valid = True

    # If all parameters are complete, use them directly
    if model_type and epochs and data_valid:
        params["model_type"] = model_type
        params["epochs"] = epochs
        params["data"] = data_file
        logger.info(f"Natural language provided complete parameters: model_type={model_type}, epochs={epochs}, data={data_file}")
        return _confirm_and_train(params)

    # ---------- 2. Enter interactive input flow ----------
    # Prompt for model type if missing
    if not model_type:
        model_type = validate_input("Model type (default yolov8n): ", None, "yolov8n")
        params["model_type"] = model_type

    # Prompt for training epochs if missing
    if not epochs:
        epochs_str = validate_input("Training epochs (default 1): ", lambda x: x.isdigit() and int(x) > 0, "1")
        epochs = int(epochs_str)
        params["epochs"] = epochs

    # If dataset configuration is missing or invalid, enter dataset selection logic
    if not data_valid:
        datasets_dir = os.path.join(os.getcwd(), "datasets")
        recent_datasets = []
        if os.path.exists(datasets_dir):
            datasets = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
            for dataset in datasets:
                yaml_path = os.path.join(datasets_dir, dataset, "data.yaml")
                if os.path.exists(yaml_path):
                    recent_datasets.append((dataset, yaml_path, os.path.getmtime(yaml_path)))
            recent_datasets.sort(key=lambda x: x[2], reverse=True)
        if recent_datasets:
            print("\nDetected the following converted datasets (most recent first):")
            for i, (name, path, _) in enumerate(recent_datasets[:3], 1):
                print(f"{i}. {name} -> {path}")
            print(f"{len(recent_datasets[:3]) + 1}. Use default dataset (coco128.yaml)")
            print(f"{len(recent_datasets[:3]) + 2}. Manually enter other dataset configuration path")
            choice = validate_input(f"Please select dataset [1-{len(recent_datasets[:3]) + 2}]: ",
                                    lambda x: x.isdigit() and 1 <= int(x) <= (len(recent_datasets[:3]) + 2),
                                    "1")
            choice = int(choice)
            if choice <= len(recent_datasets[:3]):
                params["data"] = recent_datasets[choice - 1][1]
            elif choice == len(recent_datasets[:3]) + 1:
                params["data"] = "coco128.yaml"
            else:
                params["data"] = validate_input("Please enter dataset configuration file path: ", None, "coco128.yaml")
        else:
            params["data"] = validate_input("Please enter dataset configuration file path: ", None, "coco128.yaml")
    return _confirm_and_train(params)


def _confirm_and_train(params):
    """
    Display training parameters confirmation, call send_train_request after confirmation.
    If user confirms, execute training; otherwise return cancel status.
    """
    print("\nTraining parameters confirmation:")
    print(f" - Model type: {params['model_type']}")
    print(f" - Training epochs: {params['epochs']}")
    print(f" - Data configuration: {params['data']}")

    confirm = validate_input("\nConfirm start training? (y/n): ",
                             lambda x: x.lower() in ['y', 'n', 'yes', 'no'],
                             "y")
    if confirm.lower() in ['y', 'yes']:
        print_info("Starting training...")
        result = send_train_request(params)
        if result["status"] == "success":
            print_success("Training completed!")
        else:
            print_error(f"Training failed: {result['message']}")
        return result
    else:
        print("Training operation canceled")
        return {"status": "canceled", "message": "User canceled operation"}



def _confirm_and_train(params):
    """
    Display training parameters, call send_train_request after confirmation
    """
    print("\nTraining parameters confirmation:")
    print(f" - Model type: {params['model_type']}")
    print(f" - Training epochs: {params['epochs']}")
    print(f" - Data configuration: {params['data']}")
    
    confirm = validate_input("\nConfirm start training? (y/n): ",
                             lambda x: x.lower() in ['y','n','yes','no'],
                             "y")
    if confirm.lower() in ['y','yes']:
        print_info("Starting training...")
        result = send_train_request(params)
        if result["status"] == "success":
            print_success("Training completed!")
        else:
            print_error(f"Training failed: {result['message']}")
        return result
    else:
        print("Training operation canceled")
        return {"status": "canceled", "message": "User canceled operation"}

def process_browser_operation(params):
    """Process browser operations"""
    # Add default value handling
    if 'action' not in params or not params['action']:
        params['action'] = 'open_cvat'
        logger.info("Browser action not specified, defaulting to open_cvat")
    
    action = params.get('action', 'unknown')
    
    print("\nParsing result (browser operation):")
    print(f"- Action: {action}")
    
    if action == 'navigate' and 'url' in params:
        print(f"- URL: {params['url']}")
    
    if action == 'open_cvat':
        browser_type = params.get("browser_type", "msedge")  # Default to using Edge
        params["browser_type"] = browser_type  # Ensure browser_type is included in parameters
        print(f"- Will use {browser_type} browser to open and access CVAT")
    
    confirm = validate_input("\nConfirm execute browser operation? (y/n): ", 
                          lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                          'y')
                          
    if confirm.lower() in ['y', 'yes']:
        result = send_browser_request(params)
        print("\nServer returned result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    else:
        print("Browser operation canceled")
        return {"status": "canceled", "message": "User canceled the operation"}

def test_server_connection():
    """Test server connection"""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"Server connection test successful: {SERVER_URL}")
            return True
        else:
            logger.warning(f"Server returned non-200 status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"Server connection test failed: {str(e)}")
        return False
    

def process_inference_operation(params):
    """Process model inference operation"""
    task_id = params.get('task_id')
    model_name = params.get('model_name')
    threshold = params.get('threshold', 0.5)
    batch_size = params.get('batch_size', 10)  # Added this line to get batch parameter
    
    # If task ID not provided, request user input
    if not task_id:
        task_id = validate_input("Please enter task ID: ", 
                              lambda x: x.isdigit(), 
                              None)
        if not task_id:
            print_error("Task ID must be provided")
            return {"status": "error", "message": "Task ID not provided"}
    
    print("\nParsing result (model inference operation):")
    print(f"- Task ID: {task_id}")
    print(f"- Model name: {model_name or 'Auto select'}")
    print(f"- Confidence threshold: {threshold}")
    print(f"- Batch size: {batch_size}")  # Added this line to display batch information
    
    confirm = validate_input("\nConfirm start auto annotation? (y/n): ", 
                          lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                          'y')
                          
    if confirm.lower() in ['y', 'yes']:
        print_info("Submitting auto annotation request...")
        result = cvat_api.run_auto_annotation(task_id, model_name, threshold, batch_size)
        
        if result["status"] == "success":
            print_success(f"Auto annotation request submitted: {result['message']}")
            if "job_id" in result:
                print_info(f"Task ID: {task_id}, Job ID: {result['job_id']}")
        else:
            print_error(f"Auto annotation request failed: {result['message']}")
        
        return result
    else:
        print("Auto annotation request canceled")
        return {"status": "canceled", "message": "User canceled the operation"}


def process_open_annotation_operation(params):
    """Process open annotation interface operation (only use task_id and job_id, don't request input again if already provided)"""
    print("DEBUG: process_open_annotation_operation received params:", params)
    
    # Check task_id: if already provided and not empty, use directly; otherwise prompt for input
    if 'task_id' in params and str(params['task_id']).strip():
        task_id = str(params['task_id']).strip()
    else:
        task_id = validate_input("Please enter task ID: ", lambda x: x.isdigit(), None)
        if not task_id:
            print_error("Task ID must be provided")
            return {"status": "error", "message": "Task ID not provided"}
    params['task_id'] = task_id

    # Check job_id: if already provided and not empty, use directly; otherwise set default value 1 (no prompt)
    if 'job_id' in params and str(params['job_id']).strip():
        params['job_id'] = int(str(params['job_id']).strip())
    else:
        params['job_id'] = 1

    print("\nParsing result (open annotation interface operation):")
    print(f"- Task ID: {params['task_id']}")
    print(f"- Job ID: {params['job_id']}")

    # Ask user to confirm operation
    confirm = validate_input("\nConfirm open annotation interface? (y/n): ",
                               lambda x: x.lower() in ['y', 'n', 'yes', 'no'], "y")
    if confirm.lower() in ['y', 'yes']:
        print_info("Opening annotation interface...")
        request_data = {
            "action": "open_annotation",
            "task_id": params['task_id'],
            "job_id": params['job_id']
        }
        result = send_open_annotation_request(request_data)
        if result["status"] == "success":
            print_success(f"Successfully opened annotation interface: {result['message']}")
        else:
            print_error(f"Failed to open annotation interface: {result['message']}")
        return result
    else:
        print("Open annotation interface request canceled")
        return {"status": "canceled", "message": "User canceled the operation"}

def process_download_operation(params):
    """Process download dataset operation"""
    task_id = params.get('task_id')
    format = params.get('format', 'CVAT')
    output_path = params.get('output_path')
    include_images = params.get('include_images', True)
    
    # If task ID not provided, request user input (keep or modify as needed)
    if not task_id:
        task_id = validate_input("Please enter task ID: ", lambda x: x.isdigit(), None)
        if not task_id:
            print_error("Task ID must be provided")
            return {"status": "error", "message": "Task ID not provided"}
    
    print("\nParsing result (download dataset operation):")
    print(f"- Task ID: {task_id}")
    print(f"- Download format: {format}")
    print(f"- Output path: {output_path or 'Default path'}")
    print(f"- Include images: {include_images}")
    
    confirm = validate_input("\nConfirm download dataset? (y/n): ", 
                             lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                             'y')
    if confirm.lower() in ['y', 'yes']:
        print_info("Downloading dataset...")
        from cvat_api import download_dataset  # Ensure correct import of cvat_api module
        result = download_dataset(task_id=task_id, output_path=output_path, format=format, include_images=include_images)
        if result["status"] == "success":
            print_success(f"Dataset downloaded successfully: {result['message']}")
            print_info(f"Data saved in: {result['output_path']}")
        else:
            print_error(f"Dataset download failed: {result['message']}")
        return result
    else:
        print("Dataset download request canceled")
        return {"status": "canceled", "message": "User canceled the operation"}


def process_convert_dataset(params):
    from cvat_api import convert_cvat_to_yolo
    """Process dataset format conversion operation"""
    cvat_path = params.get('cvat_path')
    output_dir = params.get('output_dir')
    dataset_name = params.get('dataset_name')
    
    # If CVAT path not provided, request user input
    if not cvat_path:
        cvat_path = validate_input(
            "Please enter CVAT dataset path (directory containing annotations.xml): ", 
            lambda x: os.path.exists(os.path.join(x, "annotations.xml")), 
            None
        )
        if not cvat_path:
            print_error("Valid CVAT dataset path must be provided")
            return {"status": "error", "message": "Valid CVAT dataset path not provided"}
    
    # If dataset name not provided, request user input
    if not dataset_name:
        dataset_name = validate_input(
            "Please enter converted dataset name (default is cvat_converted): ", 
            None, 
            "cvat_converted"
        )
    
    print("\nParsing result (dataset conversion operation):")
    print(f"- CVAT path: {cvat_path}")
    print(f"- Output directory: {output_dir or 'Default path'}")
    print(f"- Dataset name: {dataset_name}")
    
    confirm = validate_input("\nConfirm convert dataset? (y/n): ", 
                          lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                          'y')
    
    if confirm.lower() in ['y', 'yes']:
        print_info("Converting dataset...")
        result = convert_cvat_to_yolo(
            cvat_path=cvat_path, 
            output_dir=output_dir, 
            dataset_name=dataset_name
        )
        
        if result["status"] == "success":
            print_success(f"Dataset conversion successful: {result['message']}")
            print_info(f"Dataset saved in: {result['dataset_dir']}")
            print_info(f"YAML configuration file: {result['yaml_path']}")
            print_info(f"Please use this yaml file path as training parameter")
            # Modified this line to use training+validation set total or newly added image_count
            image_count = result.get('image_count', result.get('train_count', 0) + result.get('val_count', 0))
            print_info(f"Processed {image_count} images, {result.get('label_count', 0)} label categories")
        
        else:
            print_error(f"Dataset conversion failed: {result['message']}")
        
        return result
    else:
        print("Dataset conversion request canceled")
        return {"status": "canceled", "message": "User canceled the operation"}
    
import re

def extract_params_from_query(query):
    """
    Extract task_id and job_id from user input using regular expressions.
    Example: 'I want to view annotation page task is 6 job is 3' extracts {"task_id": "6", "job_id": "3"}.
    """
    params = {}
    task_match = re.search(r'task\s*is\s*(\d+)', query, re.IGNORECASE)
    job_match = re.search(r'job\s*is\s*(\d+)', query, re.IGNORECASE)
    if task_match:
        params["task_id"] = task_match.group(1)
    if job_match:
        params["job_id"] = job_match.group(1)
    return params

def find_dataset_config(data_file):
    """
    Determine if data_file points to a valid training configuration:
    - If data_file is an absolute path or exists as a relative path, return its absolute path
    - Otherwise try to find it in the current directory's datasets/<data_file>/data.yaml
    - If found, return the path to that data.yaml; otherwise return None
    """
    import os
    
    # 1) Return if it exists directly
    if os.path.exists(data_file):
        return os.path.abspath(data_file)
    
    # 2) Try in the current working directory
    candidate = os.path.join(os.getcwd(), data_file)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    
    # 3) Try to find data.yaml in a subdirectory with the same name in the datasets directory
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    dataset_name = os.path.splitext(os.path.basename(data_file))[0]
    candidate_yaml = os.path.join(datasets_dir, dataset_name, "data.yaml")
    if os.path.exists(candidate_yaml):
        return candidate_yaml
    
    return None

def find_available_datasets():
    """Find available converted datasets"""
    datasets = []
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    if os.path.exists(datasets_dir):
        for d in os.listdir(datasets_dir):
            dir_path = os.path.join(datasets_dir, d)
            yaml_path = os.path.join(dir_path, "data.yaml")
            if os.path.isdir(dir_path) and os.path.exists(yaml_path):
                datasets.append({
                    "name": d,
                    "path": yaml_path,
                    "mtime": os.path.getmtime(yaml_path)
                })
    # Sort by modification time, newest first
    return sorted(datasets, key=lambda x: x["mtime"], reverse=True)

def find_latest_download():
    """Find most recently downloaded CVAT dataset"""
    downloads_dir = os.path.join(os.getcwd(), "downloads")
    if not os.path.exists(downloads_dir):
        return None
    
    # Find all download directories
    task_dirs = [os.path.join(downloads_dir, d) for d in os.listdir(downloads_dir) 
                if os.path.isdir(os.path.join(downloads_dir, d))]
    if not task_dirs:
        return None
    
    # Sort by modification time
    latest_dir = max(task_dirs, key=os.path.getmtime)
    
    # Check if there's a task_X_CVAT subdirectory
    for item in os.listdir(latest_dir):
        if item.startswith("task_") and item.endswith("_CVAT"):
            cvat_dir = os.path.join(latest_dir, item)
            if os.path.exists(os.path.join(cvat_dir, "annotations.xml")):
                return cvat_dir
    
    # If no subdirectory but annotations.xml exists, return that directory
    if os.path.exists(os.path.join(latest_dir, "annotations.xml")):
        return latest_dir
    
    return None

def process_download_and_convert(params):
    """Process download and convert dataset operation"""
    import time
    
    # Get parameters, handle default values
    task_id = params.get('task_id')
    format = params.get('format', 'CVAT')
    dataset_name = params.get('dataset_name')
    
    # If task ID not provided, request user input
    if not task_id:
        task_id = validate_input("Please enter task ID: ", 
                              lambda x: x and x.isdigit(), # Don't allow empty value
                              None)
        if not task_id:
            print_error("Task ID must be provided")
            return {"status": "error", "message": "Task ID not provided"}
    
    # If dataset name not provided, use timestamp-based default name to avoid conflicts
    if not dataset_name:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dataset_name = f"task_{task_id}_{timestamp}"
    
    print("\nParsing result (download and convert dataset operation):")
    print(f"- Task ID: {task_id}")
    print(f"- Download format: {format}")
    print(f"- Dataset name: {dataset_name}")
    
    # Log information
    logger.info(f"Preparing one-click download and convert dataset: task_id={task_id}, format={format}, dataset_name={dataset_name}")
    
    confirm = validate_input("\nConfirm download and convert dataset? (y/n): ", 
                         lambda x: x == '' or x.lower() in ['y', 'n', 'yes', 'no'],  # Allow empty value
                         'y')
    
    if confirm.lower() not in ['y', 'yes', '']:
        print("Download and convert dataset request canceled")
        logger.info("User canceled the download and convert dataset operation")
        return {"status": "canceled", "message": "User canceled the operation"}
        
    try:
        # Step 1: Download dataset
        print_info("Step 1/2: Downloading dataset...")
        logger.info(f"Starting dataset download: task_id={task_id}, format={format}")
        
        download_result = cvat_api.download_dataset(
            task_id=task_id,
            format=format,
            include_images=True
        )
        
        if download_result["status"] != "success":
            print_error(f"Dataset download failed: {download_result['message']}")
            logger.error(f"Dataset download failed: {download_result}")
            return download_result
            
        print_success(f"Dataset download successful!")
        logger.info(f"Dataset download successful: {download_result}")
        
        # Confirm output path exists in the returned result
        if "output_path" not in download_result:
            error_msg = "API returned result missing output_path field"
            print_error(error_msg)
            logger.error(f"{error_msg}, actual return: {download_result}")
            return {"status": "error", "message": error_msg}
            
        cvat_path = download_result['output_path']
        
        # Step 2: Convert dataset
        print_info(f"Step 2/2: Converting downloaded dataset to YOLO format...")
        logger.info(f"Starting dataset conversion: cvat_path={cvat_path}, dataset_name={dataset_name}")
        
        convert_result = cvat_api.convert_cvat_to_yolo(
            cvat_path=cvat_path,
            dataset_name=dataset_name
        )
        
        if convert_result["status"] == "success":
            print_success(f"Dataset conversion successful!")
            print_info(f"YOLO dataset saved in: {convert_result['dataset_dir']}")
            print_info(f"YAML configuration file: {convert_result['yaml_path']}")
            logger.info(f"Dataset conversion successful: {convert_result}")
            
            # Return success result
            return {
                "status": "success",
                "message": "Dataset download and convert successful",
                "download_result": download_result,
                "convert_result": convert_result,
                "yaml_path": convert_result['yaml_path']
            }
        else:
            print_error(f"Dataset conversion failed: {convert_result['message']}")
            logger.error(f"Dataset conversion failed: {convert_result}")
            return {
                "status": "partial",
                "message": "Dataset download successful but conversion failed",
                "download_result": download_result,
                "convert_result": convert_result
            }
    except Exception as e:
        error_msg = f"Error occurred during dataset download and conversion: {str(e)}"
        print_error(error_msg)
        logger.exception(error_msg)
        return {"status": "error", "message": error_msg}
    

TOOLS = {
    # Other tool definitions...
    "list_datasets": {
        "description": "List available datasets (converted), returns a list containing name, path, and modification time",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}



def print_colored(text, color_code):
    """Use ANSI color codes to print colored text"""
    print(f"\033[{color_code}m{text}\033[0m")

def print_info(text):
    print_colored(f"[INFO] {text}", "34;1")  # Bold blue

def print_success(text):
    print_colored(f"[SUCCESS] {text}", "32;1")  # Bold green

def print_error(text):
    print_colored(f"[ERROR] {text}", "31;1")  # Bold red

def print_warning(text):
    print_colored(f"[WARNING] {text}", "33;1")  # Bold yellow

    

def main():
   """Main function, handles user interaction and request sending"""
   print("=" * 50)
   print_success("YOLO Training and Browser Control System - MCP Client")
   print("=" * 50)

   # Test server connection
   if not test_server_connection():
       print_error(f"Cannot connect to server {SERVER_URL}")
       print_info("Please confirm the server is running and accessible via network")
       should_continue = validate_input("Continue anyway? (y/n): ",
                                          lambda x: x.lower() in ['y', 'n', 'yes', 'no'],
                                          'n')
       if should_continue.lower() not in ['y', 'yes']:
           return

   # Check API key
   if not API_KEY:
       print_warning("OPENROUTER_API_KEY environment variable not set")
       print_info("Natural language instruction parsing will be unavailable, only manual parameter mode can be used")

   while True:
       print("\nPlease select an operation:")
       print("1. Control system using natural language instructions")
       print("2. Train YOLO model (manual parameters)")
       print("3. Control browser (manual parameters)")
       print("4. Deploy model to CVAT")
       print("5. Upload data to CVAT")
       print("6. Inference data with model")
       print("7. View annotation interface")
       print("8. Download dataset")
       print("9. Convert dataset format")  # Added
       print("10. Download and convert dataset (one-click operation)")  # Added option
       print("11. Exit program")                    # Adjusted number

       choice = validate_input("Enter option [1-11]: ", 
                        lambda x: x in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
       if choice == '1':
           # Natural language instruction mode
           if not API_KEY:
               print_error("API key not set, cannot use natural language instructions")
               continue

           user_query = validate_input(
               "\nPlease enter your instruction (e.g., 'train model' or 'open browser and access CVAT' or 'deploy model to CVAT' or 'upload data to CVAT' or 'inference data with model' or 'view annotation page' or 'download data' or 'convert format' or 'download and convert dataset'): ",
               lambda x: len(x) > 0,
               None
           )
           print_info("Parsing your instruction...")
           parsed = parse_instruction_with_llm(user_query)

           if parsed["operation"] == "train":
               process_train_operation(parsed["params"])
           elif parsed["operation"] == "browser":
               process_browser_operation(parsed["params"])
           elif parsed["operation"] == "deploy":
                deploy_params = parsed.get("params", {})
                # Automatically find latest model weights
                if "model_path" not in deploy_params or not deploy_params["model_path"]:
                    latest_model = get_latest_model_from_runs()
                    if latest_model:
                        deploy_params["model_path"] = latest_model
                        logger.info(f"Automatically selected latest model weight file: {deploy_params['model_path']}")
                    else:
                        # If not found, use default value
                        deploy_params["model_path"] = None  # Keep consistent with original code
                        logger.info("No model found in run output, using default weight file")
                
                if "force" not in deploy_params:
                    deploy_params["force"] = False
                
                process_deploy_operation(deploy_params)


           elif parsed["operation"] == "upload":
               process_upload_operation(parsed["params"])
           elif parsed["operation"] == "inference":
               process_inference_operation(parsed["params"])
           elif parsed["operation"] == "view_annotation":
               params = parsed.get("params", {})
                # If params is empty, try to extract from original input query
               if not params or not params.get("task_id"):
                   params = extract_params_from_query(user_query)
               process_open_annotation_operation(parsed["params"])
            # Added in main function instruction branch
           elif parsed["operation"] == "download_dataset":
               process_download_operation(parsed["params"])
           elif parsed["operation"] == "convert_dataset":
                params = parsed.get("params", {})
                # Check if cvat_path already exists, if not, try to find recently downloaded dataset
                if "cvat_path" not in params or not params["cvat_path"]:
                    downloads_dir = os.path.join(os.getcwd(), "downloads")
                    if os.path.exists(downloads_dir):
                        downloads = sorted(
                            [d for d in os.listdir(downloads_dir) if os.path.isdir(os.path.join(downloads_dir, d))],
                            key=lambda x: os.path.getmtime(os.path.join(downloads_dir, x)),
                            reverse=True
                        )
                        if downloads:
                            recently_downloaded = os.path.join(downloads_dir, downloads[0])
                            print(f"\nDetected recently downloaded dataset: {recently_downloaded}")
                            use_recent = validate_input(
                                "Use this dataset? (y/n): ", 
                                lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                                'y'
                            )
                            if use_recent.lower() in ['y', 'yes']:
                                # Check if there's a task_X_CVAT subdirectory
                                for item in os.listdir(recently_downloaded):
                                    if item.startswith("task_") and item.endswith("_CVAT"):
                                        cvat_subdir = os.path.join(recently_downloaded, item)
                                        if os.path.exists(os.path.join(cvat_subdir, "annotations.xml")):
                                            params["cvat_path"] = cvat_subdir
                                            break
                                # If no subdirectory found but directory itself has annotations.xml
                                if "cvat_path" not in params and os.path.exists(os.path.join(recently_downloaded, "annotations.xml")):
                                    params["cvat_path"] = recently_downloaded
                            else:
                                # User chooses not to use, manual input
                                cvat_path = validate_input(
                                    "Please enter CVAT dataset path: ", 
                                    lambda x: os.path.exists(os.path.join(x, "annotations.xml")), 
                                    None
                                )
                                if cvat_path:
                                    params["cvat_path"] = cvat_path
                # Call processing function
                process_convert_dataset(params)
           elif parsed["operation"] == "list_datasets":
                # When LLM calls list_datasets, directly call the function and return results
                datasets = list_available_datasets()
                return {"operation": "list_datasets", "params": {"datasets": datasets}}
           elif parsed["operation"] == "download_and_convert":
                process_download_and_convert(parsed["params"])

           else:
               print_error("Unable to parse instruction, please try a more explicit statement or use manual mode")

       elif choice == '2':
            # Train model (manual parameters)
            print("\nPlease enter training parameters:")
            model_type = validate_input("Model type (default yolov8n): ", None, "yolov8n")
            epochs_str = validate_input("Training epochs (default 1): ",
                                        lambda x: x.isdigit() and int(x) > 0,
                                        "1")
            # If user leaves empty, pass empty string to allow subsequent auto-selection of "latest dataset".
            # If user enters a path, try to use that path.
            data_input = validate_input("Dataset configuration (leave empty for auto-select latest/default): ", None, "")
            params = {
                "model_type": model_type,
                "epochs": int(epochs_str),
                "data": data_input
            }
            process_train_operation(params)


       elif choice == '3':
           # Control browser (manual parameters)
           print("\nPlease select browser operation:")
           print("1. Open browser and access CVAT")
           print("2. Close browser")
           print("3. Navigate to specified URL")
           print("4. Login to CVAT")
           print("5. Create CVAT project")
           print("6. Force restart browser")

           browser_choice = validate_input("Please enter option [1-6]: ",
                                             lambda x: x in ['1', '2', '3', '4', '5', '6'])
           params = {}
           if browser_choice == '1':
               params = {"action": "open_cvat"}
           elif browser_choice == '2':
               params = {"action": "close"}
           elif browser_choice == '3':
               url = validate_input("Please enter URL: ", lambda x: x.startswith("http"), None)
               params = {"action": "navigate", "url": url}
           elif browser_choice == '4':
               username = validate_input("Username (default admin): ", None, "admin")
               password = validate_input("Password (default admin): ", None, "admin")
               params = {"action": "login_cvat", "username": username, "password": password}
           elif browser_choice == '5':
               name = validate_input("Project name: ", lambda x: len(x) > 0, None)
               params = {"action": "create_project", "name": name}
           elif browser_choice == '6':
               browser_type = validate_input("Browser type (1: Chrome, 2: Edge, 3: Firefox): ",
                                             lambda x: x in ['1', '2', '3'], '1')
               browser_map = {'1': 'chromium', '2': 'msedge', '3': 'firefox'}
               selected_browser = browser_map[browser_type]
               params = {"action": "initialize",
                         "browser_type": selected_browser,
                         "force_new": True,
                         "headless": False}
           process_browser_operation(params)

       elif choice == '4':
            # Deploy model to CVAT — Auto-scan for latest model
            print("\nDeploy model to CVAT, will automatically select latest model file")
            force = validate_input("Force redeploy? (y/n): ",
                                lambda x: x.lower() in ['y', 'n', 'yes', 'no'],
                                'n')
            force = force.lower() in ['y', 'yes']
            
            # Find latest trained model
            latest_model = get_latest_model_from_runs()
            if latest_model:
                print(f"[INFO] Found latest trained model: {latest_model}")
                use_latest = validate_input("Use this model? (y/n): ",
                                        lambda x: x.lower() in ['y', 'n', 'yes', 'no'],
                                        'y')
                if use_latest.lower() in ['y', 'yes']:
                    model_path = latest_model
                else:
                    model_path = None  # If user doesn't use it, still auto-scan default location
            else:
                print("[INFO] No model found in training output directory, will use model from default location")
                model_path = None
            
            params = {"model_path": model_path, "force": force}
            process_deploy_operation(params)

       elif choice == '5':
            # Upload data to CVAT—Always create new task
            print("\nUpload data to CVAT")
            task_name = validate_input(
                "Please enter task name (optional, press Enter to use default name): ",
                None, None
            )
            data_path = validate_input(
                "Please enter data path (optional, press Enter for auto-search): ",
                None, None
            )

            print_info("Creating new task and uploading data...")
            result = cvat_api.upload_data_to_cvat(
                data_path = data_path,
                task_name = task_name
            )

            if result["status"] == "success":
                print_success(f"New task created and upload successful: {result['message']}")
                print_info(f"Task ID: {result['task_id']}")
            else:
                print_error(f"Upload failed: {result['message']}")



       elif choice == '6':
           # Inference data with model
           task_id = validate_input("Please enter task ID: ", lambda x: x.isdigit(), None)
           if not task_id:
               print_error("Task ID must be provided")
               continue
               
           model_name = validate_input("Please enter model name (leave empty for auto-select): ", None, None)
           threshold = validate_input("Please enter confidence threshold (0-1, default 0.5): ", 
                                   lambda x: x == "" or (x.replace('.', '', 1).isdigit() and 0 <= float(x) <= 1),
                                   "0.5")
           batch_size = validate_input("Please enter batch size (default 10): ", 
                                lambda x: x == "" or x.isdigit(), 
                                "1")           
           if threshold:
               threshold = float(threshold)
           if batch_size:
               batch_size = int(batch_size)
               
           params = {"task_id": task_id, "model_name": model_name, "threshold": threshold, "batch_size": batch_size}
           process_inference_operation(params)
       elif choice == '7':

           task_id = validate_input("Please enter task ID: ", lambda x: x.isdigit(), None)
           if not task_id:
               print_error("Task ID must be provided")
               continue


           job_id_input = validate_input("Please enter job ID (default is 1): ", 
                            lambda x: x == "" or x.isdigit(), 
                            "1")
           job_id = int(job_id_input) if job_id_input else 1

           params = {"task_id": task_id, "job_id": job_id}
           process_open_annotation_operation(params)
       
       
       elif choice == '8':
            # Download dataset
            task_id = validate_input("Please enter task ID: ", lambda x: x.isdigit(), None)
            if not task_id:
                print_error("Task ID must be provided")
                continue
            
            # Let user select download format
            print("\nPlease select download format:")
            print("1. CVAT for images 1.1 (default)")
            print("2. COCO 1.0")
            print("3. COCO Keypoints 1.0")
            print("4. Datumaro 1.0")
            print("5. ImageNet 1.0")
            format_choice = validate_input("Please select format [1-5]: ", 
                                        lambda x: x in ['1', '2', '3', '4', '5', ''], 
                                        '1')
            format_map = {
                '1': 'CVAT',
                '2': 'COCO',
                '3': 'COCO_KEYPOINTS',
                '4': 'DATUMARO',
                '5': 'IMAGENET'
            }
            format = format_map.get(format_choice, 'CVAT')
            
            output_path = validate_input("Please enter output path (optional, press Enter to use default path): ", None, None)
            include_images = validate_input("Include images (y/n, default y): ", 
                                            lambda x: x.lower() in ['y', 'n', 'yes', 'no', ''], 
                                            'y')
            include_images = include_images.lower() in ['y', 'yes', '']
            
            params = {
                "task_id": task_id,
                "format": format,
                "output_path": output_path,
                "include_images": include_images
            }
            process_download_operation(params)

            
       elif choice == '9':
            # Convert dataset format
            recently_downloaded = None
            # If dataset was just downloaded, provide quick selection
            downloads_dir = os.path.join(os.getcwd(), "downloads")
            if os.path.exists(downloads_dir):
                downloads = sorted(
                    [d for d in os.listdir(downloads_dir) if os.path.isdir(os.path.join(downloads_dir, d))],
                    key=lambda x: os.path.getmtime(os.path.join(downloads_dir, x)),
                    reverse=True
                )
                if downloads:
                    recently_downloaded = os.path.join(downloads_dir, downloads[0])
                    print(f"\nDetected recently downloaded dataset: {recently_downloaded}")
                    use_recent = validate_input(
                        "Use this dataset? (y/n): ", 
                        lambda x: x.lower() in ['y', 'n', 'yes', 'no'], 
                        'y'
                    )
                    if use_recent.lower() in ['y', 'yes']:
                        cvat_path = recently_downloaded
                    else:
                        cvat_path = validate_input(
                            "Please enter CVAT dataset path: ", 
                            lambda x: os.path.exists(os.path.join(x, "annotations.xml")), 
                            None
                        )
                else:
                    cvat_path = validate_input(
                        "Please enter CVAT dataset path: ", 
                        lambda x: os.path.exists(os.path.join(x, "annotations.xml")), 
                        None
                    )
            else:
                cvat_path = validate_input(
                    "Please enter CVAT dataset path: ", 
                    lambda x: os.path.exists(os.path.join(x, "annotations.xml")), 
                    None
                )
                
            if not cvat_path:
                print_error("Valid CVAT dataset path must be provided")
                continue
                
            dataset_name = validate_input("Please enter converted dataset name (default is cvat_converted): ", None, "cvat_converted")
            output_dir = validate_input("Please enter output directory (optional, press Enter to use default path): ", None, None)
            
            params = {
                "cvat_path": cvat_path,
                "output_dir": output_dir,
                "dataset_name": dataset_name
            }
            process_convert_dataset(params)
            
       elif choice == '10':
            # Download and convert dataset (one-click operation)
            task_id = None  # Let process_download_and_convert function handle parameter acquisition
            format = 'CVAT'
            dataset_name = None  # Let function automatically generate timestamp-based name
            
            params = {
                "task_id": task_id,
                "format": format,
                "dataset_name": dataset_name
            }
            process_download_and_convert(params)
            
       elif choice == '11':  # Adjusted number
            # Exit program
            print_success("Thank you for using the YOLO Training and Browser Control System, goodbye!")
            logger.info("User chose to exit program")
            break

       else:
           print_error("Invalid option, please select again")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        logger.info("Program interrupted by user (KeyboardInterrupt)")
    except Exception as e:
        print_error(f"Program execution error: {str(e)}")
        logger.exception("Program terminated unexpectedly")
        sys.exit(1)
"""
MCP Server - handles training requests and browser control requests
"""

from flask import Flask, request, jsonify
import json
import multiprocessing
import subprocess
import os
import sys
import logging

# Import training and browser control modules
from train import train_yolo
from browser import process_browser_request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_server.log')
    ]
)
logger = logging.getLogger('mcp_server')

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_endpoint():
    """Endpoint for handling training requests"""
    try:
        # Parse JSON request body
        data = request.get_json()
        if not data:
            logger.error("Missing JSON request body")
            return jsonify({"status": "error", "message": "Missing JSON request body"}), 400
        
        logger.info(f"Received training request: {json.dumps(data)}")
        
        # Extract training parameters (with defaults)
        model_type = data.get('model_type', 'yolov8n')
        epochs = data.get('epochs', 1)
        dataset = data.get('data', 'coco128.yaml')
        
        # Call training function
        result = train_yolo(model_type, epochs, dataset)
        
        logger.info(f"Training result: {json.dumps(result)}")
        
        # Return result as JSON
        return jsonify(result), 200
    except Exception as e:
        logger.exception(f"Error handling training request: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/deploy', methods=['POST'])
def deploy_endpoint():
    """Endpoint for handling model deployment to CVAT"""
    try:
        # Parse JSON request body
        data = request.get_json()
        if not data:
            logger.error("Missing JSON request body")
            return jsonify({"status": "error", "message": "Missing JSON request body"}), 400
        
        logger.info(f"Received model deployment request: {json.dumps(data)}")
        
        # Extract deployment parameters
        model_path = data.get('model_path')  # If None, auto-scan latest model
        force = data.get('force', False)
        
        # Build command: use 'wsl python3' on Windows, otherwise 'python3'
        if os.name == "nt":
            cmd = ['wsl', 'python3', 'deploy_to_cvat.py']
        else:
            cmd = ['python3', 'deploy_to_cvat.py']
        if model_path:
            cmd.append(model_path)
        if force:
            cmd.append('--force')
            
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Run deployment script with UTF-8 encoding
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        
        if result.returncode == 0:
            # Try extracting JSON result from output
            try:
                import re
                json_match = re.search(r'\{.*\}', result.stdout, re.DOTALL)
                if json_match:
                    deploy_result = json.loads(json_match.group(0))
                else:
                    deploy_result = {
                        "status": "success",
                        "message": "Model deployed successfully",
                        "details": result.stdout or "No detailed output"
                    }
            except Exception as json_err:
                logger.warning(f"Failed to parse deployment script output: {json_err}")
                deploy_result = {
                    "status": "success",
                    "message": "Model deployed successfully",
                    "details": result.stdout or "No detailed output"
                }
                
            logger.info(f"Deployment succeeded: {json.dumps(deploy_result)}")
            return jsonify(deploy_result), 200
        else:
            logger.error(f"Deployment failed: {result.stderr}")
            return jsonify({
                "status": "error",
                "message": "Model deployment failed",
                "details": result.stderr or "No error details"
            }), 500
            
    except Exception as e:
        logger.exception(f"Error handling deployment request: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/open_annotation', methods=['POST'])
def open_annotation_endpoint():
    """Endpoint for opening the CVAT annotation interface"""
    try:
        # Parse JSON request body
        data = request.get_json()
        if not data:
            logger.error("Missing JSON request body")
            return jsonify({"status": "error", "message": "Missing JSON request body"}), 400
        
        task_id = data.get('task_id')
        if not task_id:
            logger.error("Missing required parameter: task_id")
            return jsonify({"status": "error", "message": "Missing required parameter: task_id"}), 400
        
        job_id = data.get('job_id', 1)
        
        logger.info(f"Received open annotation request: task_id={task_id}, job_id={job_id}")
        
        # Build browser request
        browser_request = {
            "action": "open_annotation",
            "task_id": str(task_id),
            "job_id": str(job_id)
        }
        
        result = process_browser_request(browser_request)
        logger.info(f"Open annotation result: {json.dumps(result, ensure_ascii=False)}")
        return jsonify(result), 200
    
    except Exception as e:
        logger.exception(f"Error opening annotation interface: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/browser', methods=['POST'])
def browser_endpoint():
    """Endpoint for handling browser control requests"""
    try:
        # Parse JSON request body
        data = request.get_json()
        if not data:
            logger.error("Missing JSON request body")
            return jsonify({"status": "error", "message": "Missing JSON request body"}), 400
        
        logger.info(f"Received browser request: {json.dumps(data)}")
        
        # Process browser request
        result = process_browser_request(data)
        
        logger.info(f"Browser control result: {json.dumps(result)}")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.exception(f"Error handling browser request: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "success", "message": "Server is running"}), 200

@app.route('/', methods=['GET'])
def index():
    """Root endpoint providing service info"""
    return jsonify({
        "name": "YOLO MCP Server with Browser Control",
        "endpoints": [
            {"path": "/train", "method": "POST", "description": "Train YOLO model"},
            {"path": "/browser", "method": "POST", "description": "Control browser"},
            {"path": "/deploy", "method": "POST", "description": "Deploy model to CVAT"},
            {"path": "/open_annotation", "method": "POST", "description": "Open CVAT annotation interface"},
            {"path": "/health", "method": "GET", "description": "Health check"}
        ],
        "status": "running"
    }), 200

if __name__ == '__main__':
    # Ensure multiprocessing works correctly on Windows
    multiprocessing.freeze_support()
    
    # Get port (default 5000)
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting MCP server on port {port}")
    print(f"Starting MCP server, listening on port {port}...")
    
    # Run on 0.0.0.0 at specified port
    app.run(host='0.0.0.0', port=port, debug=False)

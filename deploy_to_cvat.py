import os
import sys
import glob
import shutil
import subprocess
import json
import time
import re
import logging

logger = logging.getLogger(__name__)


def get_latest_model(search_dir):
    """
    Find the most recent .pt model file in the given directory and print debug info.
    """
    pt_files = glob.glob(os.path.join(search_dir, "*.pt"))
    print(f"[DEBUG] Scanning directory {search_dir}, found .pt files: {pt_files}")
    if not pt_files:
        return None
    latest_model = max(pt_files, key=os.path.getmtime)
    print(f"[DEBUG] Selected latest model in {search_dir}: {latest_model}")
    return latest_model


def get_latest_model_from_runs():
    """
    Auto-search for the latest model weights under runs/detect.
    Assumes training outputs are in runs/detect/<folder>/weights.
    Returns the path to the newest .pt file.
    """
    runs_dir = os.path.join(os.getcwd(), "runs", "detect")
    if not os.path.exists(runs_dir):
        print("[DEBUG] runs/detect directory not found")
        return None

    subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
               if os.path.isdir(os.path.join(runs_dir, d))]
    if not subdirs:
        print("[DEBUG] No subdirectories under runs/detect")
        return None

    latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(d))
    weights_dir = os.path.join(latest_subdir, "weights")
    if os.path.exists(weights_dir):
        return get_latest_model(weights_dir)
    else:
        # If no weights folder, search directly in the latest directory
        return get_latest_model(latest_subdir)


def deploy_model_to_nuclio(model_path=None, function_name=None,
                            template_dir="deploy_templates",
                            cvat_path="/mnt/d/cvat", gpu=True):
    """
    Deploy a YOLO model to CVAT's Nuclio service.

    Args:
      model_path: Full path to the model file to deploy (if None, auto-scan
                  current dir or "models" folder; if not found, search runs/detect).
      function_name: Name of the deployed Nuclio function; auto-generate if not given.
      template_dir: Directory containing template files; default "deploy_templates".
      cvat_path: CVAT root directory in WSL; default "/mnt/d/cvat".
      gpu: Whether to configure GPU resources.

    Returns:
      A dict containing deployment status and details.
    """
    print(f"[DEBUG] Current working directory: {os.getcwd()}")

    # Auto-find the model file if not provided or missing
    if model_path is None or not os.path.exists(model_path):
        models_dir = os.path.join(os.getcwd(), "models")
        if os.path.exists(models_dir):
            print(f"[DEBUG] Found 'models' folder: {models_dir}")
            model_path = get_latest_model(models_dir)
        else:
            print("[DEBUG] 'models' folder not found, scanning cwd")
            model_path = get_latest_model(os.getcwd())

        # If still not found, search training outputs
        if model_path is None:
            runs_dir = os.path.join(os.getcwd(), "runs", "detect")
            if os.path.exists(runs_dir):
                print(f"[DEBUG] Scanning training outputs: {runs_dir}")
                model_path = get_latest_model(runs_dir)

        if model_path is None:
            return {"status": "error", "message": "No .pt model file found"}

    print(f"[DEBUG] Using model file at: {model_path}")

    # Timestamp for naming
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Extract letter part from filename (e.g., 'yolov8n.pt' -> 'YOLO')
    model_basename = os.path.basename(model_path)
    m = re.match(r'([a-zA-Z]+)(\d*)(.*?)\.pt', model_basename)
    if m:
        prefix = m.group(1).upper()
        version = m.group(2)
        identifier = f"{prefix}V{version}" if version else prefix
    else:
        identifier = "YOLO"

    # Generate dynamic function name if not provided
    dyn_func = f"{identifier.lower()}_custom_{timestamp}"
    if not function_name:
        function_name = dyn_func
    print(f"[DEBUG] Generated function name: {function_name}")

    # Generate model name for templates
    model_name = f"{identifier}_{timestamp}"
    print(f"[DEBUG] Generated model name: {model_name}")

    # Deployment directory: /mnt/d/cvat/serverless/custom/{function_name}
    deploy_dir = os.path.join(cvat_path, "serverless", "custom", function_name)
    deploy_dir = deploy_dir.replace('\\', '/')
    print(f"[DEBUG] Deployment directory: {deploy_dir}")
    os.makedirs(deploy_dir, exist_ok=True)

    # Copy and template replace
    for fname in ["function.yaml", "main.py"]:
        src = os.path.join(template_dir, fname)
        dst = os.path.join(deploy_dir, fname)
        print(f"[DEBUG] Copy template: {src} -> {dst}")
        if not os.path.exists(src):
            return {"status": "error", "message": f"Template not found: {src}"}
        with open(src, 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.replace("{{FUNCTION_NAME}}", function_name)
        text = text.replace("{{MODEL_NAME}}", model_name)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(text)

    # Copy model to deploy_dir/models/best.pt
    models_sub = os.path.join(deploy_dir, "models")
    os.makedirs(models_sub, exist_ok=True)
    print(f"[DEBUG] Copying model to: {models_sub}")
    shutil.copy(model_path, os.path.join(models_sub, "best.pt"))

    # Build Nuclio deploy command
    cmd = (
        f'cd {cvat_path} && '
        f'nuctl deploy --project-name cvat '
        f'--path "./serverless/custom/{function_name}" --platform local'
    )
    if gpu:
        cmd += " --resource-limit nvidia.com/gpu=1"
    print(f"[DEBUG] Deploy command: {cmd}")

    if os.name == 'nt':
        cmd = f"wsl {cmd}"
        print(f"[DEBUG] Windows WSL command: {cmd}")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True,
            text=True, encoding='utf-8', errors='replace'
        )
    except Exception as e:
        print(f"[ERROR] Deployment command exception: {e}")
        return {"status": "error", "message": f"Command exception: {e}"}

    print(f"[DEBUG] Command return: code={result.returncode}, stdout={result.stdout}, stderr={result.stderr}")

    if result.returncode == 0:
        return {
            "status": "success",
            "message": "Model successfully deployed to Nuclio",
            "function_name": function_name,
            "model_name": model_name,
            "details": result.stdout
        }
    else:
        return {
            "status": "error",
            "message": "Deployment failed",
            "details": result.stderr
        }


def convert_windows_path_to_wsl(path_str):
    """Convert a Windows path string to its WSL equivalent."""
    if os.path.exists(path_str):
        return path_str
    m = re.match(r'^([A-Za-z]):\\(.*)', path_str)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace('\\', '/')
        wsl_path = f"/mnt/{drive}/{rest}"
        print(f"[DEBUG] Converted Windows path '{path_str}' to WSL path '{wsl_path}'")
        return wsl_path
    return path_str


if __name__ == '__main__':
    # Parse CLI args
    model_arg = None
    force = False
    for a in sys.argv[1:]:
        if a == '--force':
            force = True
        else:
            model_arg = a
            if model_arg and ':\\' in model_arg:
                model_arg = convert_windows_path_to_wsl(model_arg)
                print(f"[DEBUG] Converted Windows path to WSL: {model_arg}")

    deploy_res = deploy_model_to_nuclio(model_path=model_arg,
                                        function_name=None, gpu=True)
    print(json.dumps(deploy_res, ensure_ascii=False, indent=2))

import os
import requests
import json
import zipfile
import tempfile
from datetime import datetime
import logging
import config
import time
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import shutil
import random
import math
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default to INFO; switch to DEBUG for troubleshooting

# Format mapping: user-friendly names → API format strings
FORMAT_MAP = {
    "CVAT": "CVAT for images 1.1",  # Native CVAT format, includes XML annotations
    "COCO": "COCO 1.0",
    "COCO_KEYPOINTS": "COCO Keypoints 1.0",
    "DATUMARO": "Datumaro 1.0",
    "IMAGENET": "ImageNet 1.0",
    "KITTI": "KITTI 1.0",
    "CAMVID": "CamVid 1.0",
    "CITYSCAPES": "Cityscapes 1.0"
}

# --- Added: COCO 80 classes -----------------------------------------------------------------
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]
# ------------------------------------------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_data_directory():
    """
    Locate the dataset directory.
    First check the default path, then any alternate paths.
    Returns the images subdirectory path or None if not found.
    """
    # Check the default configured location
    if os.path.exists(config.DATA_DIR):
        return config.IMAGES_DIR

    # Check each alternate candidate path
    for path in config.ALTERNATE_PATHS:
        images_dir = os.path.join(path, "images")
        if os.path.exists(images_dir):
            return images_dir

    # If no directory found, return None
    return None

def create_cvat_task(task_name=None, project_id=None, labels=None):
    """
    Create a new task in CVAT.

    Args:
        task_name (str, optional): Desired name for the CVAT task.
        project_id (int, optional): ID of the CVAT project to associate.
        labels (list, optional): List of label definitions to include.

    Returns:
        dict: {status: "success"/"error", task_id (if success), message: description}
    """
    # Use the provided name or fallback to default; if still empty, generate a timestamped name
    task_name = task_name or config.DEFAULT_TASK_NAME
    if not task_name:
        task_name = f"Task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # If no labels and no project specified, build default COCO-80 labels
    if not labels and not project_id:
        labels = [
            {
                "name": cls,
                "color": "#{:06x}".format(random.randint(0, 0xFFFFFF)),
                "attributes": []
            }
            for cls in COCO_CLASSES
        ]

    url = f"{config.CVAT_URL}/api/tasks"
    auth = (config.CVAT_USERNAME, config.CVAT_PASSWORD)

    # Assemble the JSON payload
    payload = {"name": task_name}
    if project_id:
        payload["project_id"] = project_id
    if labels:
        payload["labels"] = labels

    try:
        response = requests.post(
            url,
            auth=auth,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        if response.status_code == 201:
            task_id = response.json()["id"]
            logger.info(f"Created task with ID: {task_id}")
            return {"status": "success", "task_id": task_id, "message": "Task created successfully"}
        else:
            logger.error(f"Failed to create task: {response.text}")
            return {"status": "error", "message": f"Task creation failed: {response.text}"}
    except Exception as e:
        logger.error(f"Error during task creation: {e}")
        return {"status": "error", "message": f"Error creating task: {e}"}

def upload_data_to_task(task_id, data_path):
    """
    Upload images to an existing CVAT task.

    Args:
        task_id (int): The ID of the CVAT task.
        data_path (str): Path to a folder of images or a ZIP archive.

    Returns:
        dict: {status: "success"/"error", message: description}
    """
    url = f"{config.CVAT_URL}/api/tasks/{task_id}/data"
    auth = (config.CVAT_USERNAME, config.CVAT_PASSWORD)

    is_directory = os.path.isdir(data_path)

    try:
        # If a directory, zip its images first
        if is_directory:
            logger.info(f"Zipping image folder for upload: {data_path}")
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                zip_path = tmp.name

            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, _, files in os.walk(data_path):
                    for fname in files:
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                            fp = os.path.join(root, fname)
                            arc = os.path.relpath(fp, data_path)
                            zipf.write(fp, arcname=arc)
            file_to_upload = zip_path
        else:
            # Otherwise assume it’s a ZIP file
            file_to_upload = data_path

        # Perform the upload
        with open(file_to_upload, 'rb') as f:
            files = {'client_files[0]': (os.path.basename(file_to_upload), f)}
            data = {'image_quality': 70}
            response = requests.post(url, auth=auth, files=files, data=data)

        # Clean up the temporary ZIP if we created one
        if is_directory and os.path.exists(zip_path):
            os.unlink(zip_path)

        if response.status_code == 202:
            logger.info("Upload succeeded")
            return {"status": "success", "message": "Data uploaded successfully"}
        else:
            logger.error(f"Upload failed: {response.text}")
            return {"status": "error", "message": f"Data upload failed: {response.text}"}

    except Exception as e:
        logger.error(f"Error during data upload: {e}")
        # Ensure temporary ZIP cleanup
        if is_directory and 'zip_path' in locals() and os.path.exists(zip_path):
            os.unlink(zip_path)
        return {"status": "error", "message": f"Error uploading data: {e}"}
  
def upload_data_to_cvat(data_path=None, task_name=None, project_id=None):
    """
    Upload data to CVAT by creating a task and uploading images.

    Args:
        data_path (str, optional): Path to data (defaults to configured location).
        task_name (str, optional): Name of the CVAT task to create.
        project_id (int, optional): ID of the CVAT project to associate.

    Returns:
        dict: {status: "success"/"error", message: str, task_id: int (on success)}
    """
    # Step 1: Determine data path
    if not data_path:
        data_path = find_data_directory()
        if not data_path:
            return {"status": "error", "message": "Data directory not found; please specify a path"}
    
    # Ensure the path exists
    if not os.path.exists(data_path):
        return {"status": "error", "message": f"Specified path does not exist: {data_path}"}
    
    logger.info(f"Using data path: {data_path}")
    
    # Step 2: Create the CVAT task
    result = create_cvat_task(task_name, project_id)
    if result["status"] != "success":
        return result
        
    task_id = result["task_id"]
    
    # Step 3: Upload data to the task
    upload_result = upload_data_to_task(task_id, data_path)
    if upload_result["status"] != "success":
        return {
            "status": "error",
            "message": f"Task created but data upload failed: {upload_result['message']}",
            "task_id": task_id
        }
    
    return {
        "status": "success",
        "message": "Task created and data uploaded successfully",
        "task_id": task_id
    }


def get_models(project_id=None):
    """
    Retrieve the list of available Nuclio functions (models) in CVAT.

    Args:
        project_id (int, optional): Filter models by project ID.

    Returns:
        dict: {status: "success"/"error", models: list (on success), message: str}
    """
    url = f"{config.CVAT_URL}/api/lambda/functions"
    if project_id:
        url += f"?project={project_id}"
    auth = (config.CVAT_USERNAME, config.CVAT_PASSWORD)
    
    try:
        response = requests.get(url, auth=auth)
        if response.status_code == 200:
            models = response.json()
            logger.info(f"Retrieved {len(models)} models successfully")
            return {"status": "success", "models": models}
        else:
            logger.error(f"Failed to retrieve models: {response.text}")
            return {"status": "error", "message": f"Failed to get models: {response.text}"}
    except Exception as e:
        logger.error(f"Error retrieving model list: {e}")
        return {"status": "error", "message": f"Error getting models: {e}"}


def run_auto_annotation(task_id, model_name=None, threshold=0.5, batch_size=1):
    """
    Run automatic annotation on a specified CVAT task.

    Args:
        task_id (int): ID of the CVAT task.
        model_name (str or int, optional): Name or ID of the model to use.
        threshold (float, optional): Confidence threshold for predictions.
        batch_size (int, optional): Maximum number of images per job.

    Returns:
        dict: {status: "success"/"warning"/"error", message: str, request_id/job_id on success}
    """
    logger.info("=========== Starting auto-annotation debug ===========")
    logger.info(f"Task ID: {task_id}, Model: {model_name}, Threshold: {threshold}, Batch size: {batch_size}")
    
    # Fetch task info
    try:
        task_url = f"{config.CVAT_URL}/api/tasks/{task_id}"
        auth = (config.CVAT_USERNAME, config.CVAT_PASSWORD)
        task_response = requests.get(task_url, auth=auth)
        logger.info(f"Task info status code: {task_response.status_code}")
        if task_response.status_code == 200:
            info = task_response.json()
            logger.info(f"Task details: name={info.get('name')}, size={info.get('size','unknown')}, image_count={info.get('image_count','unknown')}")
    except Exception as e:
        logger.error(f"Error fetching task info: {e}")
    
    # Determine model ID
    if not model_name:
        models_result = get_models()
        if models_result["status"] != "success":
            logger.error(f"Model list retrieval failed: {models_result.get('message')}")
            return models_result
        if not models_result["models"]:
            logger.error("No available models found")
            return {"status": "error", "message": "No available models"}
        
        yolo_models = [m for m in models_result["models"] if "yolo" in m["name"].lower()]
        if yolo_models:
            model_id = yolo_models[0]["id"]
            logger.info(f"Auto-selected model: {yolo_models[0]['name']}")
        else:
            model_id = models_result["models"][0]["id"]
            logger.info(f"No YOLO model found; using first available: {models_result['models'][0]['name']}")
    else:
        if str(model_name).isdigit():
            model_id = int(model_name)
            logger.info(f"Using provided model ID: {model_id}")
        else:
            models_result = get_models()
            if models_result["status"] != "success":
                logger.error(f"Model list retrieval failed: {models_result.get('message')}")
                return models_result
            matches = [m for m in models_result["models"] if model_name.lower() in m["name"].lower()]
            logger.info(f"Found {len(matches)} matching models")
            if not matches:
                logger.error(f"No model contains '{model_name}' in its name")
                return {"status": "error", "message": f"No model matching '{model_name}'"}
            model_id = matches[0]["id"]
            logger.info(f"Selected model: {matches[0]['name']} (ID: {model_id})")
    
    # Prepare annotation request
    url = f"{config.CVAT_URL}/api/lambda/requests"
    auth = (config.CVAT_USERNAME, config.CVAT_PASSWORD)
    payload = {
        "task": task_id,
        "function": model_id,
        "cleanup": True,
        "threshold": float(threshold),
        "max_images_per_job": batch_size
    }
    logger.info(f"Request payload: {payload}")
    
    try:
        logger.info(f"Sending auto-annotation request to {url}")
        response = requests.post(
            url,
            auth=auth,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        logger.info(f"Response status code: {response.status_code}")
        
        if response.status_code in (200, 201):
            result = response.json()
            logger.info(f"Auto-annotation request succeeded: {result}")
            
            if result.get('exc_info'):
                logger.error(f"Server reported exception: {result['exc_info']}")
                return {
                    "status": "warning",
                    "message": "Request submitted but server reported an exception",
                    "request_id": result.get("id"),
                    "job_id": result.get("job"),
                    "exception": result.get("exc_info")
                }
            return {
                "status": "success",
                "message": "Auto-annotation request submitted",
                "request_id": result.get("id"),
                "job_id": result.get("job")
            }
        else:
            logger.error(f"Auto-annotation failed: {response.status_code}, {response.text}")
            return {"status": "error", "message": f"Annotation request failed: {response.text}"}
    
    except Exception as e:
        logger.error(f"Error sending auto-annotation request: {e}", exc_info=True)
        return {"status": "error", "message": f"Error sending annotation request: {e}"}
    
    finally:
        logger.info("=========== Finished auto-annotation debug ===========")

def download_dataset(task_id, output_path=None, format="CVAT", include_images=True):
    """
    Download and extract a task dataset from CVAT.

    Args:
        task_id (int): ID of the CVAT task to download.
        output_path (str, optional): Directory to save downloads.
            Defaults to './downloads/task_{task_id}'.
        format (str, optional): Export format; one of
            "CVAT", "COCO", "COCO_KEYPOINTS", "DATUMARO", "IMAGENET", etc.
        include_images (bool, optional): Whether to include image files. Defaults to True.

    Returns:
        dict: {
            status: "success" or "error",
            message: description,
            format: selected format (on success),
            output_path: extraction directory (on success),
            zip_path: downloaded ZIP file path (on success)
        }
    """
    try:
        # Step 1: Prepare download directory
        if not output_path:
            output_path = os.path.join(os.getcwd(), f"downloads/task_{task_id}")
        os.makedirs(output_path, exist_ok=True)

        # Load CVAT connection info
        from config import CVAT_URL, CVAT_USERNAME, CVAT_PASSWORD

        # Authenticate with CVAT to obtain a token
        auth_url = f"{CVAT_URL}/api/auth/login"
        auth_data = {"username": CVAT_USERNAME, "password": CVAT_PASSWORD}
        logger.info(f"Logging in to CVAT: {auth_url}")
        auth_resp = requests.post(auth_url, json=auth_data)
        if auth_resp.status_code != 200:
            return {"status": "error", "message": f"CVAT login failed: {auth_resp.text}"}

        token = auth_resp.json().get("key")
        headers = {"Authorization": f"Token {token}"}

        # Map user format to API format
        api_format = FORMAT_MAP.get(format, format)

        # Step 2: Initiate export request
        export_url = f"{CVAT_URL}/api/tasks/{task_id}/dataset/export"
        params = {
            "format": api_format,
            "save_images": str(include_images).lower()
        }
        logger.info(f"Initiating export: {export_url}, format={api_format}")
        export_resp = requests.post(export_url, headers=headers, params=params)
        if export_resp.status_code != 202:
            return {"status": "error", "message": f"Export start failed: {export_resp.text}"}

        rq_id = export_resp.json().get("rq_id")
        if not rq_id:
            return {"status": "error", "message": "Failed to obtain export request ID"}

        logger.info(f"Export request submitted; request ID: {rq_id}")

        # Step 3: Poll for completion (up to 60 seconds)
        status_url = f"{CVAT_URL}/api/requests/{rq_id}"
        for attempt in range(60):
            logger.info(f"Checking export status (attempt {attempt+1})")
            status_resp = requests.get(status_url, headers=headers)
            if status_resp.status_code != 200:
                return {"status": "error", "message": f"Status check failed: {status_resp.text}"}
            status_data = status_resp.json()
            if status_data.get("status") == "finished":
                result_url = status_data.get("result_url")
                if not result_url:
                    return {"status": "error", "message": "Export finished but no download URL provided"}
                if not result_url.startswith("http"):
                    result_url = f"{CVAT_URL}{result_url}"
                logger.info(f"Export finished; download URL: {result_url}")
                break
            if status_data.get("status") == "failed":
                return {"status": "error", "message": f"Export failed: {status_data.get('message','unknown error')}"}
            time.sleep(1)
        else:
            return {"status": "error", "message": "Export timed out; please check manually later"}

        # Step 4: Download the ZIP archive
        logger.info("Downloading exported dataset")
        dl_resp = requests.get(result_url, headers=headers, stream=True)
        if dl_resp.status_code != 200:
            return {"status": "error", "message": f"Download failed: {dl_resp.text}"}

        zip_path = os.path.join(output_path, f"task_{task_id}_{format}.zip")
        with open(zip_path, "wb") as f:
            logger.info(f"Saving dataset to {zip_path}")
            for chunk in dl_resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Step 5: Extract the ZIP archive
        extract_dir = os.path.join(output_path, f"task_{task_id}_{format}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            logger.info(f"Extracting dataset to {extract_dir}")
            zip_ref.extractall(extract_dir)

        return {
            "status": "success",
            "message": f"Successfully downloaded and extracted dataset for task {task_id}",
            "format": format,
            "output_path": extract_dir,
            "zip_path": zip_path
        }

    except Exception as e:
        logger.exception(f"Error during dataset download: {e}")
        return {"status": "error", "message": f"Dataset download failed: {e}"}

def convert_cvat_to_yolo(cvat_path, output_dir=None, dataset_name=None, val_split=0.2, sanity_check=True):
    """
    Convert a CVAT-exported XML+images dataset into Ultralytics YOLO format:
      - Supports <box>, <polygon>, <polyline>, <rotatedbox>, and <track> annotations
      - Outputs images/train, images/val, labels/train, labels/val, and data.yaml
    """
    try:
        # 1) Prepare the output directory
        name = dataset_name or "cvat_converted"
        out_dir = output_dir or os.path.join(os.getcwd(), "datasets", name)
        os.makedirs(out_dir, exist_ok=True)

        # 2) Locate annotations.xml
        xml_path = os.path.join(cvat_path, "annotations.xml")
        actual = cvat_path
        if not os.path.exists(xml_path):
            for sub in os.listdir(cvat_path):
                candidate = os.path.join(cvat_path, sub, "annotations.xml")
                if os.path.exists(candidate):
                    xml_path, actual = candidate, os.path.join(cvat_path, sub)
                    break
        if not os.path.exists(xml_path):
            return {"status": "error", "message": f"annotations.xml not found ({xml_path})"}

        # 3) Parse the XML
        root = ET.parse(xml_path).getroot()

        # 4) Build a label → ID mapping
        labels_map = {}
        for lab in root.findall(".//label"):
            name = lab.find("name").text
            if name not in labels_map:
                labels_map[name] = len(labels_map)
        if not labels_map:
            return {"status": "error", "message": "No <label> definitions found"}
        logger.info(f"Label map: {labels_map}")

        # 5) Collect all <image> elements and map ID → filename
        images = root.findall(".//image")
        images.sort(key=lambda im: int(im.get("id")))
        id_to_name = {int(im.get("id")): im.get("name") for im in images}

        # 6) Gather cross-frame <track> boxes grouped by image name
        track_shapes = defaultdict(list)
        for track in root.findall("track"):
            lbl = track.get("label")
            if lbl not in labels_map:
                continue
            for box in track.findall("box"):
                if box.get("outside") == "1":
                    continue
                frame = int(box.get("frame"))
                fname = id_to_name.get(frame)
                if not fname:
                    continue
                xtl, ytl, xbr, ybr = (float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
                track_shapes[fname].append((lbl, xtl, ytl, xbr, ybr))

        # 7) Locate image directories (support nested structures)
        imgs_root = os.path.join(actual, "images")
        if not os.path.exists(imgs_root):
            imgs_root = actual
        image_subdirs = [
            rootdir for rootdir, _, files in os.walk(imgs_root)
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files)
        ]

        # 8) Split images into train/val sets
        random.shuffle(images)
        if len(images) > 1:
            val_count = max(1, int(len(images) * val_split))
            val_count = min(val_count, len(images) - 1)
            train_imgs, val_imgs = images[val_count:], images[:val_count]
        else:
            train_imgs = val_imgs = images
        logger.info(f"Dataset split: train={len(train_imgs)}, val={len(val_imgs)}, total={len(images)}")

        # 9) Create output subdirectories
        dirs = {
            "train_img": os.path.join(out_dir, "images", "train"),
            "val_img":   os.path.join(out_dir, "images", "val"),
            "train_lbl": os.path.join(out_dir, "labels", "train"),
            "val_lbl":   os.path.join(out_dir, "labels", "val"),
        }
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)

        # 10) Define image processing function
        def process_image(elem, tgt_img_dir, tgt_lbl_dir):
            fname = elem.get("name")
            width, height = float(elem.get("width")), float(elem.get("height"))

            # a) Copy the image file
            src = None
            for d in image_subdirs:
                candidate_paths = [os.path.join(d, fname), os.path.join(d, os.path.basename(fname))]
                for cp in candidate_paths:
                    if os.path.exists(cp):
                        src = cp
                        break
                if src:
                    break
            if not src:
                logger.warning(f"Image not found: {fname}")
                return False
            shutil.copy2(src, os.path.join(tgt_img_dir, os.path.basename(fname)))

            # b) Gather all bounding shapes
            shapes = []
            # — Rectangular boxes
            for child in elem:
                attrs = child.attrib
                if all(k in attrs for k in ("xtl", "ytl", "xbr", "ybr")) and attrs.get("label") in labels_map:
                    coords = tuple(map(float, (attrs["xtl"], attrs["ytl"], attrs["xbr"], attrs["ybr"])))
                    shapes.append((attrs["label"], *coords))
            # — Polygons and polylines
            pts_re = re.compile(r"[; ]+")
            for poly in elem.findall("polygon") + elem.findall("polyline"):
                lbl = poly.get("label")
                if lbl not in labels_map:
                    continue
                pts = pts_re.split(poly.get("points").strip())
                xs = [float(p.split(",")[0]) for p in pts]
                ys = [float(p.split(",")[1]) for p in pts]
                shapes.append((lbl, min(xs), min(ys), max(xs), max(ys)))
            # — Rotated boxes
            for rb in elem.findall("rotatedbox"):
                lbl = rb.get("label")
                if lbl not in labels_map:
                    continue
                cx, cy, bw, bh, _ = (float(rb.get(k)) for k in ("cx", "cy", "w", "h", "angle"))
                shapes.append((lbl, cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2))
            # — Include track-based shapes
            shapes.extend(track_shapes.get(fname, []))

            # c) Convert to YOLO format lines
            lines = []
            for lbl, xtl, ytl, xbr, ybr in shapes:
                cls_id = labels_map[lbl]
                x_center = (xtl + xbr) / (2 * width)
                y_center = (ytl + ybr) / (2 * height)
                w_norm = (xbr - xtl) / width
                h_norm = (ybr - ytl) / height
                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # d) Write the .txt label file
            os.makedirs(tgt_lbl_dir, exist_ok=True)
            label_file = os.path.join(tgt_lbl_dir, Path(fname).stem + ".txt")
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.debug(f"Wrote {len(lines)} entries for {fname}")
            return True

        # 11) Process all images
        train_count = sum(process_image(img, dirs["train_img"], dirs["train_lbl"]) for img in train_imgs)
        val_count   = sum(process_image(img, dirs["val_img"],   dirs["val_lbl"])   for img in val_imgs)
        logger.info(f"Conversion complete → train={train_count}, val={val_count}")

        # 12) Write data.yaml
        yaml_path = os.path.join(out_dir, "data.yaml")
        names = [label for label, _ in sorted(labels_map.items(), key=lambda x: x[1])]
        with open(yaml_path, "w", encoding="utf-8") as f:
            base_path = out_dir.replace("\\", "/")
            f.write(
                f"# YOLO dataset\n"
                f"path: {base_path}\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"nc: {len(names)}\n"
                f"names: {names}\n"
            )

        # 13) Return success
        return {
            "status": "success",
            "message": f"CVAT data converted successfully (train={train_count}, val={val_count})",
            "dataset_dir": out_dir,
            "yaml_path": yaml_path,
            "train_count": train_count,
            "val_count": val_count,
            "labels_map": labels_map
        }

    except Exception as e:
        logger.exception("Conversion failed")
        return {"status": "error", "message": str(e)}

import os

# Base path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
DATA_DIR = os.path.join(BASE_DIR, "coco128")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

# Fallback directory paths — possible user locations
ALTERNATE_PATHS = [
    "C:/Users/JACKSON/Desktop/YOLO/agent_training_mcp_browser/coco128",
]

# CVAT settings
CVAT_URL = "http://localhost:8080"
CVAT_USERNAME = "admin"
CVAT_PASSWORD = "Yyh277132984"

# Project settings
DEFAULT_TASK_NAME = "COCO128_Dataset"

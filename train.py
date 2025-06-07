# train.py
from ultralytics import YOLO
import os

# Set UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

def train_yolo(model_type: str = 'yolov8n', epochs: int = 1, data: str = 'coco128.yaml') -> dict:
    """
    Simple function to train a YOLOv8 model.

    :param model_type: YOLO model type, e.g., 'yolov8n.pt', 'yolov8m.pt'.
    :param epochs: Number of training epochs.
    :param data: Dataset configuration file.
    :return: A dictionary containing training status and information.
    """
    try:
        # Load a pre-trained model
        model_path = f"{model_type}.pt" if not model_type.endswith(".pt") else model_type
        model = YOLO(model_path)

        # Train the model
        results = model.train(
            data=data,          # Use built-in config or the path to a custom config
            epochs=epochs,
            imgsz=640,          # Adjustable image size
            name=f'{model_type}_custom'
        )

        return {
            "status": "success",
            "message": f"Started training {model_type} for {epochs} epoch(s) using {data} dataset.",
            "details": str(results)  # Additional info (e.g., log path, weight save path)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred during training: {str(e)}"
        }


if __name__ == '__main__':
    # Run this script directly from the command line if needed
    result = train_yolo('yolov8n', 1, 'coco128.yaml')
    print(result)

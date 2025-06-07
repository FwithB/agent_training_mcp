from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

# Set base path
base_path = "C:/Users/JACKSON/Desktop/YOLO/agent_training_mcp_browser"

# Load model
model_path = os.path.join(base_path, "runs/detect/yolov8n_custom2/weights/best.pt")
model = YOLO(model_path)

# Test image path - using the first image from the COCO128 dataset
test_image_path = os.path.join(base_path, "coco128/images/train2017/000000000009.jpg")

# Run inference
results = model(test_image_path)

# Print results
for r in results:
    print(f"Detected {len(r.boxes)} objects")
    print(f"Classes: {r.boxes.cls.tolist()}")
    print(f"Class names: {[model.names[int(c)] for c in r.boxes.cls.tolist()]}")
    print(f"Confidence: {r.boxes.conf.tolist()}")

# Visualize results
fig, ax = plt.subplots(figsize=(12, 9))
result_plot = results[0].plot()
plt.imshow(result_plot)
plt.axis('off')
plt.tight_layout()

# Save result image
output_path = os.path.join(base_path, "test_result.jpg")
plt.savefig(output_path)
print(f"Results saved to: {output_path}")

# Show image
plt.show()
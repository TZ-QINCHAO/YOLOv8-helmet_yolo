from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("/root/autodl-fs/ultralytics/runs/detect/train5/weights/best.pt")

# Define path to the image file
source = "/root/autodl-fs/ultralytics/000010.jpg"

# Run inference on the source
results = model(source,device ='cpu')  # list of Results objects
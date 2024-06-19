
from ultralytics import YOLO

model = YOLO('weights/yolov8n-myy.yaml')  # 使用YOLOv8的配置文件创建模型实例
model.eval()  # 设置为评估模式

# 打印模型结构
print(model)

# 如果需要遍历每一层并打印详细信息，可以使用如下代码
for name, layer in model.named_modules():
    print(f'{name}: {layer}')
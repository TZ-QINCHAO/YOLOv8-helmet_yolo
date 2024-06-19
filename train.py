import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-fs/ultralytics-test02/weights/yolov8n-QC.yaml')  #加载模型yaml文件
    # model.load('/root/autodl-fs/ultralytics-main/yolov8n.pt') # 加载预训练权重
    model.train(data='/root/autodl-fs/ultralytics-test02/weights/helmet_yolo.yaml',  #加载数据集  数据集具体路径都在helmet_yolo文件里
                imgsz=640, #图片大小
                epochs=100,  #轮数
                batch=8,   #batch size
                close_mosaic=0,  #是否开启 图片增强
                workers=8,
                device='0',  #显卡运行
                optimizer='SGD', #优化器:SGD
                # patience=0, # 关闭早停
                project='/root/autodl-fs/ultralytics-test02/runs/train', #运行结果保存位置
                name='exp',
                )
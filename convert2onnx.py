import os
from ultralytics import YOLO
import onnx
import onnxruntime as ort
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.data import YOLODataset
from ultralytics.utils import ops
import torch

# -----------------------
# 设置路径
# -----------------------
pt_model_path = "best_pruned.pt"
onnx_model_path = "best_pruned.onnx"
int8_model_path = "best_int8.onnx"
data_yaml_path = "train.yaml"  # yolov8 格式的数据集配置文件

# -----------------------
# Step 1: 导出为 ONNX
# -----------------------
print("[1] Exporting model to ONNX...")
model = YOLO(pt_model_path)
model.export(format='onnx', dynamic=True, simplify=True)
assert os.path.exists(onnx_model_path), "ONNX export failed!"


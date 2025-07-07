import os
import yaml
import onnx
import numpy as np
from ultralytics import YOLO
from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader, QuantType, QuantFormat
import onnxruntime as ort
from tqdm import tqdm
import cv2


class YoloCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataset_yaml_path, input_name, input_shape=(1, 3, 64, 64), num_samples=2000):
        with open(dataset_yaml_path, 'r') as f:
            dataset = yaml.safe_load(f)
        self.image_paths = dataset['train'] if isinstance(dataset['train'], list) else [dataset['train']]
        self.image_paths = self._collect_images(self.image_paths)
        self.image_paths = self.image_paths[:num_samples]
        self.input_name = input_name
        self.input_shape = input_shape
        self.data_iter = iter(self._preprocess_images())

    def _collect_images(self, path_list):
        image_files = []
        for path in path_list:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_files.append(os.path.join(root, file))
        return image_files

    def _preprocess_images(self):
        for path in tqdm(self.image_paths, desc="Calibrating"):
            img = cv2.imread(path)
            img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB & HWC to CHW
            img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
            yield {self.input_name: img[np.newaxis, :]}

    def get_next(self):
        return next(self.data_iter, None)


def evaluate_model(onnx_path, dataset_yaml_path, imgsz=64, split='val'):
    model = YOLO(onnx_path)
    results = model.val(data=dataset_yaml_path, imgsz=imgsz, split=split, plots=False)
    return results.box.map  # 返回 mAP@0.5


def quantize_model(fp32_model_path, int8_model_path, dataset_yaml_path):
    # 加载模型，获取输入名
    model = onnx.load(fp32_model_path)
    input_name = model.graph.input[0].name

    # 构建量化器
    dr = YoloCalibrationDataReader(dataset_yaml_path, input_name)
    quantize_static(
        model_input=fp32_model_path,
        model_output=int8_model_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QOperator,  # QDQ 格式也可以，写成 QuantType.QDQ
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )
    print(f"[✓] 已保存 INT8 模型至 {int8_model_path}")



if __name__ == '__main__':
    fp32_model = 'best_pruned.onnx'        # 你的原始模型路径
    int8_model = 'best_pruned-int8.onnx'   # 保存量化后的模型
    dataset_yaml = 'train.yaml'       # 你的数据集配置

    print("🔍 评估浮点模型...")
    fp32_map = evaluate_model(fp32_model, dataset_yaml)
    print(f"🎯 FP32 mAP@0.5-0.95: {fp32_map:.4f}")

    print("📦 开始量化...")
    quantize_model(fp32_model, int8_model, dataset_yaml)

    print("🔍 评估量化后模型...")
    int8_map = evaluate_model(int8_model, dataset_yaml)
    print(f"🎯 INT8 mAP@0.5-0.95: {int8_map:.4f}")

    print("📊 精度损失: {:.4f}".format(fp32_map - int8_map))

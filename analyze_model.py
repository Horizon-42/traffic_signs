import onnx
import os
import numpy as np
from onnxruntime import InferenceSession
from onnx_tool import model_profile # 优先使用 onnx_tool 获取 FLOPs

def analyze_quantized_cropped_onnx_model(model_path, input_shape=None):
    """
    分析量化裁剪后的 ONNX 模型的磁盘占用、参数量、FLOPs 和权重 RAM 占用。

    Args:
        model_path (str): ONNX 模型的路径。
        input_shape (list): 模型的单个输入张量形状，例如：[1, 3, 640, 640]。
                             对于 YOLOv8 模型，通常是 [batch_size, 3, height, width]。
                             这个参数对于计算 FLOPs 至关重要。
    """
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到: {model_path}")
        return

    print(f"--- 分析模型: {os.path.basename(model_path)} ---")

    # 1. 磁盘占用
    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"**磁盘占用**: {file_size_mb:.2f} MB")

    # 加载模型
    try:
        model = onnx.load(model_path)
        graph = model.graph
    except Exception as e:
        print(f"错误: 无法加载 ONNX 模型。请确保这是一个有效的 ONNX 文件。错误信息: {e}")
        return

    # 2. 模型参数量 和 权重 RAM 占用
    total_parameters = 0
    total_weight_bytes = 0

    # ONNX 的 initializer 就是模型的权重/参数
    for initializer in graph.initializer:
        num_elements = 1
        for dim in initializer.dims:
            num_elements *= dim
        total_parameters += num_elements

        # 根据数据类型计算字节数
        dtype_size = 0
        if initializer.data_type == onnx.TensorProto.FLOAT:
            dtype_size = 4  # float32
        elif initializer.data_type == onnx.TensorProto.FLOAT16:
            dtype_size = 2  # float16
        elif initializer.data_type == onnx.TensorProto.INT8: # 量化模型常见
            dtype_size = 1  # int8
        elif initializer.data_type == onnx.TensorProto.UINT8: # 量化模型常见
            dtype_size = 1  # uint8
        elif initializer.data_type == onnx.TensorProto.INT32:
            dtype_size = 4  # int32
        elif initializer.data_type == onnx.TensorProto.INT64:
            dtype_size = 8  # int64
        elif initializer.data_type == onnx.TensorProto.DOUBLE:
            dtype_size = 8  # float64
        else:
            print(f"警告: 未知数据类型 {initializer.data_type} for initializer {initializer.name}. 无法准确计算大小。")
            continue
        
        total_weight_bytes += num_elements * dtype_size

    total_weight_mb = total_weight_bytes / (1024 * 1024)
    print(f"**模型总参数量**: {total_parameters:,}")
    print(f"**权重 RAM 占用估算**: {total_weight_mb:.2f} MB")
    print("  (注: 实际运行时 RAM 占用会更高，包含中间激活和 ONNX Runtime 自身开销)")

    print("\n--- 分析完成 ---")

if __name__ == '__main__':
    # --- 示例用法 ---
    # 请将 'your_yolov8_model_quantized_cropped.onnx' 替换为你的模型文件路径
    # 并且 `input_shape` 必须与你模型训练和导出的输入形状匹配
    # 对于 YOLOv8 模型，常见的输入形状是 [1, 3, 640, 640] 或 [1, 3, 416, 416] 等。
    # 批次大小 (第一个维度) 设为 1 用于 FLOPs 计算，除非你需要计算更大批次的 FLOPs。

    # 替换为你的实际模型路径
    your_model_path = "best_pruned-int8.onnx" 
    
    # 替换为你的模型实际输入形状
    # 对于 YOLOv8，通常是 [batch_size, channels, height, width]
    your_model_input_shape = [1, 3, 64, 64] 

    # --- 运行分析 ---
    analyze_quantized_cropped_onnx_model(your_model_path, your_model_input_shape)
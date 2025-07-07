import torch
from ultralytics import YOLO
from thop import profile, clever_format
import os

def analyze_yolov11_pt_flops(model_path, img_size=640, batch_size=1):
    """
    分析 YOLOv11 .pt 模型的 FLOPs 和参数量。

    Args:
        model_path (str): YOLOv11 .pt 模型文件的路径。例如：'yolov11n.pt'
        img_size (int): 模型期望的输入图像尺寸（正方形），例如 640。
                        YOLO 模型通常将图像缩放到这个尺寸进行推理。
        batch_size (int): 用于计算 FLOPs 的批次大小。通常设置为 1。
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the .pt file is in the correct directory or provide the full path.")
        return

    print(f"--- 分析模型: {os.path.basename(model_path)} ---")

    try:
        # 1. 加载 YOLOv11 模型
        # Ultralytics 的 YOLO 类可以直接加载 .pt 文件
        model = YOLO(model_path)
        print(f"模型 '{model_path}' 加载成功。")

        # 2. 准备一个 dummy 输入张量
        # YOLOv11 模型通常接收 [batch_size, 3, img_size, img_size] 的张量
        # 注意：模型内部可能包含一些处理逻辑（如归一化），但 thop 关注的是模型前向传播的计算图
        dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(model.device)

        # 3. 计算 FLOPs 和参数
        # profile 函数会遍历模型并计算每个操作的 FLOPs
        # 注意：Ultralytics YOLO 模型是 nn.Module 的包装，
        # 我们通常需要访问其内部的实际 PyTorch 模型。
        # 对于 Ultralytics YOLO，可以通过 model.model 访问底层 nn.Module
        macs, params = profile(model.model, inputs=(dummy_input, ), verbose=False)

        # clever_format 用于将大数字格式化为 B, M, G 等单位
        macs, params = clever_format([macs, params], "%.2f")

        print(f"**输入尺寸**: {img_size}x{img_size} (Batch Size: {batch_size})")
        print(f"**模型参数量**: {params}")
        print(f"**FLOPs (GFLOPs)**: {macs} (这里通常是 MACs，乘以2可以近似为 FLOPs)")
        print("\nNote: For convolutional layers, 1 MAC (Multiply-Accumulate) is often considered 2 FLOPs (1 multiplication + 1 addition).")
        print("      So, the actual FLOPs might be approximately twice the reported MACs.")

    except ImportError:
        print("Error: Required libraries not found.")
        print("Please install them: `pip install torch ultralytics thop`")
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Ensure the model path is correct and the model is compatible with Ultralytics YOLO library.")

    print("\n--- 分析完成 ---")

if __name__ == '__main__':
    # --- 示例用法 ---
    # 替换为你的 YOLOv11n.pt 文件路径
    # 如果文件在当前目录，直接写文件名即可
    # 如果你的模型是其他尺寸（例如 416x416），请调整 img_size
    
    # 官方的 YOLOv11n 模型通常使用 640x640 的输入尺寸
    model_file = 'best_pruned.pt' 
    input_image_size = 64

    # 运行分析
    analyze_yolov11_pt_flops(model_file, input_image_size)

    # 如果你没有 yolov11n.pt 文件，Ultralytics 库会自动下载。
    # 你可以先运行一次 `YOLO('yolov11n.pt')` 来确保模型已被下载。
    # 例如：
    # from ultralytics import YOLO
    # _ = YOLO('yolov11n.pt') # 首次运行会自动下载模型
    # 然后再执行上面的 analyze_yolov11_pt_flops 函数
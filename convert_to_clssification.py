import pandas as pd
import os
import shutil
import tqdm
import random
from collections import defaultdict

def convert_to_classification(
    csv_file_path: str,
    output_root_dir: str,
    is_train: bool = True,
    split_ratio: float = 0.2,
    seed: int = 42
):
    """
    Converts detection CSV to classification format. If is_train=True, it splits into train/val.

    Args:
        csv_file_path (str): Path to CSV file with 'Path' and 'ClassId' columns.
        output_root_dir (str): Output root directory like './classification_data'.
        is_train (bool): Whether the data is for training (split into train/val) or testing.
        split_ratio (float): Ratio for validation set if training.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)

    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ Loaded CSV with {len(df)} entries: {csv_file_path}")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return

    src_dir = os.path.dirname(csv_file_path)
    class_to_images = defaultdict(list)

    for _, row in df.iterrows():
        class_id = str(row["ClassId"])
        image_path = os.path.join(src_dir, row["Path"])
        class_to_images[class_id].append(image_path)

    total_copied = 0

    if is_train:
        train_dir = os.path.join(output_root_dir, "train")
        val_dir = os.path.join(output_root_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for class_id, image_paths in class_to_images.items():
            random.shuffle(image_paths)
            val_count = int(len(image_paths) * split_ratio)
            val_images = image_paths[:val_count]
            train_images = image_paths[val_count:]

            train_class_dir = os.path.join(train_dir, class_id)
            val_class_dir = os.path.join(val_dir, class_id)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            for path in train_images:
                if os.path.isfile(path):
                    shutil.copy(path, os.path.join(train_class_dir, os.path.basename(path)))
                    total_copied += 1

            for path in val_images:
                if os.path.isfile(path):
                    shutil.copy(path, os.path.join(val_class_dir, os.path.basename(path)))
                    total_copied += 1

        print(f"✅ Done: {total_copied} images copied into 'train' and 'val' folders.")
    
    else:  # For test set
        test_dir = os.path.join(output_root_dir, "test")
        os.makedirs(test_dir, exist_ok=True)

        for class_id, image_paths in class_to_images.items():
            test_class_dir = os.path.join(test_dir, class_id)
            os.makedirs(test_class_dir, exist_ok=True)

            for path in image_paths:
                if os.path.isfile(path):
                    shutil.copy(path, os.path.join(test_class_dir, os.path.basename(path)))
                    total_copied += 1

        print(f"✅ Done: {total_copied} images copied into 'test' folder.")

if __name__ == "__main__":
    # === 修改这里 ===
    # 输入文件路径
    input_train_csv = './data/dataset/Traffic/Data/Train.csv'
    input_test_csv = './data/dataset/Traffic/Data/Test.csv'

    # 输出根目录
    output_dir = './data/dataset/classification_data'

    # === 执行转换 ===
    convert_to_classification(
        csv_file_path=input_train_csv,
        output_root_dir=output_dir,
        is_train=True,         # 训练集：拆分 train/val
        split_ratio=0.2
    )

    convert_to_classification(
        csv_file_path=input_test_csv,
        output_root_dir=output_dir,
        is_train=False          # 测试集：直接放到 test
    )

    print("\n--- ✅ 数据准备完毕，接下来你可以用 ImageFolder 加载数据 ---")
    print("train_dataset = datasets.ImageFolder('./dataset/classification_data/train', transform=...)")
    print("val_dataset   = datasets.ImageFolder('./dataset/classification_data/val', transform=...)")
    print("test_dataset  = datasets.ImageFolder('./dataset/classification_data/test', transform=...)")

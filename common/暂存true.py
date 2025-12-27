import os
import numpy as np
# 安全忽略信息提示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from loaddata import load_dataset_with_labels

total_dataset_path = r"E:\huilanshujuji2025\dataset1"
train_x, train_y, val_x, val_y, loader = load_dataset_with_labels(total_dataset_path, img_size=(256, 256))

# 然后可以直接使用这些数据进行训练
print(f"训练数据: {train_x.shape}")
print(f"训练标签: {train_y.shape}")
print(f"类别: {loader.class_names}")
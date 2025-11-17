import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import to_categorical

class ImageDatasetLoader:
    def __init__(self, dataset_path, img_size=(256, 256)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.class_names = []
        self.label_to_name = {}
        self.name_to_label = {}
        self.failed_images = []

    def load_existing_datasets(self, train_dir="train", val_dir="val"):
        """加载训练集和验证集"""
        # 验证路径
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"路径不存在: {self.dataset_path}")

        # 自动检测数据集结构
        train_path = os.path.join(self.dataset_path, train_dir)
        val_path = os.path.join(self.dataset_path, val_dir)

        if not os.path.exists(train_path):
            train_path = self.dataset_path
            val_path = None
            print(f"使用目录作为训练集: {self.dataset_path}")
        else:
            print(f"找到训练集: {train_path}")
            if os.path.exists(val_path):
                print(f"找到验证集: {val_path}")

        # 发现类别和创建标签映射
        self.class_names = sorted([d for d in os.listdir(train_path)
                                   if os.path.isdir(os.path.join(train_path, d))])
        self.label_to_name = {i: name for i, name in enumerate(self.class_names)}
        self.name_to_label = {name: i for i, name in enumerate(self.class_names)}

        print(f"发现 {len(self.class_names)} 个类别: {self.class_names}")

        # 加载数据
        datasets = {}

        # 加载训练集
        train_images, train_labels = self._load_split_images(train_path)
        datasets['train'] = {'images': train_images, 'labels': train_labels}
        print(f"训练集: {len(train_images)} 张图像")




        # 加载验证集（如果存在）
        if val_path and os.path.exists(val_path):
            val_images, val_labels = self._load_split_images(val_path)
            datasets['val'] = {'images': val_images, 'labels': val_labels}
            print(f"验证集: {len(val_images)} 张图像")

        return datasets

    def _load_split_images(self, split_path):
        """加载单个数据集的图像和标签"""
        images = []
        labels = []
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')

        for class_name in self.class_names:
            class_dir = os.path.join(split_path, class_name)
            if not os.path.exists(class_dir):
                continue

            class_label = self.name_to_label[class_name]
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(supported_formats)]

            print(f"  加载 {class_name}: {len(image_files)} 张")

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        # 调整尺寸
                        if self.img_size:
                            img = cv2.resize(img, self.img_size)
                        # 转换颜色空间和归一化
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype(np.float32) / 255.0

                        images.append(img)
                        labels.append(class_label)
                    else:
                        self.failed_images.append({'path': img_path, 'error': '读取失败', 'class': class_name})
                except Exception as e:
                    self.failed_images.append({'path': img_path, 'error': str(e), 'class': class_name})

        return np.array(images), np.array(labels)

    def check_image_formats(self):
        """检查图片格式分布"""
        print("\n检查图片格式分布...")
        format_count = {}

        for split_name in ['train', 'val']:
            split_path = os.path.join(self.dataset_path, split_name)
            if os.path.exists(split_path):
                for class_name in self.class_names:
                    class_dir = os.path.join(split_path, class_name)
                    if os.path.exists(class_dir):
                        for img_file in os.listdir(class_dir):
                            ext = Path(img_file).suffix.lower()
                            format_count[ext] = format_count.get(ext, 0) + 1

        print("图片格式分布:")
        for ext, count in format_count.items():
            print(f"  {ext}: {count} 张")

    def display_dataset_info(self, datasets):
        """显示数据集信息"""
        print("\n" + "=" * 50)
        print("数据集信息汇总")
        print("=" * 50)

        for split_name, data in datasets.items():
            images = data['images']
            labels = data['labels']

            if len(images) > 0:
                unique, counts = np.unique(labels, return_counts=True)
                distribution = {self.label_to_name[cls]: count for cls, count in zip(unique, counts)}

                print(f"\n{split_name.upper()}集:")
                print(f"  图像数量: {len(images)}")
                print(f"  图像形状: {images[0].shape}")
                print(f"  类别分布: {distribution}")

        if self.failed_images:
            print(f"\n加载失败的图片: {len(self.failed_images)} ")


def load_dataset_with_labels(dataset_path, img_size=(256, 256), normalize=True, one_hot_label=True,channels_first=True):
    """
    主函数：加载数据集并返回数据及标签，支持数据归一化和独热标签

    参数:
        dataset_path: 数据集路径
        img_size: 图像尺寸
        normalize: 是否进行数据归一化 (0-1范围)
        one_hot_label: 是否转换为独热编码标签

    返回:
        train_images, train_labels, val_images, val_labels, loader
    """
    # 初始化加载器
    loader = ImageDatasetLoader(dataset_path, img_size=img_size)

    try:
        # 加载数据集
        print("正在加载数据集...")
        datasets = loader.load_existing_datasets(train_dir="train", val_dir="val")

        # 检查图片格式
        loader.check_image_formats()

        # 显示数据集信息
        loader.display_dataset_info(datasets)

        # 返回数据集
        train_data = datasets.get('train', {'images': np.array([]), 'labels': np.array([])})
        val_data = datasets.get('val', {'images': np.array([]), 'labels': np.array([])})

        train_images = train_data['images']
        train_labels = train_data['labels']
        val_images = val_data['images']
        val_labels = val_data['labels']

        if channels_first and len(train_images) > 0:
            print("正在进行数据格式转换...")
            train_images = np.transpose(train_images, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
            val_images = np.transpose(val_images, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)

        # 数据归一化
        if normalize and len(train_images) > 0:
            train_images = train_images.astype('float64') / 255.0
            val_images = val_images.astype('float64') / 255.0
            print("数据归一化完成 (0-1范围)")

        # 独热标签转换
        if one_hot_label and len(train_labels) > 0:
            # 自动计算类别数量
            all_labels = np.concatenate([train_labels, val_labels])
            num_classes = len(np.unique(all_labels))

            train_labels = to_categorical(train_labels, num_classes=num_classes)
            val_labels = to_categorical(val_labels, num_classes=num_classes)
            print(f"独热标签转换完成，共 {num_classes} 个类别")

        return train_images, train_labels, val_images, val_labels, loader

    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([]), None

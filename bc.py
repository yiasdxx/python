# loaddata.py

import os
import cv2
import numpy as np


def load_dataset(dataset_path, img_size):
    """
    数据加载器 - 纯NumPy实现
    返回: (训练图片, 训练标签), (验证图片, 验证标签), 类别列表
    """

    def load_folder(folder_path):
        """加载单个文件夹的所有图片"""
        images = []
        labels = []
        classes = []

        # os.path:获取文件的属性
        # 检查路径是否存在
        if not os.path.exists(folder_path):
            raise ValueError(f"路径不存在: {folder_path}")

        # listdir:列出目录内容，只有文件名没有路径
        # 获取所有类别文件夹
        # 遍历文件夹所有项（包括文件与子文件),item代表当前文件夹名称
        for item in os.listdir(folder_path):
            # 将文件夹路径和项目名称组合成完整的文件路径
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                classes.append(item)

        classes.sort()  # 保证顺序一致，，，，，按照字母升序进行排序

        # 加载每个类别的图片
        for class_name in classes:
            class_path = os.path.join(folder_path, class_name)
            image_count = 0

            # 现在来到每一个子文件中的图片读取
            for img_name in os.listdir(class_path):
                # img_name.lower(): 将文件名转换为小写（避免大小写问题）
                # .endswith(...): 检查文件扩展名是否是图像格式
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_path, img_name)

                    try:
                        # 使用cv2加载图片
                        # cv2.IMREAD_COLOR,读取为3通道BGR图像
                        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        if img is None:
                            print(f"警告: 无法加载图片 {img_path}")
                            continue

                        # 调整尺寸
                        img = cv2.resize(img, img_size)

                        # 转换颜色空间 BGR -> RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # 转换为通道优先格式 (H, W, C) -> (C, H, W)
                        img = np.transpose(img, (2, 0, 1))

                        # 归一化
                        img = img.astype(np.float32) / 255.0

                        images.append(img)
                        labels.append(class_name)
                        image_count += 1

                    except Exception as e:
                        print(f"警告: 无法加载图片 {img_path}: {e}")
                        continue

            print(f"  加载类别 '{class_name}': {image_count} 张图片")

        # 转换数据格式
        images = np.array(images)

        # 标签转数字，
        class_to_idx = {cls: i for i, cls in enumerate(classes)}#标签编码
        label_indices = np.array([class_to_idx[label] for label in labels])#转换索引

        return images, label_indices, classes

    print(f"正在加载数据集: {dataset_path}")

    # 加载训练集
    train_images, train_labels, classes = load_folder(os.path.join(dataset_path, 'train'))

    # 加载验证集
    val_images, val_labels, _ = load_folder(os.path.join(dataset_path, 'val'))

    # 输出统计信息
    print(f"\n📊 数据集统计:")
    print(f"  训练集: {len(train_images)} 张图片")
    print(f"  验证集: {len(val_images)} 张图片")
    print(f"  类别数: {len(classes)}")
    if len(train_images) > 0:
        print(f"  图片维度: {train_images.shape[1:]}")
    print(f"  类别列表: {classes}")

    return train_images, train_labels, val_images, val_labels, classes
import os
import numpy as np
import cv2
from conv import *
import pickle



def load_folder(folder_path,img_size):
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

def model_accuracy(model, test_images, test_labels,batch_size=32):
    #总的准确率
    total_accuracy = model.accuracy(test_images, test_labels, batch_size=batch_size)
    print(f"\n总体准确率: {total_accuracy:.4f} ({total_accuracy * 100:.2f}%)")


def main():
    test_dataset_path ="competition\dataset\val"  # 这是测试集路径
    model_path = "simple_convnet_model.pkl"#这是模型路径
    img_size = (64, 64)

    # 1.加载模型
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    class_names = data.get('class_names', [f'类别{i}' for i in range(data['num_classes'])])

    model = SimpleConvNet(
        input_dim=data['input_dim'],
        conv1_param={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
        conv2_param={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
        hidden_size=100,
        output_size=data['num_classes'],
        use_batchnorm=True
    )

    # 加载参数
    model.params = data['params']

    # 更新所有层的参数
    model.layers['Conv1'].W = model.params['W1']
    model.layers['Conv1'].b = model.params['b1']
    model.layers['Conv2'].W = model.params['W2']
    model.layers['Conv2'].b = model.params['b2']
    model.layers['Affine1'].W = model.params['W3']
    model.layers['Affine1'].b = model.params['b3']
    model.layers['Affine2'].W = model.params['W4']
    model.layers['Affine2'].b = model.params['b4']

    # 更新gamma和beta参数
    model.layers['BatchNorm1'].gamma = model.params['gamma1']
    model.layers['BatchNorm1'].beta = model.params['beta1']
    model.layers['BatchNorm2'].gamma = model.params['gamma2']
    model.layers['BatchNorm2'].beta = model.params['beta2']
    model.layers['BatchNorm3'].gamma = model.params['gamma3']
    model.layers['BatchNorm3'].beta = model.params['beta3']

    # 加载running_mean和running_var
    bn_params = data['bn_params']
    model.layers['BatchNorm1'].running_mean = bn_params['BatchNorm1_running_mean']
    model.layers['BatchNorm1'].running_var = bn_params['BatchNorm1_running_var']
    model.layers['BatchNorm2'].running_mean = bn_params['BatchNorm2_running_mean']
    model.layers['BatchNorm2'].running_var = bn_params['BatchNorm2_running_var']
    model.layers['BatchNorm3'].running_mean = bn_params['BatchNorm3_running_mean']
    model.layers['BatchNorm3'].running_var = bn_params['BatchNorm3_running_var']

    print(f"✅ 模型已加载，{len(class_names)}个类别")

    #2.加载测试数据集
    test_images, test_labels, dataset_classes =load_folder(test_dataset_path,img_size)

    #3.计算精度
    model_accuracy(model, test_images, test_labels, batch_size=32)

if __name__ == "__main__":
    main()


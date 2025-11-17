# main.py
import numpy as np
import matplotlib.pyplot as plt
from conv import SimpleConvNet,Trainer



def main():
    # 设置随机种子
    np.random.seed(42)

    print("=== 大尺寸图像分类 - 卷积神经网络 ===")
    print("使用ReLU激活函数和He初始化")

    # 1. 加载数据
    print("\n1. 加载数据集...")
    from loaddata import load_dataset_with_labels

    dataset_path = r"E:\huilanshujuji2025\dataset1"

    train_x, train_y, val_x, val_y, loader = load_dataset_with_labels(dataset_path, img_size=(64, 64))

    print(f"训练集数据: {train_x.shape}")
    print(f"训练集标签: {train_y.shape} (one-hot编码)")
    print(f"验证集数据: {val_x.shape}")
    print(f"验证集标签: {val_y.shape} (one-hot编码)")

    num_samples, channels, height, width = train_x.shape
    input_dim = (channels, height, width)

    # 将one-hot标签转换为整数标签
    train_y_int = train_y.argmax(axis=1)
    val_y_int = val_y.argmax(axis=1)

    # 动态获取类别数
    if train_y.ndim == 2:  # one-hot 编码
        num_classes = train_y.shape[1]
        print(f"检测到 one-hot 编码，类别数: {num_classes}")
    else:  # 整数标签
        num_classes = len(np.unique(train_y))
        print(f"检测到整数标签，类别数: {num_classes}")


    print(f"类别数: {num_classes}")

    print(f"转换后训练标签: {train_y_int.shape}")
    print(f"转换后验证标签: {val_y_int.shape}")
    print(f"训练标签值范围: {train_y_int.min()} ~ {train_y_int.max()}")
    print(f"验证标签值范围: {val_y_int.min()} ~ {val_y_int.max()}")
    print(f"训练标签唯一值: {np.unique(train_y_int)}")
    print(f"验证标签唯一值: {np.unique(val_y_int)}")

    # # 强制设置类别数为8（根据你的one-hot编码维度）
    # num_classes = 8  # 直接设置为8，因为你的one-hot编码是8维
    # print(f"设置类别数: {num_classes}")

    # 2. 创建网络 - 使用动态输入维度和类别数
    print("\n2. 初始化网络...")
    network = SimpleConvNet(
        input_dim=input_dim,  # 动态输入维度
        conv1_param={'filter_num': 16, 'filter_size': 3, 'pad': 0, 'stride': 1},
        conv2_param={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
        hidden_size=100,
        output_size=num_classes  # 动态类别数
    )

    # 立即测试网络输出
    print("\n3. 测试网络输出形状...")
    test_output = network.predict(train_x[:2])
    print(f"网络输出形状: {test_output.shape}")
    print(f"网络输出类别数: {test_output.shape[1]}")

    if test_output.shape[1] != num_classes:
        print(f"❌ 错误: 网络输出类别数 {test_output.shape[1]} 不等于期望的 {num_classes}")
        return

    # 3. 训练网络
    print("\n4. 开始训练...")
    trainer = Trainer(
        network=network,
        x_train=train_x,
        t_train=train_y_int,
        x_val=val_x,
        t_val=val_y_int,
        epochs=100,
        batch_size=100,
        optimizer='adam',  # 使用Adam优化器
        learning_rate=0.001  # Adam通常使用更小的学习率
    )

    train_loss, train_acc, val_acc = trainer.train()

    # 4. 绘制训练结果
    print("\n4. 绘制训练结果...")
    trainer.plot_training_history()

    # 5. 显示预测示例
    print("\n5. 显示预测示例...")
    show_prediction_examples(network, val_x, val_y, "验证集预测示例")

    # 6. 最终报告
    print("\n6. 训练报告:")
    print(f"最佳训练准确率: {max(train_acc):.4f}")
    print(f"最佳验证准确率: {max(val_acc):.4f}")
    print(f"最终训练准确率: {train_acc[-1]:.4f}")
    print(f"最终验证准确率: {val_acc[-1]:.4f}")
    print(f"过拟合程度: {train_acc[-1] - val_acc[-1]:.4f}")

    # 7. 保存模型（可选）
    save_model(network, 'simple_convnet_model.pkl')


def show_prediction_examples(network, x_data, t_data, title, num_examples=12):
    """显示预测示例"""
    indices = np.random.choice(len(x_data), min(num_examples, len(x_data)), replace=False)

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i + 1)

        # 显示图像 (3,256,256) -> (256,256,3) 用于显示
        img = x_data[idx].transpose(1, 2, 0)

        # 如果图像是归一化的，反归一化显示
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        plt.imshow(img)

        # 获取预测
        true_label = t_data[idx]
        pred = network.predict(x_data[idx:idx + 1])
        pred_label = np.argmax(pred)
        confidence = np.max(pred)

        # 设置标题颜色
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}',
                  color=color, fontsize=10)
        plt.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def save_model(network, input_dim, num_classes, filename):
    """保存模型"""
    import pickle
    model_data = {
        'params': network.params,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'network_config': {
            'conv1_filter_num': 16,
            'conv2_filter_num': 32,
            'hidden_size': 100
        }
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"模型已保存到: {filename}")


if __name__ == "__main__":
    main()
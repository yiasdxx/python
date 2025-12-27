import numpy as np
from zh01 import load_dataset
import pickle
from conv import SimpleConvNet, Trainer


def main():
    # 设置随机种子
    np.random.seed(42)

    print("=== 大尺寸图像分类 - 卷积神经网络 ===")
    print("使用ReLU激活函数和He初始化")

    # 1. 加载数据
    print("\n1. 加载数据集...")
    dataset_path = r"E:\huilanshujuji2025\dataset2"

    train_x, train_y, val_x, val_y, class_names = load_dataset(
        dataset_path,
        img_size=(64, 64),
    )

    print(f"训练集数据: {train_x.shape} (N, C, H, W)")
    print(f"训练集标签: {train_y.shape}")
    if len(val_x) > 0:
        print(f"验证集数据: {val_x.shape} (N, C, H, W)")
        print(f"验证集标签: {val_y.shape}")

    # 获取输入维度和类别数
    if train_x.ndim == 4:
        num_samples, channels, height, width = train_x.shape
        input_dim = (channels, height, width)
    else:
        print(f"❌ 错误: 训练数据维度异常: {train_x.shape}")
        return

    # load_dataset 返回的 train_y 和 val_y 已经是整数标签
    num_classes = len(class_names)  # 从类别名称获取类别数
    train_y_int = train_y  # 已经是整数标签
    if len(val_y) > 0:
        val_y_int = val_y  # 已经是整数标签
    else:
        val_y_int = np.array([])

    # 2. 创建网络
    print("\n2. 初始化网络...")
    network = SimpleConvNet(
        input_dim=input_dim,
        conv1_param={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
        conv2_param={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
        hidden_size=100,
        use_batchnorm=True,
        output_size=num_classes,
        dropout_ratio=0.5,
    )

    # 测试网络输出
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
        t_train=train_y_int,  # 使用整数标签
        x_val=val_x,
        t_val=val_y_int,  # 使用整数标签
        epochs=100,
        batch_size=16,
        optimizer='adam',
        learning_rate=0.0005,
        use_augmentation=True,
        use_cutmix=True,
        cutmix_alpha=1.0,
        cutmix_prob=0.5,
        patience=100,
        aug_start_threshold=0.9
    )

    train_loss, train_acc, val_acc = trainer.train()

    # 4. 绘制训练结果
    print("\n5. 绘制训练结果...")
    trainer.plot_training_history()

    # 5. 最终报告
    print("\n6. 训练报告:")
    print(f"最佳训练准确率: {max(train_acc):.4f}")
    if len(val_acc) > 0:
        print(f"最佳验证准确率: {max(val_acc):.4f}")
        print(f"最终验证准确率: {val_acc[-1]:.4f}")
        print(f"过拟合程度: {train_acc[-1] - val_acc[-1]:.4f}")
    else:
        print("验证准确率: 无验证集")
    print(f"最终训练准确率: {train_acc[-1]:.4f}")

    # 6. 保存模型
    save_model(network, input_dim, num_classes, 'simple_convnet_model2.pkl',class_names)

def save_model(network, input_dim, num_classes, filename, class_names):
    """保存模型（简化版）"""
    # 收集参数
    model_params = network.params

    # 收集BN参数
    bn_params = {}
    if hasattr(network, 'layers'):
        for name, layer in network.layers.items():
            if hasattr(layer, 'running_mean'):
                bn_params[f'{name}_running_mean'] = layer.running_mean
                bn_params[f'{name}_running_var'] = layer.running_var

    # 保存数据
    model_data = {
        'params': model_params,
        'bn_params': bn_params,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'class_names': class_names
    }

    # 保存文件
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✅ 模型已保存到: {filename}")
    return filename


if __name__ == "__main__":
    main()
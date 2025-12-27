import numpy as np
import sys
import os
import time
import argparse
from typing import Dict, List, Tuple
from loaddata import load_dataset_with_labels

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从RNconv导入所有必要的类和函数
try:
    from RNconv import (
        ResNet, resnet18, resnet34,
        cross_entropy_loss, accuracy, learning_rate_scheduler,
        ResNetTrainer
    )

    print("✅ 成功导入RNconv模块")
except ImportError as e:
    print(f"❌ 导入RNconv模块失败: {e}")
    print("请确保RNconv.py文件在同一目录下")
    sys.exit(1)


# ==================== 数据加载器 ====================

class SimpleDataLoader:
    """简单的数据加载器 - 用于测试和演示"""

    def __init__(self, images, labels, batch_size=32, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(images)
        self.indices = np.arange(self.num_samples)
        self.current_idx = 0

        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration

        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[self.current_idx:end_idx]

        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]

        self.current_idx = end_idx
        return batch_images, batch_labels

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


# ==================== 命令行参数解析 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ResNet训练脚本')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34'],
                        help='选择ResNet模型架构 (默认: resnet18)')
    parser.add_argument('--num-classes', type=int, default=8,
                        help='分类类别数 (默认: 8)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练epoch数 (默认: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小 (默认: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='学习率 (默认: 0.1)')
    parser.add_argument('--img-size', type=int, nargs=2, default=[64, 64],
                        help='输入图像尺寸 (默认: 64 64)')
    parser.add_argument('--data-dir', type=str, default=r"E:\huilanshujuji2025\dataset2",
                        help='数据集目录路径')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='模型保存目录 (默认: checkpoints)')

    return parser.parse_args()


# ==================== 模型测试函数 ====================

def test_model_forward_pass(model, img_size=(64, 64)):
    """测试模型的前向传播"""
    print("\n测试模型前向传播...")

    # 创建测试数据
    test_batch = np.random.randn(2, 3, img_size[0], img_size[1]).astype(np.float32)

    # 测试训练模式
    output_train = model.forward(test_batch, training=True)
    print(f"训练模式输出形状: {output_train.shape}")

    # 测试推理模式
    output_inference = model.predict(test_batch)
    print(f"推理模式输出形状: {output_inference.shape}")

    # 检查输出是否合理
    print(f"输出范围: [{output_inference.min():.4f}, {output_inference.max():.4f}]")

    return True


# ==================== 主函数 ====================

def main():
    """主函数"""
    args = parse_args()

    print("=" * 70)
    print("ResNet 训练程序")
    print("=" * 70)
    print(f"模型架构: {args.model}")
    print(f"类别数: {args.num_classes}")
    print(f"图像尺寸: {args.img_size}")
    print(f"训练参数: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.learning_rate}")
    print(f"数据目录: {args.data_dir}")
    print("=" * 70)

    # 创建模型
    print("\n1. 创建模型...")
    model_creators = {
        'resnet18': resnet18,
        'resnet34': resnet34
    }

    model = model_creators[args.model](num_classes=args.num_classes)
    print(f"✅ 创建 {args.model} 模型成功")

    # 测试模型
    test_model_forward_pass(model, tuple(args.img_size))

    # 加载数据
    print("\n2. 加载数据集...")
    dataset_path = args.data_dir

    train_x, train_y, val_x, val_y, loader = load_dataset_with_labels(
        dataset_path,
        img_size=tuple(args.img_size),
        train_dir="train",
        val_dir="valid",
        normalize=True,
        one_hot_label=False,  # 改为False，因为你的模型需要整数标签
    )

    # 检查数据是否加载成功
    if len(train_x) == 0:
        print("❌ 错误: 数据集加载失败")
        print("请检查:")
        print(f"   1. 路径是否正确: {dataset_path}")
        print(f"   2. 数据集结构是否正确 (应有 train/ 和 valid/ 目录)")
        print(f"   3. 图像文件格式是否支持")
        return

    print(f"✅ 数据加载成功:")
    print(f"   训练集: {train_x.shape}")
    print(f"   训练标签: {train_y.shape}")
    print(f"   验证集: {val_x.shape}")
    print(f"   验证标签: {val_y.shape}")

    # 创建数据加载器
    print("\n3. 创建数据加载器...")
    train_loader = SimpleDataLoader(train_x, train_y, args.batch_size, shuffle=True)
    val_loader = SimpleDataLoader(val_x, val_y, args.batch_size, shuffle=False) if len(val_x) > 0 else None

    print(f"✅ 数据加载器创建完成")
    print(f"   训练批次: {len(train_loader)}")
    if val_loader:
        print(f"   验证批次: {len(val_loader)}")

    # 测试第一个batch
    try:
        sample_batch, sample_labels = next(iter(train_loader))
        print(f"✅ 数据形状检查通过: {sample_batch.shape}")
        print(f"✅ 标签形状检查通过: {sample_labels.shape}")
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        return

    # 创建训练器
    print("\n4. 创建训练器...")
    trainer = ResNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate
    )
    print("✅ 训练器创建完成")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_path = os.path.join(args.save_dir, f'{args.model}_best.pkl')

    # 开始训练
    print("\n5. 开始训练...")
    print("=" * 70)

    start_time = time.time()
    trainer.train(epochs=args.epochs, save_path=model_save_path)
    training_time = time.time() - start_time

    # 训练完成
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)

    # 获取训练摘要
    summary = trainer.get_training_summary()

    print(f"训练总时间: {training_time:.2f}秒 ({training_time / 60:.2f}分钟)")
    print(f"最佳验证准确率: {summary['best_accuracy']:.4f}")
    print(f"总训练轮数: {summary['current_epoch']}")
    print(f"最终训练准确率: {summary['train_accuracies'][-1]:.4f}")

    if summary['val_accuracies']:
        print(f"最终验证准确率: {summary['val_accuracies'][-1]:.4f}")

    # 绘制训练历史
    try:
        print("\n绘制训练历史图表...")
        trainer.plot_training_history()
    except Exception as e:
        print(f"绘制图表失败: {e} (可能缺少matplotlib)")

    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, f'{args.model}_final.pkl')
    model.save(final_model_path)
    print(f"✅ 最终模型已保存到: {final_model_path}")

    print("\n" + "=" * 70)
    print("程序执行完毕!")
    print("=" * 70)


if __name__ == '__main__':
    main()
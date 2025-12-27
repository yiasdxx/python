import numpy as np
import pickle
from typing import List, Tuple


# ==================== 基础层实现（带反向传播） ====================

class Conv2D:
    """2D卷积层 - 包含完整的前向和反向传播"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, use_bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias

        # He初始化
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * std
        self.weights_grad = np.zeros_like(self.weights)

        if use_bias:
            self.bias = np.zeros((out_channels, 1, 1, 1))
            self.bias_grad = np.zeros_like(self.bias)
        else:
            self.bias = None
            self.bias_grad = None

        self.input = None
        self.output_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.input = x
        batch_size, in_channels, in_height, in_width = x.shape

        # 计算输出尺寸
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.output_shape = (batch_size, self.out_channels, out_height, out_width)

        # 填充输入
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                                  (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x

        # 初始化输出
        output = np.zeros(self.output_shape)

        # 执行卷积
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        region = x_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, oc, oh, ow] = np.sum(region * self.weights[oc])

        # 添加偏置
        if self.use_bias and self.bias is not None:
            output += self.bias

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播"""
        batch_size, in_channels, in_height, in_width = self.input.shape
        _, out_channels, out_height, out_width = dout.shape

        # 初始化输入梯度
        dx = np.zeros_like(self.input)

        # 填充输入用于梯度计算
        if self.padding > 0:
            x_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding),
                                           (self.padding, self.padding)), mode='constant')
            dx_padded = np.zeros_like(x_padded)
        else:
            x_padded = self.input
            dx_padded = dx

        # 计算权重梯度和输入梯度
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        region = x_padded[b, :, h_start:h_end, w_start:w_end]
                        self.weights_grad[oc] += dout[b, oc, oh, ow] * region

                        if self.padding > 0:
                            dx_padded[b, :, h_start:h_end, w_start:w_end] += dout[b, oc, oh, ow] * self.weights[oc]
                        else:
                            dx[b, :, h_start:h_end, w_start:w_end] += dout[b, oc, oh, ow] * self.weights[oc]

        # 计算偏置梯度
        if self.use_bias and self.bias is not None:
            self.bias_grad += np.sum(dout, axis=(0, 2, 3)).reshape(self.bias.shape)

        # 如果使用了填充，需要去除填充
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dx

    def update(self, learning_rate: float):
        """更新参数"""
        self.weights -= learning_rate * self.weights_grad
        if self.use_bias and self.bias is not None:
            self.bias -= learning_rate * self.bias_grad

        # 重置梯度
        self.weights_grad.fill(0)
        if self.bias_grad is not None:
            self.bias_grad.fill(0)


class BatchNorm2D:
    """批归一化层 - 包含完整的前向和反向传播"""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)

        # 运行统计量
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

        # 缓存
        self.input = None
        self.x_hat = None
        self.mean = None
        self.var = None
        self.std = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """前向传播"""
        self.input = x

        if training:
            # 计算当前批次的统计量
            self.mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            self.var = np.var(x, axis=(0, 2, 3), keepdims=True)
            self.std = np.sqrt(self.var + self.eps)

            # 更新运行统计量
            self.running_mean = (self.momentum * self.running_mean +
                                 (1 - self.momentum) * self.mean)
            self.running_var = (self.momentum * self.running_var +
                                (1 - self.momentum) * self.var)

            # 归一化
            self.x_hat = (x - self.mean) / self.std
        else:
            # 推理时使用运行统计量
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # 缩放和平移
        output = self.gamma * self.x_hat + self.beta
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播"""
        batch_size = self.input.shape[0]

        # 计算gamma和beta的梯度
        self.gamma_grad += np.sum(dout * self.x_hat, axis=(0, 2, 3), keepdims=True)
        self.beta_grad += np.sum(dout, axis=(0, 2, 3), keepdims=True)

        # 计算输入梯度
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (self.input - self.mean) * -0.5 * (self.var + self.eps) ** (-1.5),
                      axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dx_hat * -1 / self.std, axis=(0, 2, 3), keepdims=True) + \
                dvar * np.sum(-2 * (self.input - self.mean), axis=(0, 2, 3), keepdims=True) / batch_size

        dx = dx_hat / self.std + dvar * 2 * (self.input - self.mean) / batch_size + dmean / batch_size

        return dx

    def update(self, learning_rate: float):
        """更新参数"""
        self.gamma -= learning_rate * self.gamma_grad
        self.beta -= learning_rate * self.beta_grad

        # 重置梯度
        self.gamma_grad.fill(0)
        self.beta_grad.fill(0)


class ReLU:
    """ReLU激活函数"""

    def __init__(self):
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (self.input > 0)


class GlobalAvgPool2D:
    """全局平均池化"""

    def __init__(self):
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        # 对高度和宽度维度进行平均，保持通道维度
        return np.mean(x, axis=(2, 3), keepdims=False)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_size, channels = dout.shape
        _, _, height, width = self.input_shape

        # 将梯度广播到原始空间维度
        dx = np.zeros(self.input_shape)
        for b in range(batch_size):
            for c in range(channels):
                dx[b, c, :, :] = dout[b, c] / (height * width)

        return dx


class Linear:
    """全连接层"""

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        # He初始化
        std = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(out_features, in_features) * std
        self.weights_grad = np.zeros_like(self.weights)

        if use_bias:
            self.bias = np.zeros(out_features)
            self.bias_grad = np.zeros_like(self.bias)
        else:
            self.bias = None
            self.bias_grad = None

        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        output = x @ self.weights.T
        if self.bias is not None:
            output += self.bias
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # 计算权重梯度
        self.weights_grad += dout.T @ self.input

        # 计算偏置梯度
        if self.bias is not None:
            self.bias_grad += np.sum(dout, axis=0)

        # 计算输入梯度
        dx = dout @ self.weights
        return dx

    def update(self, learning_rate: float):
        """更新参数"""
        self.weights -= learning_rate * self.weights_grad
        if self.bias is not None:
            self.bias -= learning_rate * self.bias_grad

        # 重置梯度
        self.weights_grad.fill(0)
        if self.bias_grad is not None:
            self.bias_grad.fill(0)


class MaxPool2D:
    """最大池化层 - 修复版本"""

    def __init__(self, kernel_size: int, stride: int, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input = None
        self.max_indices = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.input = x
        batch_size, channels, in_height, in_width = x.shape

        # 计算输出尺寸
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 填充输入
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                                  (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x

        # 初始化输出和最大索引
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)

        # 执行最大池化
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        region = x_padded[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, oh, ow] = np.max(region)

                        # 保存最大值的索引（相对于填充后的图像）
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        self.max_indices[b, c, oh, ow] = [h_start + max_idx[0], w_start + max_idx[1]]

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播 - 修复版本"""
        batch_size, channels, in_height, in_width = self.input.shape
        dx = np.zeros_like(self.input)

        # 如果使用了填充，需要创建填充版本的梯度
        if self.padding > 0:
            dx_padded = np.pad(dx, ((0, 0), (0, 0), (self.padding, self.padding),
                                    (self.padding, self.padding)), mode='constant')
        else:
            dx_padded = dx

        # 将梯度传递到前向传播中的最大值位置
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(dout.shape[2]):
                    for ow in range(dout.shape[3]):
                        h_idx, w_idx = self.max_indices[b, c, oh, ow]

                        # 确保索引在有效范围内
                        if (0 <= h_idx < dx_padded.shape[2] and
                                0 <= w_idx < dx_padded.shape[3]):
                            dx_padded[b, c, h_idx, w_idx] += dout[b, c, oh, ow]

        # 如果使用了填充，需要去除填充
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dx

# ==================== 残差块实现 ====================

class BasicBlock:
    """基础残差块 - 用于ResNet-18/34"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: bool = False):
        self.stride = stride
        self.downsample = downsample

        # 主路径
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3,
                            stride=stride, padding=1, use_bias=False)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()

        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3,
                            padding=1, use_bias=False)
        self.bn2 = BatchNorm2D(out_channels)
        self.relu2 = ReLU()

        # 下采样路径
        if downsample:
            self.downsample_conv = Conv2D(in_channels, out_channels, kernel_size=1,
                                          stride=stride, use_bias=False)
            self.downsample_bn = BatchNorm2D(out_channels)
        else:
            self.downsample_conv = None
            self.downsample_bn = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        identity = x

        # 主路径
        out = self.conv1.forward(x)
        out = self.bn1.forward(out, training)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training)

        # 快捷连接
        if self.downsample:
            identity = self.downsample_conv.forward(x)
            identity = self.downsample_bn.forward(identity, training)

        # 残差连接 + ReLU
        out += identity
        out = self.relu2.forward(out)

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # 通过ReLU2的反向传播
        dout = self.relu2.backward(dout)

        # 残差连接的反向传播
        dmain = dout
        didentity = dout

        # 主路径的反向传播
        dmain = self.bn2.backward(dmain)
        dmain = self.conv2.backward(dmain)
        dmain = self.relu1.backward(dmain)
        dmain = self.bn1.backward(dmain)
        dmain = self.conv1.backward(dmain)

        # 快捷连接的反向传播
        if self.downsample:
            didentity = self.downsample_bn.backward(didentity)
            didentity = self.downsample_conv.backward(didentity)

        # 合并梯度
        dx = dmain + didentity
        return dx

    def update(self, learning_rate: float):
        """更新所有参数"""
        self.conv1.update(learning_rate)
        self.bn1.update(learning_rate)
        self.conv2.update(learning_rate)
        self.bn2.update(learning_rate)

        if self.downsample:
            self.downsample_conv.update(learning_rate)
            self.downsample_bn.update(learning_rate)


# ==================== ResNet模型 ====================

class ResNet:
    """完整的ResNet实现 - 包含完整的训练功能"""

    def __init__(self, block_type: str = 'basic', layers: List[int] = [2, 2, 2, 2],
                 num_classes: int = 10, input_channels: int = 3):
        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes

        # 初始卷积层
        self.conv1 = Conv2D(input_channels, 64, kernel_size=7, stride=2, padding=3, use_bias=False)
        self.bn1 = BatchNorm2D(64)
        self.relu = ReLU()
        self.maxpool = MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 四个阶段
        self.in_channels = 64
        if block_type == 'basic':
            self.layer1 = self._make_layer(64, layers[0], stride=1)
            self.layer2 = self._make_layer(128, layers[1], stride=2)
            self.layer3 = self._make_layer(256, layers[2], stride=2)
            self.layer4 = self._make_layer(512, layers[3], stride=2)
            final_features = 512
        else:
            raise ValueError("只支持basic块类型")

        # 分类头
        self.avgpool = GlobalAvgPool2D()
        self.fc = Linear(final_features, num_classes)

        # 训练状态
        self.training = True

    def _make_layer(self, channels: int, blocks: int, stride: int = 1):
        """创建基础残差块层"""
        layers = []
        downsample = (stride != 1) or (self.in_channels != channels)
        layers.append(BasicBlock(self.in_channels, channels, stride, downsample))
        self.in_channels = channels

        for _ in range(1, blocks):
            layers.append(BasicBlock(channels, channels))

        return layers

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """前向传播"""
        self.training = training

        # 初始层
        x = self.conv1.forward(x)
        x = self.bn1.forward(x, training)
        x = self.relu.forward(x)
        x = self.maxpool.forward(x)

        # 四个阶段
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block.forward(x, training)

        # 分类头
        x = self.avgpool.forward(x)
        x = self.fc.forward(x)

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播"""
        # 分类头的反向传播
        dout = self.fc.backward(dout)
        dout = self.avgpool.backward(dout)

        # 四个阶段的反向传播
        for layer in [self.layer4, self.layer3, self.layer2, self.layer1]:
            for block in reversed(layer):
                dout = block.backward(dout)

        # 初始层的反向传播
        dout = self.maxpool.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)

        return dout

    def update(self, learning_rate: float):
        """更新所有参数"""
        self.conv1.update(learning_rate)
        self.bn1.update(learning_rate)
        self.fc.update(learning_rate)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.update(learning_rate)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """推理预测"""
        return self.forward(x, training=False)

    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'params': {
                'conv1_weights': self.conv1.weights,
                'bn1_gamma': self.bn1.gamma,
                'bn1_beta': self.bn1.beta,
                'fc_weights': self.fc.weights,
                'fc_bias': self.fc.bias,
            },
            'running_stats': {
                'bn1_mean': self.bn1.running_mean,
                'bn1_var': self.bn1.running_var,
            },
            'config': {
                'block_type': self.block_type,
                'layers': self.layers,
                'num_classes': self.num_classes
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"模型已保存到: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # 加载参数
        self.conv1.weights = model_data['params']['conv1_weights']
        self.bn1.gamma = model_data['params']['bn1_gamma']
        self.bn1.beta = model_data['params']['bn1_beta']
        self.fc.weights = model_data['params']['fc_weights']
        self.fc.bias = model_data['params']['fc_bias']

        # 加载运行统计量
        self.bn1.running_mean = model_data['running_stats']['bn1_mean']
        self.bn1.running_var = model_data['running_stats']['bn1_var']

        print(f"模型已从 {filepath} 加载")


# ==================== 预定义的ResNet配置 ====================

def resnet18(num_classes: int = 10):
    """ResNet-18模型"""
    return ResNet('basic', [2, 2, 2, 2], num_classes)


def resnet34(num_classes: int = 10):
    """ResNet-34模型"""
    return ResNet('basic', [3, 4, 6, 3], num_classes)


# ==================== 工具函数 ====================

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """交叉熵损失"""
    m = targets.shape[0]
    probs = softmax(predictions)

    # 计算损失
    log_likelihood = -np.log(probs[range(m), targets] + 1e-8)
    loss = np.sum(log_likelihood) / m

    # 计算梯度
    grad = probs.copy()
    grad[range(m), targets] -= 1
    grad /= m

    return loss, grad


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算准确率"""
    pred_labels = np.argmax(predictions, axis=1)
    return np.mean(pred_labels == targets)


def learning_rate_scheduler(initial_lr: float, epoch: int, decay_epochs: int = 30, decay_rate: float = 0.1) -> float:
    """学习率调度器"""
    return initial_lr * (decay_rate ** (epoch // decay_epochs))


import numpy as np
import time
from typing import Dict


class ResNetTrainer:
    """完整的ResNet训练器 - 包含真正的反向传播"""

    def __init__(self, model: ResNet, train_loader, val_loader=None,
                 learning_rate: float = 0.01, weight_decay: float = 1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # 训练状态
        self.current_epoch = 0
        self.best_accuracy = 0.0

        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        print(f"初始化训练器 - 学习率: {learning_rate}")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # 确保数据是通道优先格式 (batch, channels, height, width)
            if data.ndim == 4 and data.shape[1] not in [1, 3]:
                if data.shape[3] in [1, 3]:
                    data = data.transpose(0, 3, 1, 2)

            # 前向传播
            output = self.model.forward(data, training=True)

            # 计算损失和梯度
            loss, output_grad = cross_entropy_loss(output, target)
            acc = accuracy(output, target)

            # 反向传播
            self.model.backward(output_grad)

            # 参数更新
            self.model.update(self.learning_rate)

            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches

        # 记录历史
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)

        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'time': epoch_time
        }

    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if self.val_loader is None:
            return {}

        val_loss = 0.0
        val_acc = 0.0
        num_batches = 0
        start_time = time.time()

        for data, target in self.val_loader:
            if data.ndim == 4 and data.shape[1] not in [1, 3]:
                if data.shape[3] in [1, 3]:
                    data = data.transpose(0, 3, 1, 2)

            # 前向传播
            output = self.model.predict(data)

            # 计算损失和准确率
            loss, _ = cross_entropy_loss(output, target)
            acc = accuracy(output, target)

            val_loss += loss
            val_acc += acc
            num_batches += 1

        epoch_time = time.time() - start_time
        avg_loss = val_loss / num_batches
        avg_acc = val_acc / num_batches

        # 更新最佳准确率
        if avg_acc > self.best_accuracy:
            self.best_accuracy = avg_acc

        # 记录验证历史
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'time': epoch_time
        }

    def train(self, epochs: int, save_path: str = None):
        """完整的训练循环"""
        print(f"开始训练，共 {epochs} 个epochs")
        print("=" * 60)

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            start_time = time.time()

            # 调整学习率
            current_lr = learning_rate_scheduler(self.learning_rate, epoch)
            if current_lr != self.learning_rate:
                print(f"调整学习率: {self.learning_rate:.6f} -> {current_lr:.6f}")
                self.learning_rate = current_lr

            # 训练一个epoch
            print(f"Epoch {epoch + 1}/{epochs + self.current_epoch}")
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            epoch_time = time.time() - start_time

            # 打印进度
            self._print_epoch_progress(epoch + 1, epochs + self.current_epoch,
                                       train_metrics, val_metrics, epoch_time)

            # 保存最佳模型
            if save_path and val_metrics and val_metrics['accuracy'] >= self.best_accuracy:
                self.model.save(save_path)
                print(f"  ✓ 保存最佳模型，准确率: {val_metrics['accuracy']:.4f}")

        self.current_epoch += epochs
        print("训练完成!")

    def _print_epoch_progress(self, epoch: int, total_epochs: int,
                              train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """打印epoch进度"""
        print(f'Epoch {epoch:3d}/{total_epochs} | '
              f'Time: {epoch_time:.2f}s | '
              f'Train Loss: {train_metrics["loss"]:.4f} | '
              f'Train Acc: {train_metrics["accuracy"]:.4f}', end='')

        if val_metrics:
            print(f' | Val Loss: {val_metrics["loss"]:.4f} | '
                  f'Val Acc: {val_metrics["accuracy"]:.4f}')
        else:
            print()

    def get_training_summary(self) -> Dict:
        """获取训练摘要"""
        return {
            'current_epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }

    def plot_training_history(self):
        """绘制训练历史"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        if self.val_accuracies:
            plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()



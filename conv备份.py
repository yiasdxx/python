# conv.py

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


def im2col(input_data, filter_h, filter_w, stride=1, pad=1):
    """
    将4D输入数据转换为2D矩阵
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 填充输入数据
    img = np.pad(input_data,
                 [(0, 0), (0, 0), (pad, pad), (pad, pad)],
                 'constant')

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=1):
    """
    将2D矩阵转换回4D数据
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def softmax(x):
    """Softmax函数"""
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def cross_entropy_error(y, t):
    """交叉熵误差"""
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    # 处理one-hot编码标签
    if t.ndim == 2 and t.shape[1] > 1:
        t = t.argmax(axis=1)

    # 确保标签是整数类型
    t = t.astype(np.int64).flatten()

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)


def relu_grad(x):
    """ReLU梯度"""
    grad = np.zeros_like(x)
    grad[x > 0] = 1
    return grad


class BatchNormalization:
    """批归一化层"""

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        self.running_mean = running_mean
        self.running_var = running_var
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape

        # 处理4D卷积数据
        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
            out = self._forward_2d(x, train_flg)
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            return out
        else:
            return self._forward_2d(x, train_flg)

    def _forward_2d(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 1e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            std = np.sqrt(self.running_var + 1e-7)
            xn = xc / std

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
            dx = self._backward_2d(dout)
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            dx = self._backward_2d(dout)

        return dx

    def _backward_2d(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx


class Convolution:
    """卷积层"""

    def __init__(self, W, b, stride=1, pad=1):
        self.W = W  # 滤波器权重 (FN, C, FH, FW)
        self.b = b  # 偏置 (FN,)
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        OH = (H + 2 * self.pad - FH) // self.stride + 1
        OW = (W + 2 * self.pad - FW) // self.stride + 1

        # 将输入和滤波器转换为2D矩阵
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        # 卷积计算
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)  # (N, FN, OH, OW)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    """池化层"""

    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        OH = (H - self.pool_h) // self.stride + 1
        OW = (W - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w

        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


class Affine:
    """全连接层"""

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(self.original_shape)
        return dx


class Relu:
    """ReLU激活层"""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class SoftmaxWithLoss:
    """Softmax与损失层"""

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 处理标签格式
        if self.t.ndim == 2 and self.t.shape[1] > 1:
            self.t = self.t.argmax(axis=1)#666使用硬标签版cutmix
        self.t = self.t.astype(np.int64).flatten()

        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx = dx / batch_size
        return dx


class Dropout:
    """Dropout层"""

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Adam:
    """Adam优化器"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1

        for key in params.keys():
            if key in grads:
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

                # 偏置校正
                m_hat = self.m[key] / (1 - self.beta1 ** self.iter)
                v_hat = self.v[key] / (1 - self.beta2 ** self.iter)

                params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SimpleConvNet:
    """简单的卷积神经网络"""

    def __init__(self, input_dim=(3, 64, 64),
                 conv1_param={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv2_param={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=100, output_size=10,
                 use_batchnorm=False, dropout_ratio=0.5):

        self.use_batchnorm = use_batchnorm
        self.dropout_ratio = dropout_ratio

        # 卷积参数
        FN1 = conv1_param['filter_num']
        FS1 = conv1_param['filter_size']
        FP1 = conv1_param['pad']
        FSTR1 = conv1_param['stride']

        FN2 = conv2_param['filter_num']
        FS2 = conv2_param['filter_size']
        FP2 = conv2_param['pad']
        FSTR2 = conv2_param['stride']

        # 计算各层输出尺寸
        input_size = input_dim[1]
        conv1_out = (input_size + 2 * FP1 - FS1) // FSTR1 + 1
        pool1_out = conv1_out // 2
        conv2_out = (pool1_out + 2 * FP2 - FS2) // FSTR2 + 1
        pool2_out = conv2_out // 2
        fc_input_size = FN2 * pool2_out * pool2_out

        print("网络结构计算:")
        print(f"输入: {input_dim}")
        print(f"卷积1输出: {FN1} x {conv1_out} x {conv1_out}")
        print(f"池化1输出: {FN1} x {pool1_out} x {pool1_out}")
        print(f"卷积2输出: {FN2} x {conv2_out} x {conv2_out}")
        print(f"池化2输出: {FN2} x {pool2_out} x {pool2_out}")
        print(f"全连接输入: {fc_input_size}")

        # 初始化权重参数
        self.params = {}

        # 卷积层权重 - He初始化
        he_std1 = np.sqrt(2.0 / (input_dim[0] * FS1 * FS1))
        self.params['W1'] = he_std1 * np.random.randn(FN1, input_dim[0], FS1, FS1)
        self.params['b1'] = np.zeros(FN1)

        he_std2 = np.sqrt(2.0 / (FN1 * FS2 * FS2))
        self.params['W2'] = he_std2 * np.random.randn(FN2, FN1, FS2, FS2)
        self.params['b2'] = np.zeros(FN2)

        # 全连接层权重
        he_std3 = np.sqrt(2.0 / fc_input_size)
        self.params['W3'] = he_std3 * np.random.randn(fc_input_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)

        he_std4 = np.sqrt(2.0 / hidden_size)
        self.params['W4'] = he_std4 * np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        # BatchNorm参数
        if use_batchnorm:
            self.params['gamma1'] = np.ones(FN1)
            self.params['beta1'] = np.zeros(FN1)
            self.params['gamma2'] = np.ones(FN2)
            self.params['beta2'] = np.zeros(FN2)
            self.params['gamma3'] = np.ones(hidden_size)
            self.params['beta3'] = np.zeros(hidden_size)

        # 构建网络层
        self.layers = OrderedDict()

        # 第一卷积块
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], FSTR1, FP1)
        if use_batchnorm:
            self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(2, 2, 2)

        # 第二卷积块
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], FSTR2, FP2)
        if use_batchnorm:
            self.layers['BatchNorm2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(2, 2, 2)

        # 全连接层
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        if use_batchnorm:
            self.layers['BatchNorm3'] = BatchNormalization(self.params['gamma3'], self.params['beta3'])
        self.layers['Relu3'] = Relu()
        self.layers['Dropout'] = Dropout(dropout_ratio)
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        """前向传播预测"""
        for key, layer in self.layers.items():
            if 'BatchNorm' in key or 'Dropout' in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        """计算损失"""
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        """计算准确率"""
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        if x.shape[0] < batch_size:
            batch_size = x.shape[0]

        acc = 0.0
        for i in range(0, x.shape[0], batch_size):
            tx = x[i:i + batch_size]
            tt = t[i:i + batch_size]# 真实标签
            y = self.predict(tx, train_flg=False)# 预测
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        """计算梯度"""
        # 前向传播
        self.loss(x, t)

        # 反向传播
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()#反转列表
        for layer in layers:
            dout = layer.backward(dout)

        # 收集梯度
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Conv2'].dW
        grads['b2'] = self.layers['Conv2'].db
        grads['W3'] = self.layers['Affine1'].dW
        grads['b3'] = self.layers['Affine1'].db
        grads['W4'] = self.layers['Affine2'].dW
        grads['b4'] = self.layers['Affine2'].db

        if self.use_batchnorm:
            grads['gamma1'] = self.layers['BatchNorm1'].dgamma
            grads['beta1'] = self.layers['BatchNorm1'].dbeta
            grads['gamma2'] = self.layers['BatchNorm2'].dgamma
            grads['beta2'] = self.layers['BatchNorm2'].dbeta
            grads['gamma3'] = self.layers['BatchNorm3'].dgamma
            grads['beta3'] = self.layers['BatchNorm3'].dbeta

        return grads


class Trainer:
    """只有训练集和验证集的训练器"""

    def __init__(self, network, x_train, t_train, x_val, t_val,
                 epochs=20, batch_size=100, optimizer='adam', learning_rate=0.001,
                 use_augmentation=True, use_cutmix=True, cutmix_alpha=1.0, cutmix_prob=0.5, patience=100,
                 aug_start_threshold=0.85):  # 新增：数据增强启动阈值

        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val
        self.patience = patience

        # 早停相关属性
        self.best_val_acc = 0.0
        self.no_improve_count = 0
        self.best_params = None

        # 数据增强控制
        self.use_augmentation = use_augmentation
        self.use_cutmix = use_cutmix
        self.aug_start_threshold = aug_start_threshold  # 数据增强启动阈值
        self.aug_enabled = False  # 初始不启用数据增强
        self.high_acc_count = 0  # 连续高准确率计数

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.optimizer_type = optimizer

        # 初始化数据增强器（但不立即使用）
        if use_augmentation:
            from increase2025 import DataAugmentor
            self.augmenter = DataAugmentor(use_cutmix=use_cutmix,
                                           cutmix_alpha=cutmix_alpha,
                                           cutmix_prob=cutmix_prob)
            print(f"✅ 数据增强已配置，将在训练准确率连续3次达到{aug_start_threshold}后启用")
            if use_cutmix:
                print(f"   CutMix配置 - alpha: {cutmix_alpha}, 概率: {cutmix_prob}")
        else:
            self.augmenter = None
            print("❌ 数据增强已禁用")

        # 优化器设置
        if optimizer == 'adam':
            self.optimizer = Adam(lr=learning_rate)
        else:
            self.optimizer = None

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)

        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []

    def augment_batch(self, x_batch, t_batch=None):
        """动态增强批次数据 - 每个训练步骤都会调用"""
        if not self.aug_enabled or self.augmenter is None:
            return x_batch, t_batch

        augmented_batch = []

        # 基础数据增强（单张图像）
        for i in range(x_batch.shape[0]):
            img = x_batch[i]

            # 确保图像格式正确 (C, H, W) -> (H, W, C) 进行增强
            if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
                # 转置为 (H, W, C) 进行增强
                if img.shape[0] == 3:  # RGB
                    img_hwc = img.transpose(1, 2, 0)
                else:  # 灰度图
                    img_hwc = img[0]  # (1, H, W) -> (H, W)
            else:
                img_hwc = img

            # 反归一化到0-255范围进行增强
            if img_hwc.max() <= 1.0:
                img_255 = (img_hwc * 255).astype(np.uint8)
            else:
                img_255 = img_hwc.astype(np.uint8)

            try:
                # 应用单张图像增强
                aug_img = self.augmenter.augment_single_image(img_255)
            except Exception as e:
                print(f"⚠️ 图像增强失败，使用原图: {e}")
                aug_img = img_255  # 如果增强失败，使用原图

            # 重新归一化到0-1范围
            if img_hwc.max() <= 1.0:
                aug_img = aug_img.astype(np.float32) / 255.0
            else:
                aug_img = aug_img.astype(np.float32)

            # 转回原始格式 (H, W, C) -> (C, H, W)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                if img.shape[0] == 3:  # RGB
                    aug_img = aug_img.transpose(2, 0, 1)
                else:  # 灰度图
                    aug_img = aug_img[np.newaxis, :, :]  # (H, W) -> (1, H, W)

            augmented_batch.append(aug_img)

        x_augmented = np.array(augmented_batch)

        # 应用CutMix增强
        if self.use_cutmix and t_batch is not None and t_batch.ndim == 2:  # 需要one-hot标签
            try:
                x_augmented, t_batch = self.augmenter.apply_cutmix(x_augmented, t_batch)
            except Exception as e:
                print(f"⚠️ CutMix增强失败: {e}")
                # 如果CutMix失败，使用基础增强的数据

        return x_augmented, t_batch

    def check_augmentation_condition(self, train_acc):
        """检查是否满足启用数据增强的条件"""
        if not self.use_augmentation or self.aug_enabled:
            return

        # 检查准确率是否达到阈值
        if train_acc >= self.aug_start_threshold:
            self.high_acc_count += 1
            print(f"🎯 高准确率计数: {self.high_acc_count}/3 (当前准确率: {train_acc:.4f})")
        else:
            self.high_acc_count = 0  # 重置计数

        # 如果连续3次达到阈值，启用数据增强
        if self.high_acc_count >= 3:
            self.aug_enabled = True
            print("🚀 训练准确率连续3次达到阈值，现在启用数据增强！")
            print("   这将帮助模型更好地泛化，减少过拟合风险")

    def train_step(self):
        """单次训练步骤"""
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 动态数据增强（仅在启用时使用）
        if self.aug_enabled and self.augmenter is not None:
            x_batch, t_batch = self.augment_batch(x_batch, t_batch)

        # 计算梯度
        grads = self.network.gradient(x_batch, t_batch)

        # 更新参数
        if self.optimizer_type == 'adam':
            self.optimizer.update(self.network.params, grads)
        else:
            for key in grads.keys():
                self.network.params[key] -= self.lr * grads[key]

        # 计算损失
        loss = self.network.loss(x_batch, t_batch)#调用loss->调用predict,返回时调用forward
        self.train_loss_list.append(loss)

        # 每个epoch结束时计算准确率
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            train_acc = self.network.accuracy(self.x_train, self.t_train)
            val_acc = self.network.accuracy(self.x_val, self.t_val)

            self.train_acc_list.append(train_acc)
            self.val_acc_list.append(val_acc)

            # 检查是否满足数据增强启用条件
            self.check_augmentation_condition(train_acc)

            print(f"=== Epoch {self.current_epoch} ===")
            print(f"训练集准确率: {train_acc:.4f}",end=" ")
            print(f"验证集准确率: {val_acc:.4f}",end=" ")
            print(f"损失: {loss:.4f}",end=" ")
            if self.aug_enabled:
                print("🌟 数据增强: 已启用")
            else:
                print("⏳ 数据增强: 等待触发")

        self.current_iter += 1

    def train(self):
        """完整训练过程"""
        print("开始训练...")
        print(f"网络结构: Conv-BN-Relu-Pool-Conv-BN-Relu-Pool-Affine-BN-Relu-Dropout-Affine")
        print(f"训练轮次: {self.epochs}")
        print(f"批次大小: {self.batch_size}")
        print(f"学习率: {self.lr}")
        print(f"训练集样本: {self.train_size}")
        print(f"验证集样本: {len(self.x_val)}")
        print(f"优化器: {self.optimizer_type}")
        print(f"使用BatchNorm: {self.network.use_batchnorm}")
        print(f"数据增强策略: 动态启用 (阈值: {self.aug_start_threshold})")
        print(f"最终使用CutMix: {self.use_cutmix}\n")

        # 重置早停计数器
        self.best_val_acc = 0.0
        self.no_improve_count = 0
        self.best_params = None
        self.aug_enabled = False  # 确保初始状态
        self.high_acc_count = 0

        for i in range(self.max_iter):
            self.train_step()

            # 每个epoch结束时检查早停条件
            if self.current_iter % self.iter_per_epoch == 0 and len(self.val_acc_list) > 0:
                current_val_acc = self.val_acc_list[-1]

                # 如果当前验证准确率比最佳值好，保存模型参数
                if current_val_acc > self.best_val_acc:
                    self.best_val_acc = current_val_acc
                    self.no_improve_count = 0
                    # 保存当前最佳参数
                    self.best_params = {}
                    for key, val in self.network.params.items():
                        self.best_params[key] = val.copy()
                else:
                    self.no_improve_count += 1

                # 早停检查
                if self.no_improve_count >= self.patience and len(self.x_val) > 0:
                    print(f"\n早停: 验证准确率在 {self.patience} 个epoch内未提升")
                    # 恢复最佳参数
                    if self.best_params is not None:
                        for key in self.best_params.keys():
                            self.network.params[key] = self.best_params[key].copy()
                    break

        # 最终评估
        final_train_acc = self.network.accuracy(self.x_train, self.t_train)
        final_val_acc = self.network.accuracy(self.x_val, self.t_val) if len(self.x_val) > 0 else 0

        print("\n=== 训练完成 ===")
        print(f"最终训练准确率: {final_train_acc:.4f}")
        print(f"最终验证准确率: {final_val_acc:.4f}")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"数据增强状态: {'已启用' if self.aug_enabled else '未启用'}")
        if len(self.x_val) > 0:
            print(f"过拟合程度: {final_train_acc - final_val_acc:.4f}")

        return self.train_loss_list, self.train_acc_list, self.val_acc_list

    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_list)
        plt.title('Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)

        # 准确率曲线
        plt.subplot(1, 2, 2)
        epochs = range(1, len(self.train_acc_list) + 1)
        plt.plot(epochs, self.train_acc_list, label='Train Accuracy', marker='o')
        if len(self.val_acc_list) > 0:
            plt.plot(epochs, self.val_acc_list, label='Validation Accuracy', marker='s')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
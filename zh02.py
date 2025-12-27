import numpy as np
from common.util import im2col
from collections import OrderedDict


def im2col(input_data, filter_h, filter_w, stride=1, pad=1):
    """
    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
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
    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------
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
    """Softmax函数实现"""
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def cross_entropy_error(y, t):
    """交叉熵误差 - 普通函数"""
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    # 如果监督数据是one-hot向量，转换为正确解标签的索引
    if t.ndim == 2 and t.shape[1] > 1:  # one-hot编码
        t = t.argmax(axis=1)

    # 确保标签是整数类型的一维数组
    t = t.astype(np.int64)
    if t.ndim > 1:
        t = t.flatten()

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def relu(x):
    """ReLU函数"""
    return np.maximum(0, x)


def relu_grad(x):
    """ReLU函数的梯度"""
    grad = np.zeros_like(x)
    grad[x > 0] = 1
    return grad


# Batch Normalization层
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        self.reshape_from_4d = False
        self.original_4d_shape = None
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
        self.reshape_from_4d = False

        # 处理4维卷积数据
        if x.ndim == 4:
            N, C, H, W = x.shape

            self.reshape_from_4d = True
            self.original_4d_shape = (N, C, H, W)

            x_transposed = x.transpose(0, 2, 3, 1)

            x_reshaped = x_transposed.reshape(-1, C)

            out_reshaped = self.__forward_2d(x_reshaped, train_flg)

            out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            return out
        else:
            return self.__forward_2d(x, train_flg)

    def __forward_2d(self, x, train_flg):

        if self.running_mean is None:
            D = x.shape[1]
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

        if self.reshape_from_4d:
            N, C, H, W = self.original_4d_shape
            dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            dout_reshaped = dout.reshape(self.xn.shape if self.xn is not None else dout.shape)

        dx_reshaped = self.__backward_2d(dout_reshaped)

        if self.reshape_from_4d:
            N, C, H, W = self.original_4d_shape
            dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            dx = dx_reshaped.reshape(*self.input_shape)

        return dx

    def __backward_2d(self, dout):

        if self.batch_size is None:
            self.batch_size = dout.shape[0]

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


# 网络层
class Convolution:
    def __init__(self, W, b, stride=1, pad=1):
        self.W = W
        self.b = b
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
        OH = int(1 + (H + 2 * self.pad - FH) / self.stride)
        OW = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_w = self.W.reshape(FN, -1).T
        out = np.dot(col, col_w) + self.b
        out = out.reshape(N, OH, OW, -1).transpose((0, 3, 1, 2))

        self.x = x
        self.col = col
        self.col_W = col_w

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
    def __init__(self, pool_h, pool_w, stride, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        OH = int(1 + (H - self.pool_h) / self.stride)
        OW = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        out = out.reshape(N, OH, OW, C).transpose((0, 3, 1, 2))
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


class Adam:
    """修正的Adam优化器 - 添加偏置校正"""

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

                m_hat = self.m[key] / (1 - self.beta1 ** self.iter)
                v_hat = self.v[key] / (1 - self.beta2 ** self.iter)

                params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Affine:
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

class Dropout:
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

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return np.maximum(x, 0)

    def backward(self, dout):
        dx = dout * self.mask
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 处理one-hot编码标签
        if self.t.ndim == 2 and self.t.shape[1] > 1:
            self.t = self.t.argmax(axis=1)

        # 确保标签是整数类型的一维数组
        self.t = self.t.astype(np.int64)
        if self.t.ndim > 1:
            self.t = self.t.flatten()

        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx = dx / batch_size
        return dx


# 具体网络模型
class SimpleConvNet:
    """针对大尺寸输入的卷积神经网络 - 全连接层也使用BN"""

    def __init__(self, input_dim=(3, 64, 64), dropout_ratio=0.5,
                 conv1_param={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv2_param={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv3_param={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=100, output_size=10, use_batchnorm=False):

        self.dropout_ratio = dropout_ratio
        self.use_batchnorm = use_batchnorm

        # 第一层卷积参数
        filter_num1 = conv1_param['filter_num']
        filter_size1 = conv1_param['filter_size']
        filter_pad1 = conv1_param['pad']
        filter_stride1 = conv1_param['stride']

        # 第二层卷积参数
        filter_num2 = conv2_param['filter_num']
        filter_size2 = conv2_param['filter_size']
        filter_pad2 = conv2_param['pad']
        filter_stride2 = conv2_param['stride']

        # 第三层卷积参数
        filter_num3 = conv3_param['filter_num']
        filter_size3 = conv3_param['filter_size']
        filter_pad3 = conv3_param['pad']
        filter_stride3 = conv3_param['stride']

        input_size = input_dim[1]

        # 计算各层输出尺寸
        conv1_output_size = (input_size - filter_size1 + 2 * filter_pad1) // filter_stride1 + 1
        pool1_output_size = conv1_output_size // 2
        conv2_output_size = (pool1_output_size - filter_size2 + 2 * filter_pad2) // filter_stride2 + 1
        pool2_output_size = conv2_output_size // 2
        conv3_output_size = (pool2_output_size - filter_size3 + 2 * filter_pad3) // filter_stride3 + 1
        pool3_output_size = conv3_output_size // 2
        fc_input_size = filter_num3 * pool3_output_size * pool3_output_size

        print(f"网络结构计算:")
        print(f"输入: {input_dim}")
        print(f"卷积1输出: {filter_num1} x {conv1_output_size} x {conv1_output_size}")
        print(f"池化1输出: {filter_num1} x {pool1_output_size} x {pool1_output_size}")
        print(f"卷积2输出: {filter_num2} x {conv2_output_size} x {conv2_output_size}")
        print(f"池化2输出: {filter_num2} x {pool2_output_size} x {pool2_output_size}")
        print(f"卷积3输出: {filter_num3} x {conv3_output_size} x {conv3_output_size}")
        print(f"池化3输出: {filter_num3} x {pool3_output_size} x {pool3_output_size}")
        print(f"全连接层输入: {fc_input_size}")
        print(f"使用Batch Normalization: {use_batchnorm}")

        # 初始化权重
        self.params = {}

        # 卷积层权重
        he_std_conv1 = np.sqrt(2.0 / (input_dim[0] * filter_size1 * filter_size1))
        self.params['W1'] = he_std_conv1 * np.random.randn(filter_num1, input_dim[0], filter_size1, filter_size1)
        self.params['b1'] = np.zeros(filter_num1)

        he_std_conv2 = np.sqrt(2.0 / (filter_num1 * filter_size2 * filter_size2))
        self.params['W2'] = he_std_conv2 * np.random.randn(filter_num2, filter_num1, filter_size2, filter_size2)
        self.params['b2'] = np.zeros(filter_num2)

        # 第三层卷积权重
        he_std_conv3 = np.sqrt(2.0 / (filter_num2 * filter_size3 * filter_size3))
        self.params['W3'] = he_std_conv3 * np.random.randn(filter_num3, filter_num2, filter_size3, filter_size3)
        self.params['b3'] = np.zeros(filter_num3)

        # 全连接层权重
        he_std_fc1 = np.sqrt(2.0 / fc_input_size)
        self.params['W4'] = he_std_fc1 * np.random.randn(fc_input_size, hidden_size)
        self.params['b4'] = np.zeros(hidden_size)

        he_std_fc2 = np.sqrt(2.0 / hidden_size)
        self.params['W5'] = he_std_fc2 * np.random.randn(hidden_size, output_size)
        self.params['b5'] = np.zeros(output_size)

        # Batch Normalization 参数 - 为所有层添加BN参数
        if self.use_batchnorm:
            # 卷积层的BN参数
            self.params['gamma1'] = np.ones(filter_num1)
            self.params['beta1'] = np.zeros(filter_num1)
            self.params['gamma2'] = np.ones(filter_num2)
            self.params['beta2'] = np.zeros(filter_num2)
            self.params['gamma3'] = np.ones(filter_num3)
            self.params['beta3'] = np.zeros(filter_num3)

            # 全连接层的BN参数（新增）
            self.params['gamma4'] = np.ones(hidden_size)  # 第一个全连接层后的BN
            self.params['beta4'] = np.zeros(hidden_size)

        # 生成层
        self.layers = OrderedDict()

        # 第一卷积块
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv1_param['stride'], conv1_param['pad'])
        if self.use_batchnorm:
            self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 第二卷积块
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv2_param['stride'], conv2_param['pad'])
        if self.use_batchnorm:
            self.layers['BatchNorm2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 第三卷积块
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'],
                                           conv3_param['stride'], conv3_param['pad'])
        if self.use_batchnorm:
            self.layers['BatchNorm3'] = BatchNormalization(self.params['gamma3'], self.params['beta3'])
        self.layers['Relu3'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 第一个全连接块（添加BN层）
        self.layers['Affine1'] = Affine(self.params['W4'], self.params['b4'])
        if self.use_batchnorm:
            self.layers['BatchNorm4'] = BatchNormalization(self.params['gamma4'], self.params['beta4'])
        self.layers['Relu4'] = Relu()
        self.layers['Dropout1'] = Dropout(dropout_ratio)

        # 第二个全连接层（输出层，通常不加BN）
        self.layers['Affine2'] = Affine(self.params['W5'], self.params['b5'])

        self.last_layer = SoftmaxWithLoss()
    def predict(self, x, train_flg=False):
        """前向传播预测

        Args:
            x: 输入数据
            train_flg: 是否为训练模式（影响BN层和Dropout层行为）
        """
        for key, layer in self.layers.items():
            if 'BatchNorm' in key or 'Dropout' in key:
                # BN层和Dropout层需要train_flg参数
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

        acc = 0.0
        for i in range(0, x.shape[0], batch_size):
            tx = x[i:i + batch_size]
            tt = t[i:i + batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        """计算梯度"""
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Conv2'].dW
        grads['b2'] = self.layers['Conv2'].db
        grads['W3'] = self.layers['Conv3'].dW
        grads['b3'] = self.layers['Conv3'].db
        grads['W4'] = self.layers['Affine1'].dW
        grads['b4'] = self.layers['Affine1'].db
        grads['W5'] = self.layers['Affine2'].dW
        grads['b5'] = self.layers['Affine2'].db

        if self.use_batchnorm:
            grads['gamma1'] = self.layers['BatchNorm1'].dgamma
            grads['beta1'] = self.layers['BatchNorm1'].dbeta
            grads['gamma2'] = self.layers['BatchNorm2'].dgamma
            grads['beta2'] = self.layers['BatchNorm2'].dbeta
            grads['gamma3'] = self.layers['BatchNorm3'].dgamma
            grads['beta3'] = self.layers['BatchNorm3'].dbeta
            grads['gamma4'] = self.layers['BatchNorm4'].dgamma  # 新增全连接层BN梯度
            grads['beta4'] = self.layers['BatchNorm4'].dbeta  # 新增全连接层BN梯度

        return grads

# 训练器
# 在 Trainer 类中添加数据增强功能
class Trainer:
    """只有训练集和验证集的训练器"""

    def __init__(self, network, x_train, t_train, x_val, t_val,
                 epochs=20, batch_size=100, optimizer='adam', learning_rate=0.001,
                 augmentor=None, enable_augmentation_threshold=0.8):  # 新增参数
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.optimizer_type = optimizer

        # 数据增强相关属性
        self.augmentor = augmentor
        self.enable_augmentation_threshold = enable_augmentation_threshold
        self.augmentation_enabled = False
        self.high_accuracy_count = 0
        self.last_train_accuracies = []

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

    def check_and_enable_augmentation(self, current_train_acc):
        """检查是否应该启用数据增强"""
        if self.augmentor is None or self.augmentation_enabled:
            return

        # 记录最近的准确率
        self.last_train_accuracies.append(current_train_acc)
        if len(self.last_train_accuracies) > 3:
            self.last_train_accuracies.pop(0)

        # 检查是否连续三次达到阈值
        if len(self.last_train_accuracies) == 3:
            if all(acc >= self.enable_augmentation_threshold for acc in self.last_train_accuracies):
                self.augmentation_enabled = True
                print(f"\n🎯 训练准确率连续三次达到 {self.enable_augmentation_threshold}，启用数据增强！")
                print(f"最近三次准确率: {self.last_train_accuracies}")

    def prepare_training_batch(self, batch_mask):
        """准备训练批次，应用数据增强"""
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 如果启用了数据增强且增强器可用
        if self.augmentation_enabled and self.augmentor is not None:
            try:
                # 将数据转换为适合增强的格式 (N, C, H, W) -> (N, H, W, C)
                if x_batch.ndim == 4 and x_batch.shape[1] in [1, 3]:  # 通道在前格式
                    x_batch_aug = x_batch.transpose(0, 2, 3, 1)
                else:
                    x_batch_aug = x_batch.copy()

                # 反归一化到0-255范围进行增强
                if x_batch_aug.max() <= 1.0:
                    x_batch_aug = (x_batch_aug * 255).astype(np.uint8)

                # 应用增强到每个图像
                augmented_batch = []
                for i in range(len(x_batch_aug)):
                    augmented_img = self.augmentor.augment_single_image(x_batch_aug[i])
                    augmented_batch.append(augmented_img)

                x_batch_aug = np.array(augmented_batch)

                # 重新归一化并转换回原始格式
                x_batch_aug = x_batch_aug.astype(np.float32) / 255.0
                if x_batch.ndim == 4 and x_batch.shape[1] in [1, 3]:
                    x_batch_aug = x_batch_aug.transpose(0, 3, 1, 2)

                return x_batch_aug, t_batch

            except Exception as e:
                print(f"数据增强失败，使用原始数据: {e}")
                return x_batch, t_batch

        return x_batch, t_batch

    def train_step(self):
        """单次训练步骤"""
        batch_mask = np.random.choice(self.train_size, self.batch_size)

        # 准备批次数据（可能应用增强）
        x_batch, t_batch = self.prepare_training_batch(batch_mask)

        # 计算梯度
        grads = self.network.gradient(x_batch, t_batch)

        # 更新参数
        if self.optimizer_type == 'adam':
            self.optimizer.update(self.network.params, grads)
        else:
            for key in grads.keys():
                self.network.params[key] -= self.lr * grads[key]

        # 计算损失
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        # 每个epoch结束时计算准确率
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            train_acc = self.network.accuracy(self.x_train, self.t_train)
            val_acc = self.network.accuracy(self.x_val, self.t_val)

            self.train_acc_list.append(train_acc)
            self.val_acc_list.append(val_acc)

            # 检查并可能启用数据增强
            self.check_and_enable_augmentation(train_acc)

            print(f"=== Epoch {self.current_epoch} ===")
            print(f"训练集准确率: {train_acc:.4f}")
            print(f"验证集准确率: {val_acc:.4f}")
            print(f"损失: {loss:.4f}")
            if self.augmentation_enabled:
                print("🔧 数据增强: 已启用")
            else:
                print(f"数据增强: 未启用 (需要连续3次准确率 ≥ {self.enable_augmentation_threshold})")

        self.current_iter += 1

    def train(self):
        """完整训练"""
        print("开始训练...")
        print(f"网络结构: Conv-Relu-Pool-Conv-Relu-Pool-Affine-Relu-Affine")
        print(f"激活函数: ReLU")
        print(f"权重初始化: He初始化")
        print(f"训练轮次: {self.epochs}")
        print(f"批次大小: {self.batch_size}")
        print(f"学习率: {self.lr}")
        print(f"训练集样本数: {self.train_size}")
        print(f"验证集样本数: {self.x_val.shape[0]}")
        print(f"使用优化器: {self.optimizer_type}")
        print(f"使用BatchNorm: {self.network.use_batchnorm}")
        print(f"数据增强阈值: 连续3次训练准确率 ≥ {self.enable_augmentation_threshold}")
        print(f"数据增强器: {'已配置' if self.augmentor is not None else '未配置'}\n")

        for i in range(self.max_iter):
            self.train_step()

        # 最终评估
        final_train_acc = self.network.accuracy(self.x_train, self.t_train)
        final_val_acc = self.network.accuracy(self.x_val, self.t_val)

        print("=== 训练完成 ===")
        print(f"最终训练集准确率: {final_train_acc:.4f}")
        print(f"最终验证集准确率: {final_val_acc:.4f}")
        print(f"过拟合程度: {final_train_acc - final_val_acc:.4f}")
        if self.augmentation_enabled:
            print("数据增强状态: 已启用")
        else:
            print("数据增强状态: 未启用")

        return self.train_loss_list, self.train_acc_list, self.val_acc_list

    def plot_training_history(self):
        """绘制训练历史"""
        import matplotlib.pyplot as plt

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
        plt.plot(epochs, self.val_acc_list, label='Validation Accuracy', marker='s')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
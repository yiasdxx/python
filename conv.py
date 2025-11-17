import numpy as np
from  common.util  import im2col
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
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # transpose调整后(N,out_h, out_w, C, filter_h, filter_w)每一行是一个滤波器框到的对应数据
    # 第一行中，通道一元素依次排列，接着是通道二的数据，一直到所有的通道数据显示完毕后，进入下一行
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
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
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
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

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross(self,y,t):
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

    # 调试信息
    # print(f"交叉熵调试 - y形状: {y.shape}, t形状: {t.shape}")
    # print(f"交叉熵调试 - t数据类型: {t.dtype}, t值范围: {t.min()}~{t.max()}")

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def relu(x):
    """ReLU函数"""
    return np.maximum(0, x)

def relu_grad(x):
    """ReLU函数的梯度"""
    grad = np.zeros_like(x)
    grad[x > 0] = 1
    return grad

#网络层
class Convolution:
    def __init__(self, W, b,stride=1,pad=1):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        # 添加缓存
        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self,x):
        FN,C,FH,FW = self.W.shape
        N,C,H,W = x.shape
        OH= int(1+(H+2*self.pad-FH)/self.stride)
        OW= int(1+(W+2*self.pad-FW)/self.stride)

        col=im2col(x,FH,FW,self.stride,self.pad)
        col_w= self.W.reshape(FN,-1).T
        out = np.dot(col,col_w) + self.b
        out = out.reshape(N,OH,OW,-1).transpose((0,3,1,2))
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride,pad=1):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        # 需要添加缓存
        self.x = None
        self.arg_max = None  # 需要记录最大值位置

    def forward(self,x):
        N,C,H,W = x.shape
        OH = int(1 + (H  - self.pool_h) / self.stride)
        OW = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col,axis=1)

        out = out.reshape(N,OH,OW,C).transpose((0,3,1,2))
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
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
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
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            # self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            # self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

class Affine:
    def __init__(self , W ,b):
        self.W = W
        self.b = b

        self.x = None
        self.original_shape = None

        self.dW = None
        self.db = None

    def forward(self,x):

        self.original_shape = x.shape
        x = x.reshape(x.shape[0],-1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self,dout):
        dx =np.dot(dout, self.W.T)
        self.dW =np.dot(self.x.T,dout)
        self.db =np.sum(dout,axis=0)

        dx=dx.reshape(self.original_shape)
        return dx

class Relu:
    def __init__(self):
            self.mask = None

    def forward(self, x):
            self.mask = (x > 0)
            return np.maximum(x,0)

    def backward(self, dout):
            dx = dout * self.mask  # 正值位置梯度通过，负值位置梯度为0
            return dx


# common/layers.py
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 处理one-hot编码标签 - 确保转换为一维整数数组
        if self.t.ndim == 2 and self.t.shape[1] > 1:
            self.t = self.t.argmax(axis=1)

        # 确保标签是整数类型的一维数组
        self.t = self.t.astype(np.int64)
        if self.t.ndim > 1:
            self.t = self.t.flatten()

        # 调试信息
        # print(f"SoftmaxWithLoss调试 - x形状: {x.shape}, t形状: {self.t.shape}")
        # print(f"SoftmaxWithLoss调试 - y形状: {self.y.shape}")

        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        # 注意：这里self.t已经是一维整数标签了
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx = dx / batch_size

        return dx

#具体网络模型
# simple_convnet.py
import numpy as np
from collections import OrderedDict
from common.layers import *

class SimpleConvNet:
    """针对大尺寸输入(3,256,256)的卷积神经网络

    网络结构:
    conv1 - relu1 - pool1 - conv2 - relu2 - pool2 - affine1 - relu3 - affine2 - softmax

    使用ReLU激活函数和He初始化
    """

    def __init__(self, input_dim=(3, 256, 256),
                 conv1_param={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv2_param={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=100, output_size=10):

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

        input_size = input_dim[1]  #图像尺寸

        # 计算各层输出尺寸
        # 第一层卷积输出
        conv1_output_size = (input_size - filter_size1 + 2 * filter_pad1) // filter_stride1 + 1
        # 第一层池化输出 (2x2池化，步长2)
        pool1_output_size = conv1_output_size // 2
        # 第二层卷积输出
        conv2_output_size = (pool1_output_size - filter_size2 + 2 * filter_pad2) // filter_stride2 + 1
        # 第二层池化输出
        pool2_output_size = conv2_output_size // 2
        # 全连接层输入尺寸
        fc_input_size = filter_num2 * pool2_output_size * pool2_output_size

        print(f"网络结构计算:")
        print(f"输入: {input_dim}")
        print(f"卷积1输出: {filter_num1} x {conv1_output_size} x {conv1_output_size}")
        print(f"池化1输出: {filter_num1} x {pool1_output_size} x {pool1_output_size}")
        print(f"卷积2输出: {filter_num2} x {conv2_output_size} x {conv2_output_size}")
        print(f"池化2输出: {filter_num2} x {pool2_output_size} x {pool2_output_size}")
        print(f"全连接层输入: {fc_input_size}")

        # He初始化权重
        self.params = {}

        # 第一层卷积 - He初始化
        he_std_conv1 = np.sqrt(2.0 / (input_dim[0] * filter_size1 * filter_size1))
        self.params['W1'] = he_std_conv1 * np.random.randn(filter_num1, input_dim[0], filter_size1, filter_size1)
        self.params['b1'] = np.zeros(filter_num1)

        # 第二层卷积 - He初始化
        he_std_conv2 = np.sqrt(2.0 / (filter_num1 * filter_size2 * filter_size2))
        self.params['W2'] = he_std_conv2 * np.random.randn(filter_num2, filter_num1, filter_size2, filter_size2)
        self.params['b2'] = np.zeros(filter_num2)

        # 第一个全连接层 - He初始化
        he_std_fc1 = np.sqrt(2.0 / fc_input_size)
        self.params['W3'] = he_std_fc1 * np.random.randn(fc_input_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)

        # 输出层 - He初始化
        he_std_fc2 = np.sqrt(2.0 / hidden_size)
        self.params['W4'] = he_std_fc2 * np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        # 第一卷积块
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv1_param['stride'], conv1_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 第二卷积块
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv2_param['stride'], conv2_param['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 全连接层
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """前向传播"""
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """计算损失"""
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        """计算准确率"""
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        # 分批处理以避免内存不足
        for i in range(0, x.shape[0], batch_size):
            tx = x[i:i + batch_size]
            tt = t[i:i + batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)# 预测结果转换为整数标签
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        """计算梯度（误差反向传播法）"""

        # 前向传播
        self.loss(x, t)
        # 反向传播
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 保存梯度
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Conv2'].dW
        grads['b2'] = self.layers['Conv2'].db
        grads['W3'] = self.layers['Affine1'].dW
        grads['b3'] = self.layers['Affine1'].db
        grads['W4'] = self.layers['Affine2'].dW
        grads['b4'] = self.layers['Affine2'].db

        return grads
# 训练器
# trainer.py
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    """只有训练集和验证集的训练器"""

    def __init__(self, network, x_train, t_train, x_val, t_val,
                 epochs=100, batch_size=100, optimizer='adam',learning_rate=0.001):

        self.network = network
        self.x_train = x_train  # 训练数据
        self.t_train = t_train
        self.x_val = x_val  # 验证数据
        self.t_val = t_val

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.optimizer_type = optimizer

        if optimizer == 'adam':
            self.lr = learning_rate  # Adam通常使用更小的学习率
            self.optimizer = Adam(lr=learning_rate)
        else:  # sgd
            self.lr = learning_rate * 10  # SGD通常需要更大的学习率
            self.optimizer = None

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // batch_size, 1)
        print(self.train_size)
        print(batch_size)
        print(self.iter_per_epoch)
        self.max_iter = int(epochs * self.iter_per_epoch)

        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []  # 训练集准确率
        self.val_acc_list = []  # 验证集准确率

    def train_step(self):
        """单次训练步骤"""
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # # 确保标签是整数类型
        # if t_batch.ndim == 2 and t_batch.shape[1] > 1:
        #     print("调试 - 检测到one-hot编码标签，正在转换...")
        #     t_batch = t_batch.argmax(axis=1)
        #     print(f"调试 - 转换后标签形状: {t_batch.shape}")
        #     print(f"调试 - 转换后标签值: {t_batch[:10]}")

        # 计算梯度
        grads = self.network.gradient(x_batch, t_batch)

        # 更新参数
        if self.optimizer_type == 'adam':
            self.optimizer.update(self.network.params, grads)
        else:  # sgd
            for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'):
                if key in grads:
                    self.network.params[key] -= self.lr * grads[key]

        # 计算损失
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        # 每个epoch结束时计算准确率
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            # 训练集准确率
            train_acc = self.network.accuracy(self.x_train, self.t_train)
            # 验证集准确率
            val_acc = self.network.accuracy(self.x_val, self.t_val)

            self.train_acc_list.append(train_acc)
            self.val_acc_list.append(val_acc)

            print(f"=== Epoch {self.current_epoch} ===")
            print(f"训练集准确率: {train_acc:.4f}")
            print(f"验证集准确率: {val_acc:.4f}")
            print(f"损失: {loss:.4f}\n")
            if hasattr(self, 'optimizer') and self.optimizer:
                print(f"学习率: {self.lr * np.sqrt(1.0 - self.optimizer.beta2 ** self.optimizer.iter) 
                                 / (1.0 - self.optimizer.beta1 ** self.optimizer.iter):.6f}")
        self.current_iter += 1

    def train(self):
        """完整训练"""
        print("开始训练...")
        # print(f"输入尺寸: (3, 256, 256)")
        print(f"网络结构: Conv-Relu-Pool-Conv-Relu-Pool-Affine-Relu-Affine")
        print(f"激活函数: ReLU")
        print(f"权重初始化: He初始化")
        print(f"训练轮次: {self.epochs}")
        print(f"批次大小: {self.batch_size}")
        print(f"学习率: {self.lr}")
        print(f"训练集样本数: {self.train_size}")
        print(f"验证集样本数: {self.x_val.shape[0]}\n")

        for i in range(self.max_iter):
            self.train_step()

        # 最终评估
        final_train_acc = self.network.accuracy(self.x_train, self.t_train)
        final_val_acc = self.network.accuracy(self.x_val, self.t_val)

        print("=== 训练完成 ===")
        print(f"最终训练集准确率: {final_train_acc:.4f}")
        print(f"最终验证集准确率: {final_val_acc:.4f}")
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
        plt.plot(epochs, self.val_acc_list, label='Validation Accuracy', marker='s')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
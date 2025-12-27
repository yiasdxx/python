import gradio as gr
import cv2
import numpy as np
import pickle

# 加载模型
with open('simple_convnet_model2.pkl', 'rb') as f:
    data = pickle.load(f)

class_names = data.get('class_names', [f'类别{i}' for i in range(data['num_classes'])])

# 创建网络
from conv import SimpleConvNet

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



def predict(img):
    # 预处理
    img = cv2.resize(img, (64, 64))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = img[np.newaxis, :]

    # 预测
    output = model.predict(img, train_flg=False)[0]
    probs = np.exp(output) / np.sum(np.exp(output))

    # 获取结果
    pred_idx = np.argmax(probs)
    name = class_names[pred_idx] if pred_idx < len(class_names) else f'类别{pred_idx}'

    return f"{name} ({probs[pred_idx]:.1%})"


# 创建界面
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="图像分类",
    description="上传图片识别类别"
).launch()
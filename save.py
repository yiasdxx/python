def save_model(network, filename):
    """保存模型（最简化版）"""
    import pickle

    # 从网络获取必要信息
    model_data = {
        'params': network.params,  # 所有参数
        'num_classes': network.params['W4'].shape[1],  # 从最后一层获取类别数
    }

    # 如果有BN层，收集BN参数
    if hasattr(network, 'layers'):
        bn_params = {}
        for name, layer in network.layers.items():
            if hasattr(layer, 'running_mean') and hasattr(layer, 'running_var'):
                bn_params[f'{name}_running_mean'] = layer.running_mean
                bn_params[f'{name}_running_var'] = layer.running_var
        if bn_params:
            model_data['bn_params'] = bn_params

    # 保存
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✅ 模型已保存到: {filename}")
    return filename
from micrograd import MLP, Value  # 从引擎导入 MLP 和 Value

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]  # 输入数据集
ys = [1.0, -1.0, -1.0, 1.0]    # 期望输出

# 建模型
model = MLP(3, [4, 4, 1])  # 输入维度 3，两层隐藏层各 4 个神经元，输出层 1 个神经元

# 训练循环
for k in range(20):     
    ypred = [model(x) for x in xs]
    loss = sum((ypred - yt)**2 for ypred, yt in zip(ypred, ys))   # ← loss 是 Value
    for p in model.parameters():     # 遍历所有参数
        p.grad = 0.0                 # 清零上一轮的梯度
    loss.backward()                  # 从 loss 出发反向传播，算出每个参数梯度
    learning_rate = 0.05
    for p in model.parameters():     # 遍历所有参数
        p.data -= learning_rate * p.grad   # 往让 loss 变小的方向挪一小步        # 把这一轮的 loss 记下来
    print(k, loss.data)   

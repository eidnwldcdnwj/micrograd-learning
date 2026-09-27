from micrograd import MLP, Value  # 从引擎导入 MLP 和 Value

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]  # 输入数据集
ys = [1.0, -1.0, -1.0, 1.0]    # 期望输出

# 建模型
model = MLP(3, [4, 4, 1])  # 输入维度 3，两层隐藏层各 4 个神经元，输出层 1 个神经元

# 损失函数：平方误差
def loss_fn():
    ypred = [model(x) for x in xs]          # 前向算出每个预测值
    # 用 (p - y) * (p - y) 代替 (p - y) ** 2（Value 没实现 ** 运算符）
    return sum(((p - y) * (p - y) for p, y in zip(ypred, ys)), Value(0.0))

# 训练循环
for k in range(20):                  # 训练 20 轮
    for p in model.parameters():     # 遍历所有参数
        p.grad = 0.0                 # 清零上一轮的梯度（很重要！）
    loss = loss_fn()                 # 算出这一轮的 loss（是一个 Value 对象）
    loss.backward()                  # 从 loss 出发反向传播，算出每个参数梯度
    learning_rate = 0.05
    for p in model.parameters():     # 遍历所有参数
        p.data -= learning_rate * p.grad   # 往让 loss 变小的方向挪一小步
    print(k, round(loss.data, 4))    # 打印轮次和当前 loss

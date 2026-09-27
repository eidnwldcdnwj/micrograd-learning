# micrograd-learning

 Karpathy 的《Neural Networks: Zero to Hero》

中秋快乐！

## 文件

- `micrograd.py`：核心。Value 类（自动求导）+ Neuron / Layer / MLP。


## 实现到的东西

- `Value`：`+ - * / tanh exp **`，每次运算自动建计算图，记录前驱节点和运算类型
- `backward()`：DFS 拓扑排序，逆序逐节点回传梯度
- `Neuron` / `Layer` / `MLP`：`__call__` 统一前向接口，`parameters()` 把全部参数摊平成一个列表给训练循环用

## tips

- `_backward` 里梯度是用 `+=` 累加的，所以每轮训练开头必须先把所有 `grad` 清零，不然会叠加上一轮的旧值，loss 会乱跳。


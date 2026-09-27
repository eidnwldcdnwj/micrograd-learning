# micrograd-learning

从零手写 micrograd（Karpathy《Neural Networks: Zero to Hero》教学项目的复现）：
一个极简的标量级自动求导引擎 + 神经网络库

## 文件结构

```
micrograd-learning/
├── micrograd.py   # 核心：Value（自动Neuron / Layer / MLP
├── README.md
└── .gitignore
```

## 已实现

- **Value**：`+` `-` `*` `/` `tanh` `exp`，运算符重载自动构建计算图，
  每个节点记录 `_prev`（前驱）与 `_op`（运算类型）
- **backward()**：DFS 构建拓扑排序 → 逆序逐节点执行 `_backward` 闭包回传梯度
- **Neuron / Layer / MLP**：`__call__` 统一前向接口，`parameters()` 把全部参数
  摊平成一个列表供训练循环更新

## 验证

`MLP(3, [4,4,1])`（41 个参数）前向 + 反向实测：

- 41/41 个参数全部拿到非零梯度
- 解析梯度与数值梯度（中心差分法）一致到小数点后 8 位

## TODO

- [ ] 训练循环（loss → backward → 梯度下降更新参数）
- [ ] 更多运算的梯度测试

import math
import random
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data=data #存储数值
        self.grad=0 #存储1梯度 默认0
        self._backward= lambda: None #反向传播函数，默认空操作
        self._prev = set(_children) # 前驱节点（构建计算图连边）
        self._op = _op  #产生该节点的运算

    def __repr__(self):
        return f"Value(data={self.data},grad={self.grad})"
#运算符重载，每做一次运算，就产生一个新的Value节点，同时记录父节点和运算类型
    def __add__ (self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data *other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward =_backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad #tanh导数 = 1-tanh的2(x)
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self,),'exp')
        def _backward():
            self.grad += out.data * out.grad #ex导数为ex
        out._backward = _backward
        return out

    def __truediv__(self,other):  #真除 a/b self/other
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data / other.data,(self,other),'/')
        def _backward():
            self.grad +=(1.0/other.data)*out.grad    #对a求偏导为1/b，局部导数乘上游梯度
            other.grad += (-self.data/other.data**2)*out.grad #对b求偏导为-a/b方
        out._backward =_backward #把这份"说明书"挂到 out 节点上
        return out  #返回新节点

    def __neg__(self): #neg取负运算
        return self * -1
    def __sub__(self,other):
        return self + (-other)

    
    def backward(self):  #用DFS构建拓扑排序
        topo=[]
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        #初始化输出节点梯度为1
        self.grad = 1.0
        #逆序遍历，逐个执行_backward
        for v in reversed(topo):
            v._backward()
class Neuron:#单个神经元
    def __init__(self,nin):  #nin输入维度，init造一个还没学习过的神经元
        self.w = [Value(random.uniform(-1,1))for _ in range(nin)]#random.uniform(-1, 1) 在 [-1, 1) 随机取数，并用 Value 包起来
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x): #让n（x）生效，规定了 "当有人对这个对象加括号调用时，做什么"
        #算出 Σwᵢxᵢ + b（蓝色路径），过 tanh 得到输出。让实例能像函数一样调用：n(x)
        act = sum((wi  * xi for wi, xi in zip(self.w,x)),self.b) #zip(self.w, x)：把两个列表按位置配对 → [(w₁,x₁), (w₂,x₂)]
        out = act.tanh()
        return out
    def parameters(self):           #parameters() = "把我需要被训练的所有东西列个清单交给你"，训练循环拿这个清单来清零梯度、更新参数。
        return self.w + [self.b] #列表拼接，加方括号把b变成列表
#在 nout 个 Neuron 并排，同一份输入 x 同时喂给每个神经元，每个神经元各算各的、各出一个输出，所以输出是 nout 个数。
class Layer:    #Layer(3, 4) → 4 个神经元，每个都有 3 个权重 + 1 个偏置，共 4×4=16 
    def __init__(self,nin,nout):   # nin = 输入维度,每个神经元内部会有 nin 个权重 nout = 该层神经元个数，层的输出有几个数
        self.neurons = [Neuron(nin)for _ in range(nout)]  #重复执行 nout 次，每次新建一个 Neuron(nin)，装进列表。
    def __call__(self, x):
        outs = [n(x)for n in self.neurons] #对 self.neurons里的每一个n，算出n(x)，把结果装进列表
        return outs[0] if len(outs) == 1 else outs # 单输出时直接返回标量outs0，否则返回整个列表
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()] #先读外层的for对self.neurons里的每个neuron，再对它 parameters() 里的每个 p，把 p 收集起来
class MLP:  #多层感知机
    def __init__(self,nin,nouts):
        # nin = 输入维度, nouts = 各层输出维度列表
        # 例如 MLP(3, [4, 4, 1]) 表示：3输入 → 4隐藏 → 4隐藏 → 1输出 nouts为[4,4,1]
        sz = [nin] + nouts      #sz size 一个临时列表，把整个网络 "从输入到输出每一级的节点数" 串在一起[3,4,4,1] 
        self.layers = [Layer(sz[i],sz[i+1])for i in range (len(nouts))] #len(nouts) = 3，所以 i 依次是 0、1、2，每次取 sz[i] 和 sz[i+1]→ Layer(3,4)、Layer(4,4)、Layer(4,1)
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
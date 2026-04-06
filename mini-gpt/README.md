[中文](#中文) | [English](#english)

---

# 中文

## Mini-GPT 学习笔记

从零实现 GPT，不用 PyTorch，手写 autograd + numpy。

### 进度

- [x] Tensor 类 + 计算图
- [x] `add`, `sub`, `mul`, `pow` 前向 + 反向传播
- [x] `backward()` 拓扑排序
- [x] 线性回归跑通
- [x] `relu`, `tanh`, `gelu`
- [x] `sum`
- [x] `matmul`
- [x] 两层神经网络跑通
- [ ] `softmax`
- [ ] Linear 层
- [ ] Embedding 层
- [ ] Attention
- [ ] LayerNorm
- [ ] GPT 组装 + 训练

### 学到的概念

#### 计算图

每个 `Tensor` 记录它的父节点（`parents`），构成一个有向无环图（DAG）。
前向传播正常计算结果，反向传播从 loss 出发沿图反向把梯度一层层传回去。

#### 链式法则

```
d(loss)/d(w) = d(loss)/d(loss) × d(loss)/d(y_pred) × d(y_pred)/d(w)
```

每个 `_backward` 函数只负责把 `out.grad` 乘以本地导数，再传给父节点。

#### 为什么 `self.grad = 1`

`backward()` 从 loss 节点出发。loss 对自己的导数是 1（任何变量对自己求导都是 1）。
如果不设这个初始值，所有梯度都是 0，什么都传不下去。

#### 拓扑排序

必须按照"先处理 loss，再处理中间节点，最后处理参数"的顺序执行 `_backward`。
拓扑排序保证了这个顺序，`reversed()` 让我们从 loss 反向遍历。

#### matmul 的反向传播

设 `C = A @ B`，形状：`(m,k) @ (k,n) = (m,n)`

```
dL/dA = dL/dC @ B.T    → (m,n) @ (n,k) = (m,k) ✅
dL/dB = A.T @ dL/dC    → (k,m) @ (m,n) = (k,n) ✅
```

口诀：**谁在右边，对面就转置乘过来**。

为什么这样？矩阵乘法的链式法则要求梯度的形状必须和原矩阵一致，转置是唯一能让维度对齐的方式。

#### 广播与 `_unbroadcast`

加法时 bias 会发生广播：
```
X @ W1 的形状：(4, 4)
b1 的形状：    (1, 4)   ← numpy 自动广播成 (4,4) 参与运算
```

反向传播时 `out.grad` 形状是 `(4,4)`，但 `b1.grad` 必须是 `(1,4)`。
所以要把梯度沿广播的维度 **求和** 压回原来的形状：

```
(4,4) --sum axis=0--> (1,4)
```

`_unbroadcast` 做的就是这件事：
1. 如果 grad 维度比原 shape 多，先 sum 掉多出的维度
2. 如果某个维度原来是 1，sum 掉那个维度并保持 `keepdims=True`

#### GELU

GPT 用的激活函数，比 relu 更平滑，负数区域有小的非零输出：

```
GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

不需要新的 `_backward`，用已有的 `tanh`, `mul`, `add`, `pow` 拼出来，autograd 自动处理反向传播。

### 犯的错误

#### 1. `_backward` 里梯度传播方向写反了

```python
# ❌ 错误
def _backward():
    self.grad += other.grad
    other.grad += self.grad

# ✅ 正确
def _backward():
    self.grad += out.grad
    other.grad += out.grad
```

`_backward` 的职责是把 `out.grad` 传给父节点，不是节点之间互传。

#### 2. 忘记给 `out.parents` 赋值

没有 `parents`，拓扑排序只能遍历到 loss 一个节点，梯度根本传不出去。

#### 3. 忘记 `self.grad = 1`

loss 的初始梯度是 0，所有下游梯度都是 0，参数完全不更新。

#### 4. 只用一个样本训练

`x=1.0` 时 `w` 和 `b` 的梯度完全相同，两个参数永远同步，无法收敛到正确值。需要多个不同的 x 才能让模型区分 `w` 和 `b`。

#### 5. 变量名和函数名冲突

```python
loss = loss(y, y_pred)  # ❌ loss 函数被变量覆盖了
L = loss_fn(y, y_pred)  # ✅
```

#### 6. Python 语法错误

```python
for step in 100:        # ❌
for step in range(100): # ✅
```

---

# English

## Mini-GPT Learning Notes

Building GPT from scratch — no PyTorch, handwritten autograd + numpy.

### Progress

- [x] Tensor class + compute graph
- [x] `add`, `sub`, `mul`, `pow` forward + backward
- [x] `backward()` topological sort
- [x] Linear regression working
- [x] `relu`, `tanh`, `gelu`
- [x] `sum`
- [x] `matmul`
- [x] Two-layer neural network working
- [ ] `softmax`
- [ ] Linear layer
- [ ] Embedding layer
- [ ] Attention
- [ ] LayerNorm
- [ ] GPT assembly + training

### Concepts Learned

#### Compute Graph

Each `Tensor` records its parent nodes (`parents`), forming a directed acyclic graph (DAG).
The forward pass computes results normally; the backward pass propagates gradients back through the graph starting from loss.

#### Chain Rule

```
d(loss)/d(w) = d(loss)/d(loss) × d(loss)/d(y_pred) × d(y_pred)/d(w)
```

Each `_backward` only multiplies `out.grad` by the local derivative, then passes it to parent nodes.

#### Why `self.grad = 1`

`backward()` starts from the loss node. The derivative of any variable w.r.t. itself is 1.
Without this initialization, all gradients are 0 and nothing propagates.

#### Topological Sort

`_backward` must run in reverse topological order — loss first, then intermediate nodes, then parameters.
Topological sort guarantees this order; `reversed()` lets us traverse from loss backward.

#### Matmul Backward Pass

Given `C = A @ B` with shapes `(m,k) @ (k,n) = (m,n)`:

```
dL/dA = dL/dC @ B.T    → (m,n) @ (n,k) = (m,k) ✅
dL/dB = A.T @ dL/dC    → (k,m) @ (m,n) = (k,n) ✅
```

Rule: **whichever matrix is on the right, transpose the other and multiply from the opposite side.**

The transpose is necessary to make the gradient shapes match the original matrix shapes.

#### Broadcasting and `_unbroadcast`

When adding a bias, numpy broadcasts automatically:
```
X @ W1 shape: (4, 4)
b1 shape:     (1, 4)   ← broadcast to (4,4) during forward pass
```

During backward, `out.grad` has shape `(4,4)` but `b1.grad` must be `(1,4)`.
So we **sum** the gradient along the broadcasted dimensions to compress it back:

```
(4,4) --sum axis=0--> (1,4)
```

`_unbroadcast` does this:
1. If grad has more dimensions than the original shape, sum away the extra dims
2. If a dimension was originally 1, sum along that axis with `keepdims=True`

#### GELU

The activation function used in GPT. Smoother than ReLU — has small non-zero output for negative inputs:

```
GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

No custom `_backward` needed — built by composing existing ops (`tanh`, `mul`, `add`, `pow`). Autograd handles the backward pass automatically.

### Mistakes Made

#### 1. Gradient direction was backwards in `_backward`

```python
# ❌ Wrong
def _backward():
    self.grad += other.grad
    other.grad += self.grad

# ✅ Correct
def _backward():
    self.grad += out.grad
    other.grad += out.grad
```

The job of `_backward` is to pass `out.grad` to parents, not swap gradients between siblings.

#### 2. Forgot to set `out.parents`

Without `out.parents`, topological sort only sees the loss node — gradients never propagate.

#### 3. Forgot `self.grad = 1`

Loss starts with `grad = 0`, so all downstream gradients are 0 and parameters never update.

#### 4. Training on only one sample

With only `x=1.0`, gradients for `w` and `b` are identical — they move in lockstep and can't converge to correct values. Multiple different x values are needed.

#### 5. Variable name shadowed function name

```python
loss = loss(y, y_pred)  # ❌ function overwritten by variable
L = loss_fn(y, y_pred)  # ✅
```

#### 6. Python syntax error

```python
for step in 100:        # ❌
for step in range(100): # ✅
```

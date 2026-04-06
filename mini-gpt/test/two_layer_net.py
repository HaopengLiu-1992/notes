import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from autograd import Tensor
import numpy as np

# 数据：y = x1 + x2（两个输入，一个输出）
X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0],
              [7.0, 8.0]])
y = np.array([[3.0], [7.0], [11.0], [15.0]])  # x1 + x2

# 参数初始化
np.random.seed(0)
W1 = Tensor(np.random.randn(2, 4) * 0.1)  # (2, 4)
b1 = Tensor(np.zeros((1, 4)))             # (1, 4)
W2 = Tensor(np.random.randn(4, 1) * 0.1)  # (4, 1)
b2 = Tensor(np.zeros((1, 1)))             # (1, 1)

lr = 0.001

for step in range(500):
    # 清零梯度
    W1.grad = 0; b1.grad = 0
    W2.grad = 0; b2.grad = 0

    # 前向传播
    Xt = Tensor(X, requires_grad=False)
    yt = Tensor(y, requires_grad=False)

    h = (Xt @ W1 + b1).relu()   # (4, 4)
    pred = h @ W2 + b2           # (4, 1)

    diff = pred - yt
    loss = (diff * diff).sum()

    # 反向传播
    loss.backward()

    # 梯度下降
    W1.data -= lr * W1.grad
    b1.data -= lr * b1.grad
    W2.data -= lr * W2.grad
    b2.data -= lr * b2.grad

    if step % 100 == 0:
        print(f"step {step}: loss={loss.data:.4f}")

print(f"\nfinal loss: {loss.data:.4f}")
print(f"pred: {pred.data.ravel()}")
print(f"true: {y.ravel()}")

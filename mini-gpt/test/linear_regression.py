import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from autograd import Tensor

# y = 2x + 1
xs = [1.0, 2.0, 3.0, 4.0]
ys = [3.0, 5.0, 7.0, 9.0]

w = Tensor(1.0)
b = Tensor(1.0)

for step in range(1000):
    w.grad = 0
    b.grad = 0

    for xi, yi in zip(xs, ys):
        x = Tensor(xi, requires_grad=False)
        y_pred = w * x + b
        diff = y_pred - Tensor(yi, requires_grad=False)
        L = diff ** 2
        L.backward()

    w.data -= 0.01 * w.grad
    b.data -= 0.01 * b.grad

print(f"w={w.data:.4f}, b={b.data:.4f}")  # 期望 w≈2, b≈1

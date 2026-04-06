import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from autograd import Tensor

def test_add():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a + b
    c.backward()
    assert c.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0
    print("test_add OK")

def test_sub():
    a = Tensor(5.0)
    b = Tensor(3.0)
    c = a - b
    c.backward()
    assert c.data == 2.0
    assert a.grad == 1.0
    assert b.grad == -1.0
    print("test_sub OK")

def test_mul():
    a = Tensor(3.0)
    b = Tensor(4.0)
    c = a * b
    c.backward()
    assert c.data == 12.0
    assert a.grad == 4.0
    assert b.grad == 3.0
    print("test_mul OK")

def test_pow():
    x = Tensor(3.0)
    y = x ** 2
    y.backward()
    assert x.grad == 6.0
    print("test_pow OK")

def test_chain():
    # y = (x + 2) * x = x^2 + 2x, dy/dx = 2x + 2 = 8 when x=3
    x = Tensor(3.0)
    y = (x + Tensor(2.0)) * x
    y.backward()
    assert x.grad == 8.0
    print("test_chain OK")

def test_matmul():
    import numpy as np
    # A: (2,3), B: (3,2), out: (2,2)
    A = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    B = Tensor(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    C = A @ B
    loss = C.sum()
    loss.backward()
    print("test_matmul A.grad:", A.grad)
    print("test_matmul B.grad:", B.grad)
    print("test_matmul OK")

def test_relu():
    x = Tensor(3.0)
    y = x.relu()
    y.backward()
    assert x.grad == 1.0
    print("test_relu OK")

def test_relu_negative():
    x = Tensor(-2.0)
    y = x.relu()
    y.backward()
    assert x.grad == 0.0
    print("test_relu_negative OK")

def test_tanh():
    x = Tensor(0.0)
    y = x.tanh()
    y.backward()
    assert y.data == 0.0
    assert x.grad == 1.0
    print("test_tanh OK")

def test_gelu():
    import numpy as np
    x = Tensor(0.0)
    y = x.gelu()
    y.backward()
    assert abs(y.data) < 1e-6       # gelu(0) = 0
    assert abs(x.grad - 0.5) < 1e-4 # gelu'(0) ≈ 0.5
    print("test_gelu OK")

if __name__ == "__main__":
    test_add()
    test_sub()
    test_mul()
    test_pow()
    test_chain()
    test_matmul()
    test_relu()
    test_relu_negative()
    test_tanh()
    test_gelu()

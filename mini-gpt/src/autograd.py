import numpy as np
import math

class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self.parents = set()
        self._backward = lambda: None

    def __add__(self, other):
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        out.parents.add(self)
        out.parents.add(other)
        return out
    
    def __sub__(self, other):
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        out.parents.add(self)
        out.parents.add(other)
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad* self.data
        out._backward = _backward
        out.parents.add(self)
        out.parents.add(other)
        return out
    
    def __pow__(self, n):
        out = Tensor(self.data ** n, requires_grad=self.requires_grad)
        def _backward():
            self.grad += out.grad * n * (self.data ** (n - 1))
        out._backward = _backward
        out.parents.add(self)
        return out
    
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        out.parents.add(self)
        out.parents.add(other)
        return out
    
    def sum(self):
        out = Tensor(np.sum(self.data))
        def _backward():
            self.grad += out.grad * np.ones_like(self.data)
        out._backward = _backward
        out.parents.add(self)
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        def _backward():
            self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        out.parents.add(self)
        return out
    
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad)
        def _backward():
            self.grad += out.grad * (1 - t**2)
        out._backward = _backward
        out.parents.add(self)
        return out
    
    def gelu(self):
        k = Tensor(math.sqrt(2 / math.pi), requires_grad=False)
        c = Tensor(0.044715, requires_grad=False)
        inner = (k * (self + c * self ** 3)).tanh()
        return Tensor(0.5, requires_grad=False) * self * (Tensor(1.0, requires_grad=False) + inner)

    def backward(self):
        self.grad = 1
        visited = set()
        topo = []
        def build(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                build(parent)
            topo.append(node)
        build(self)
        for node in reversed(topo):
            node._backward()

def _unbroadcast(grad, shape):
    grad = np.array(grad)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
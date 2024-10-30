import numpy as np
import torch

import psfit as pf
from psfit import get_backend
from psfit.backend.cpu import Numpy

m, n = 10, 5


def test_tensor_addition_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    x_np = backend.randn(m, n)
    y_np = backend.randn(m, n)

    x_pt = torch.tensor(x_np, requires_grad = True)
    y_pt = torch.tensor(y_np, requires_grad = True)

    x_pf = pf.Tensor(x_np, required_grad = True)
    y_pf = pf.Tensor(y_np, required_grad = True)

    z_pt = x_pt + y_pt
    z_pf = x_pf + y_pf

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())
    backend.assert_close(y_pt.grad, y_pf.gradient.array())


def test_tensor_scalar_addition_backward():
    backend = get_backend()

    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    a = 2.0
    x_np = backend.randn(m, n)

    x_pt = torch.tensor(x_np, requires_grad = True)

    x_pf = pf.Tensor(x_np, required_grad = True)

    z_pt = x_pt + a
    z_pf = x_pf + a

    z_pf.backward()
    ones = torch.ones(z_pt.shape, device = device)
    z_pt.backward(ones)

    backend.assert_close(x_pt.grad, x_pf.gradient.array())


def test_tensor_subtraction_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"

    x_np = backend.randn(m, n)
    y_np = backend.randn(m, n)

    x_pt = torch.tensor(x_np, requires_grad = True)
    y_pt = torch.tensor(y_np, requires_grad = True)

    x_pf = pf.Tensor(x_np, required_grad = True)
    y_pf = pf.Tensor(y_np, required_grad = True)

    z_pt = x_pt - y_pt
    z_pf = x_pf - y_pf

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())
    backend.assert_close(y_pt.grad, y_pf.gradient.array())


def test_tensor_scalar_subtraction_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    a = 2.0
    x_np = backend.randn(m, n)

    x_pt = torch.tensor(x_np, requires_grad = True)

    x_pf = pf.Tensor(x_np, required_grad = True)

    z_pt = x_pt - a
    z_pf = x_pf - a

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())


def test_tensor_elementwise_multiplication_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    x_np = backend.randn(m, n)
    y_np = backend.randn(m, n)

    x_pt = torch.tensor(x_np, requires_grad = True)
    y_pt = torch.tensor(y_np, requires_grad = True)

    x_pf = pf.Tensor(x_np, required_grad = True)
    y_pf = pf.Tensor(y_np, required_grad = True)

    z_pt = x_pt * y_pt
    z_pf = x_pf * y_pf

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())
    backend.assert_close(y_pt.grad, y_pf.gradient.array())


def test_tensor_scalar_multiplication_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    a = 2.0
    x_np = backend.randn(m, n)

    x_pt = torch.tensor(x_np, requires_grad = True)

    x_pf = pf.Tensor(x_np, required_grad = True)

    z_pt = x_pt * a
    z_pf = x_pf * a

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())


def test_tensor_power_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"

    x_np = backend.randn(m, n)
    y_np = backend.ones(m, n)

    x_pt = torch.tensor(x_np, requires_grad = True)
    y_pt = torch.tensor(y_np, requires_grad = False)

    x_pf = pf.Tensor(x_np, required_grad = True)
    y_pf = pf.Tensor(y_np, required_grad = False)

    z_pt = x_pt ** y_pt
    z_pf = x_pf ** y_pf

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())


def test_tensor_scalar_power_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    a = 2.0
    x_np = backend.randn(m, n)

    x_pt = torch.tensor(x_np, requires_grad = True)

    x_pf = pf.Tensor(x_np, required_grad = True)

    z_pt = x_pt ** a
    z_pf = x_pf ** a

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())


def test_tensor_transpose_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"

    x_np = backend.randn(m, n)

    x_pt = torch.tensor(x_np, requires_grad = True)

    x_pf = pf.Tensor(x_np, required_grad = True)

    z_pt = x_pt.T
    z_pf = x_pf.T

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())


def test_matrix_multiplication_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    x_np = backend.randn(m, n)
    y_np = backend.ones(n, m)

    x_pt = torch.tensor(x_np, requires_grad = True)
    y_pt = torch.tensor(y_np, requires_grad = True)

    x_pf = pf.Tensor(x_np, required_grad = True)
    y_pf = pf.Tensor(y_np, required_grad = True)

    z_pt = x_pt @ y_pt
    z_pf = x_pf @ y_pf

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())
    backend.assert_close(y_pt.grad, y_pf.gradient.array())


def test_logarithm_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    x_np = backend.rand(m, n)

    x_pt = torch.tensor(x_np, requires_grad = True)

    x_pf = pf.Tensor(x_np, required_grad = True)

    z_pt = torch.log(x_pt)
    z_pf = pf.log(x_pf)

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())


def test_exp_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    x_np = backend.rand(m, n)

    x_pt = torch.tensor(x_np, requires_grad = True)

    x_pf = pf.Tensor(x_np, required_grad = True)

    z_pt = torch.exp(x_pt)
    z_pf = pf.exp(x_pf)

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())


def test_sigmoid_backward():
    backend = get_backend()
    x_np = backend.rand(m, n)
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    x_pf = pf.Tensor(x_np)
    x_pt = torch.tensor(x_np, requires_grad = True)

    z_pf = 1 / (1 + pf.exp(-x_pf))
    z_pt = 1 / (1 + torch.exp(-x_pt))

    z_pf.backward()
    z_pt.backward(torch.ones(z_pt.shape, device = device))

    backend.assert_close(x_pt.grad, x_pf.gradient.array())

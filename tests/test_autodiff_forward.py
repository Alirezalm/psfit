import numpy as np
import torch

import psfit as pf
from psfit import get_backend
from psfit.backend.cpu import Numpy
from psfit.backend.gpu import PyTorch

m, n = 10, 5


def test_create_tensor_from_raw_data():
    backend = get_backend()
    data = [[1.], [2.], [3.]]
    tensor = pf.Tensor(data)

    if isinstance(backend, PyTorch):
        backend.assert_close(torch.tensor(data, device = "cuda"), tensor.array())

    elif isinstance(backend, Numpy):
        backend.assert_close(np.array(data), tensor.array())
    else:
        raise TypeError("unknown backend")


def test_tensor_addition_forward():
    backend = get_backend()

    x_np = backend.randn(m, n)
    y_np = backend.randn(m, n)

    x = pf.Tensor(x_np)
    y = pf.Tensor(y_np)

    z = x + y

    backend.assert_close(z.array(), x_np + y_np)


def test_tensor_scalar_addition_forward():
    backend = get_backend()
    x_np = backend.randn(m, n)
    a = 1.0

    x = pf.Tensor(x_np)
    z = x + a

    backend.assert_close(z.array(), x_np + a)


def test_tensor_subtraction_forward():
    backend = get_backend()

    x_np = backend.randn(m, n)
    y_np = backend.randn(m, n)

    x = pf.Tensor(x_np)
    y = pf.Tensor(y_np)

    z = x - y

    backend.assert_close(z.array(), x_np - y_np)


def test_tensor_scalar_subtraction_forward():
    backend = get_backend()
    x_np = backend.randn(m, n)
    a = 1.0

    x = pf.Tensor(x_np)
    z = x - a

    backend.assert_close(z.array(), x_np - a)


def test_tensor_elementwise_multiplication_forward():
    backend = get_backend()

    x_np = backend.randn(m, n)
    y_np = backend.randn(m, n)

    x = pf.Tensor(x_np)
    y = pf.Tensor(y_np)

    z = x * y

    backend.assert_close(z.array(), x_np * y_np)


def test_tensor_scalar_multiplication_forward():
    backend = get_backend()
    x_np = backend.randn(m, n)
    a = 3.0

    x = pf.Tensor(x_np)
    z = x * a

    backend.assert_close(z.array(), x_np * a)


def test_tensor_power_forward():
    backend = get_backend()

    x_np = backend.randn(m, n)
    y_np = backend.ones(m, n) + 1.0

    x = pf.Tensor(x_np)
    y = pf.Tensor(y_np)

    z = x ** y

    backend.assert_close(z.array(), x_np ** y_np)


def test_tensor_scalar_power_forward():
    backend = get_backend()
    x_np = backend.randn(m, n)
    a = 2.0

    x = pf.Tensor(x_np)
    z = x ** a

    backend.assert_close(z.array(), x_np ** a)


def test_tensor_transpose_forward():
    backend = get_backend()
    x_np = backend.randn(m, n)

    x = pf.Tensor(x_np)
    z = x.T

    backend.assert_close(z.array(), x_np.T)


def test_reshape_forward():
    backend = get_backend()
    x_np = backend.randn(m, n)

    x = pf.Tensor(x_np)
    z = x.reshape(m * n, 1)

    backend.assert_close(z.array(), x_np.reshape(m * n, 1))


def test_summation_axis_row_forward():
    backend = get_backend()
    x_np = backend.randn(m, n)

    x = pf.Tensor(x_np)
    z = pf.sum(x, axis = 0, keepdims = True)

    backend.assert_close(z.array(), backend.sum(x_np, axis = 0, keepdims = True))


def test_summation_axis_cols_forward():
    backend = get_backend()
    x_np = backend.randn(m, n)

    x = pf.Tensor(x_np)
    z = pf.sum(x, axis = 1, keepdims = True)

    backend.assert_close(z.array(), backend.sum(x_np, axis = 1, keepdims = True))


def test_matmul_forward():
    backend = get_backend()

    x_np = backend.randn(m, n)
    y_np = backend.ones(n, m)

    x = pf.Tensor(x_np)
    y = pf.Tensor(y_np)

    z = x @ y

    backend.assert_close(z.array(), x_np @ y_np)


def test_tensor_scalar_division_right_forward():
    backend = get_backend()
    x_np = backend.rand(m, n)
    a = 10.0
    x = pf.Tensor(x_np)
    z = a / x

    backend.assert_close(z.array(), a / x_np)


def test_tensor_scalar_division_left_forward():
    backend = get_backend()
    x_np = backend.rand(m, n)
    a = 10.0
    x = pf.Tensor(x_np)
    z = x / 10

    backend.assert_close(z.array(), x_np / 10)


def test_tensor_division_forward():
    backend = get_backend()

    x_np = backend.randn(m, n)
    y_np = backend.ones(m, n)

    x = pf.Tensor(x_np)
    y = pf.Tensor(y_np)

    z = x / y

    backend.assert_close(z.array(), x_np / y_np)


def test_logarithm_forward():
    backend = get_backend()
    x_np = backend.rand(m, n)

    x = pf.Tensor(x_np)
    z = pf.log(x)

    backend.assert_close(z.array(), backend.log(x_np))


def test_exp_forward():
    backend = get_backend()
    x_np = backend.rand(m, n)

    x = pf.Tensor(x_np)
    z = pf.exp(x)

    backend.assert_close(z.array(), backend.exp(x_np))


def test_sigmoid_forward():
    backend = get_backend()
    x_np = backend.rand(m, n)

    x_pf = pf.Tensor(x_np)
    x_pt = torch.tensor(x_np)

    z_pf = 1.0 / (1.0 + pf.exp(-x_pf))
    z_pt = 1.0 / (1.0 + torch.exp(-x_pt))

    backend.assert_close(z_pf.array(), z_pt)

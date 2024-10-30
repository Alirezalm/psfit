import torch

import psfit as pf
from psfit import get_backend
from psfit.backend.cpu import Numpy

m, n, c = 100, 20, 10


def test_linear_forward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    data_type = "float32"

    if isinstance(backend, Numpy):
        # raw data
        ynp = backend.randint(0, c, (m, 1)).astype(data_type)
        Xnp = backend.randn(m, n).astype(data_type)
    else:
        ynp = backend.randint(0, c, (m, 1))
        Xnp = backend.randn(m, n)

    Xpf = pf.tensor(Xnp)

    model = pf.Linear(n, c)
    logits = model(Xpf)

    Xpt = torch.tensor(Xnp, device = device)
    wpt = torch.zeros(n, c, device = device)
    bpt = torch.zeros(1, c, device = device)

    zpt = Xpt @ wpt + bpt
    backend.assert_close(logits.array(), zpt)


def test_linear_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    data_type = "float32"

    if isinstance(backend, Numpy):
        # raw data
        Xnp = backend.randn(m, n).astype(data_type)
    else:
        Xnp = backend.randn(m, n)

    Xpf = pf.tensor(Xnp)

    model = pf.Linear(n, c)
    logits = model(Xpf)

    logits.backward()

    Xpt = torch.tensor(Xnp, device = device)
    wpt = torch.zeros(n, c, device = device, requires_grad = True)
    bpt = torch.zeros(1, c, device = device, requires_grad = True)

    zpt = Xpt @ wpt + bpt
    zpt.backward(torch.ones(*zpt.shape, device = device))

    backend.assert_close(wpt.grad, model.weights.gradient.array())
    backend.assert_close(bpt.grad, model.bias.gradient.array())


def test_softmax_loss_forward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    data_type = "float32"

    if isinstance(backend, Numpy):
        # raw data
        ynp = backend.randint(0, c, (m, 1))
        Xnp = backend.randn(m, n).astype(data_type)
    else:
        ynp = backend.randint(0, c, (m, 1))
        Xnp = backend.randn(m, n)

    Xpf = pf.tensor(Xnp)
    ypf = pf.tensor(ynp)

    model = pf.Linear(n, c)
    logits_pf = model(Xpf)

    Xpt = torch.tensor(Xnp, device = device)
    ypt = torch.tensor(ynp, device = device)
    wpt = torch.zeros(n, c, device = device)
    bpt = torch.zeros(1, c, device = device)

    logits_pt = Xpt @ wpt + bpt

    loss_pf = pf.module.SoftmaxLoss()
    loss_pt = torch.nn.CrossEntropyLoss()

    loss_value_pf = loss_pf(logits_pf, ypf)
    loss_value_pt = loss_pt(logits_pt, ypt.flatten())

    backend.assert_close(loss_value_pt, loss_value_pf.array().squeeze())


def test_softmax_loss_backward():
    backend = get_backend()
    device = "cpu" if isinstance(backend, Numpy) else "cuda"
    data_type = "float32"

    if isinstance(backend, Numpy):
        # raw data
        ynp = backend.randint(0, c, (m, 1))
        Xnp = backend.randn(m, n).astype(data_type)
    else:
        ynp = backend.randint(0, c, (m, 1))
        Xnp = backend.randn(m, n)

    Xpf = pf.tensor(Xnp)
    ypf = pf.tensor(ynp)

    model = pf.Linear(n, c)
    logits_pf = model(Xpf)

    Xpt = torch.tensor(Xnp, device = device)
    ypt = torch.tensor(ynp, device = device)
    wpt = torch.zeros(n, c, device = device, requires_grad = True)
    bpt = torch.zeros(1, c, device = device, requires_grad = True)

    logits_pt = Xpt @ wpt + bpt

    loss_pf = pf.module.SoftmaxLoss()
    loss_pt = torch.nn.CrossEntropyLoss()

    loss_value_pf = loss_pf(logits_pf, ypf)
    loss_value_pt = loss_pt(logits_pt, ypt.flatten())

    loss_value_pf.backward()
    loss_value_pt.backward()

    wpf, bpf = model.parameters

    backend.assert_close(bpf.gradient.array(), bpt.grad)
    backend.assert_close(wpf.gradient.array(), wpt.grad)

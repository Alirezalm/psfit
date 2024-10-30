from abc import ABC
from typing import List, Union, Tuple

import numpy as np
import torch

from .cpu import Backend, PTArray

# Mapping between string dtype and PyTorch dtype
torch_datatype = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int": torch.int32
}


class GPU(Backend, ABC):
    """
    Base class for GPU-based computations, inheriting from Backend.
    """
    pass

    @property
    def name(self):
        return self.__class__.__name__


class PyTorch(GPU):
    """
    PyTorch-based backend for GPU computations, inheriting from the GPU base class.

    Attributes:
        device: Specifies the device to be used, in this case 'cuda'.
    """
    device = "cuda"

    def make_array(self, data: List[Union[float, List[float]]], *, dtype: str = "float32") -> torch.Tensor:
        """
        Creates a PyTorch tensor from the input data and casts it to the specified dtype.

        :param data: List of floats or lists of floats to create a tensor.
        :param dtype: Data type of the tensor, defaults to 'float32'.
        :return: PyTorch tensor created from the input data.
        :raises KeyError: If the dtype provided is not supported.
        """
        if dtype not in torch_datatype:
            raise KeyError(f"Unsupported dtype: {dtype}. Supported types are: {list(torch_datatype.keys())}")
        return torch.tensor(data, dtype = torch_datatype[dtype], device = self.device)

    def sum(self, x: torch.Tensor, axis: int, *, keepdims: bool = True) -> torch.Tensor:
        """
        Sums the elements of the tensor along the specified axis.

        :param x: Input tensor.
        :param axis: Axis along which to sum.
        :param keepdims: Whether to retain reduced dimensions, defaults to True.
        :return: Tensor with summed values.
        """
        return torch.sum(x, dim = axis, keepdim = keepdims)

    def rand(self, m: int, n: int, *, dtype: str = "float32") -> torch.Tensor:
        """
        Generates a random tensor of shape (m, n) from a uniform distribution.

        :param m: Number of rows.
        :param n: Number of columns.
        :param dtype: Data type of the tensor, defaults to 'float32'.
        :return: Randomly generated tensor.
        :raises KeyError: If the dtype provided is not supported.
        """
        if dtype not in torch_datatype:
            raise KeyError(f"Unsupported dtype: {dtype}. Supported types are: {list(torch_datatype.keys())}")
        return torch.rand(m, n, device = self.device, dtype = torch_datatype[dtype])

    def randn(self, m: int, n: int, *, dtype: str = "float32") -> torch.Tensor:
        """
        Generates a random tensor of shape (m, n) from a standard normal distribution.

        :param m: Number of rows.
        :param n: Number of columns.
        :param dtype: Data type of the tensor, defaults to 'float32'.
        :return: Randomly generated tensor from standard normal distribution.
        :raises KeyError: If the dtype provided is not supported.
        """
        if dtype not in torch_datatype:
            raise KeyError(f"Unsupported dtype: {dtype}. Supported types are: {list(torch_datatype.keys())}")
        return torch.randn(m, n, device = self.device, dtype = torch_datatype[dtype])

    def ones(self, m: int, n: int, *, dtype: str = "float32") -> torch.Tensor:
        """
        Creates a tensor of ones with shape (m, n) and the specified dtype.

        :param m: Number of rows.
        :param n: Number of columns.
        :param dtype: Data type of the tensor, defaults to 'float32'.
        :return: Tensor filled with ones.
        :raises KeyError: If the dtype provided is not supported.
        """
        if dtype not in torch_datatype:
            raise KeyError(f"Unsupported dtype: {dtype}. Supported types are: {list(torch_datatype.keys())}")
        return torch.ones(m, n, device = self.device, dtype = torch_datatype[dtype])

    def zeros(self, m: int, n: int, *, dtype: str = "float32") -> torch.Tensor:
        """
        Creates a tensor of zeros with shape (m, n) and the specified dtype.

        :param m: Number of rows.
        :param n: Number of columns.
        :param dtype: Data type of the tensor, defaults to 'float32'.
        :return: Tensor filled with zeros.
        :raises KeyError: If the dtype provided is not supported.
        """
        if dtype not in torch_datatype:
            raise KeyError(f"Unsupported dtype: {dtype}. Supported types are: {list(torch_datatype.keys())}")
        return torch.zeros(m, n, device = self.device, dtype = torch_datatype[dtype])

    def eye(self, n: int, *, dtype: str = "float32") -> torch.Tensor:
        """
        Creates an identity matrix of size (n, n) with the specified dtype.

        :param n: Size of the identity matrix.
        :param dtype: Data type of the tensor, defaults to 'float32'.
        :return: Identity matrix tensor.
        :raises KeyError: If the dtype provided is not supported.
        """
        if dtype not in torch_datatype:
            raise KeyError(f"Unsupported dtype: {dtype}. Supported types are: {list(torch_datatype.keys())}")
        return torch.eye(n, device = self.device, dtype = torch_datatype[dtype])

    def get_dim(self, x: PTArray) -> int:
        """
        Returns the number of dimensions of the input tensor.

        :param x: Input tensor.
        :return: Number of dimensions of the tensor.
        """
        return x.dim()

    def log(self, x: PTArray) -> torch.Tensor:
        """
        Computes the natural logarithm of the input tensor.

        :param x: Input tensor.
        :return: Tensor with the natural logarithm of the elements.
        """
        return torch.log(x)

    def exp(self, x: PTArray) -> torch.Tensor:
        """
        Computes the exponential of the elements in the input tensor.

        :param x: Input tensor.
        :return: Tensor with the exponential of the elements.
        """
        return torch.exp(x)

    def max(self, x: PTArray, axis: int = 0, *, keepdims: bool = True) -> torch.Tensor:
        """
        Returns the maximum values along the specified axis.

        :param x: Input tensor.
        :param axis: Axis along which to compute the maximum.
        :param keepdims: Whether to retain reduced dimensions, defaults to True.
        :return: Tensor with the maximum values.
        """
        return torch.max(x, dim = axis, keepdim = keepdims)[0]

    def randint(self, s: int, e: int, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generates a random tensor of integers between the specified range [s, e).

        :param s: Start of the range.
        :param e: End of the range.
        :param size: Shape of the output tensor.
        :return: Tensor of random integers.
        """
        return torch.randint(s, e, size = size, device = self.device)

    def random_choice(self, e: int, size: Union[int, Tuple[int, int]], *, replace: bool = False) -> torch.Tensor:
        """
        Generates an array of random choices from the range [0, e).

        :param e: Upper bound of the range.
        :param size: Shape of the output tensor.
        :param replace: Whether to sample with replacement, defaults to False.
        :return: Tensor with random choices.
        """
        return torch.tensor(np.random.choice(e, size = size, replace = replace), device = self.device)

    def dtype_to(self, data: PTArray, dtype: str) -> torch.Tensor:
        """
        Converts the data type of the tensor to the specified dtype.

        :param data: Input tensor.
        :param dtype: Desired data type ('float32', 'float64', or 'int').
        :return: Tensor converted to the specified dtype.
        :raises KeyError: If the dtype provided is not supported.
        """
        if dtype not in torch_datatype:
            raise KeyError(f"Unsupported dtype: {dtype}. Supported types are: {list(torch_datatype.keys())}")
        return data.to(torch_datatype[dtype])

    def keep_largest_k_element(self, v: PTArray, nnz: int) -> torch.Tensor:
        """
        Keeps the largest `nnz` elements in the tensor and sets the rest to zero.

        :param v: Input tensor.
        :param nnz: Number of largest elements to keep.
        :return: Tensor with only the largest `nnz` elements kept, the rest set to zero.
        """
        v_flat = v.view(-1)
        _, indices = torch.topk(torch.abs(v_flat), nnz)
        mask = torch.zeros_like(v_flat)
        mask[indices] = 1.0
        return (v_flat * mask).view(v.shape)

    def norm(self, v: PTArray, order: int = 2):
        return torch.norm(v, order)

    def abs(self, v: PTArray):
        return torch.abs(v)

    def __repr__(self) -> str:
        """
        Returns a string representation of the PyTorch backend.
        """
        return f"backend(dev={self.device}, api=pytorch)"

    @staticmethod
    def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
        """
        Asserts that two tensors are close within a tolerance.

        :param actual: Actual tensor.
        :param expected: Expected tensor.
        """
        torch.testing.assert_close(actual, expected)


def pytorch_backend() -> PyTorch:
    """
    Factory function to return a PyTorch backend instance.

    :return: PyTorch backend instance.
    """
    return PyTorch()

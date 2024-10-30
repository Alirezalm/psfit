from typing import Union

import numpy as np
import torch

NPArray = np.ndarray
PTArray = torch.Tensor

Array = Union[NPArray, PTArray]

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Tuple


class Backend(ABC):
    """
    Abstract backend class that defines the interface for various array and mathematical operations.
    """

    @abstractmethod
    def make_array(self, data: List[Union[float, List[float]]], *, dtype: str = "float32"):
        """
        Creates an array from the provided data with the specified dtype.

        :param data: List of floats or lists of floats to create an array.
        :type data: List[Union[float, List[float]]]
        :param dtype: Data type of the array, defaults to 'float32'.
        :type dtype: str
        :return: An array of the specified dtype.
        """
        pass

    @abstractmethod
    def sum(self, x, axis: int, *, keepdims: bool = True):
        """
        Sums the elements along the specified axis.

        :param x: Input array.
        :param axis: Axis along which the sum is computed.
        :param keepdims: Whether to retain reduced dimensions, defaults to True.
        """
        pass

    @abstractmethod
    def rand(self, m: int, n: int, *, dtype: str = "float32"):
        """
        Generates a random array of shape (m, n) with values from a uniform distribution over [0, 1).

        :param m: Number of rows.
        :param n: Number of columns.
        :param dtype: Data type of the array, defaults to 'float32'.
        :return: Randomly generated array.
        """
        pass

    @abstractmethod
    def randn(self, m: int, n: int, *, dtype: str = "float32"):
        """
        Generates a random array of shape (m, n) with values from a standard normal distribution.

        :param m: Number of rows.
        :param n: Number of columns.
        :param dtype: Data type of the array, defaults to 'float32'.
        :return: Randomly generated array.
        """
        pass

    @abstractmethod
    def ones(self, m: int, n: int, *, dtype: str = "float32"):
        """
        Creates an array of ones with shape (m, n) and the specified dtype.

        :param m: Number of rows.
        :param n: Number of columns.
        :param dtype: Data type of the array, defaults to 'float32'.
        :return: Array of ones.
        """
        pass

    @abstractmethod
    def zeros(self, m: int, n: int, *, dtype: str = "float32"):
        """
        Creates an array of zeros with shape (m, n) and the specified dtype.

        :param m: Number of rows.
        :param n: Number of columns.
        :param dtype: Data type of the array, defaults to 'float32'.
        :return: Array of zeros.
        """
        pass

    @abstractmethod
    def eye(self, n: int, *, dtype: str = "float32"):
        """
        Creates an identity matrix of size (n, n) with the specified dtype.

        :param n: Size of the identity matrix.
        :param dtype: Data type of the array, defaults to 'float32'.
        :return: Identity matrix.
        """
        pass

    @abstractmethod
    def get_dim(self, x):
        """
        Returns the number of dimensions of the input array.

        :param x: Input array.
        :return: Number of dimensions.
        """
        pass

    @abstractmethod
    def log(self, x):
        """
        Computes the natural logarithm of the elements of the input array.

        :param x: Input array.
        :return: Array with natural logarithms of the elements.
        """
        pass

    @abstractmethod
    def exp(self, x):
        """
        Computes the exponential of the elements of the input array.

        :param x: Input array.
        :return: Array with the exponential of the elements.
        """
        pass

    @abstractmethod
    def max(self, x, axis: int = 0, *, keepdims = True):
        """
        Returns the maximum values along the specified axis.

        :param x: Input array.
        :param axis: Axis along which to compute the maximum.
        :param keepdims: Whether to retain reduced dimensions, defaults to True.
        """
        pass

    @abstractmethod
    def randint(self, s: int, e: int, size: Tuple[int, int]):
        """
        Generates an array of random integers between the range [s, e).

        :param s: Start of the range.
        :param e: End of the range.
        :param size: Shape of the output array.
        :return: Array of random integers.
        """
        pass

    @abstractmethod
    def random_choice(self, e: int, size: Union[int, Tuple[int, int]], *, replace = False):
        """
        Generates an array of random choices from the range [0, e).

        :param e: Upper bound of the range.
        :param size: Shape of the output array.
        :param replace: Whether to sample with replacement, defaults to False.
        :return: Array of random choices.
        """
        pass

    @abstractmethod
    def dtype_to(self, data, dtype: str):
        """
        Converts the data type of the array to the specified dtype.

        :param data: Input array.
        :param dtype: Desired data type.
        :return: Array with the converted data type.
        """
        pass

    @abstractmethod
    def keep_largest_k_element(self, v, nnz: int):
        """
        Keeps the largest `nnz` elements in the input array and zeroes out the rest.

        :param v: Input array.
        :param nnz: Number of largest elements to keep.
        :return: Array with only the largest `nnz` elements kept.
        """
        pass

    @abstractmethod
    def norm(self, v: Array, order: int = 2):
        pass

    @abstractmethod
    def abs(self, v: Array):
        pass

    @staticmethod
    @abstractmethod
    def assert_close(actual, expected):
        """
        Asserts that two arrays are close within tolerance.

        :param actual: Actual array.
        :param expected: Expected array.
        """
        pass


class CPU(Backend):
    """
    Base class for CPU-based computations.
    """
    device = "cpu"

    @property
    def name(self):
        return self.__class__.__name__


class Numpy(CPU):
    """
    Numpy-based backend implementation of the CPU computations.
    """

    def make_array(self, data: List[Union[float, List[float]]], dtype: str = "float32") -> np.ndarray:
        """
        Creates a numpy array from the input data and casts it to the specified dtype.
        """
        try:
            return np.array(data, dtype = dtype)
        except Exception as e:
            raise ValueError(f"Error in creating array with dtype {dtype}: {str(e)}")

    def sum(self, x: np.ndarray, axis: int, *, keepdims: bool = True) -> np.ndarray:
        """
        Returns the sum of the elements along the specified axis.
        """
        return np.sum(x, axis = axis, keepdims = keepdims)

    def rand(self, m: int, n: int, *, dtype: str = "float32") -> np.ndarray:
        """
        Generates a random array of shape (m, n) from a uniform distribution.
        """
        return np.random.rand(m, n).astype(dtype)

    def randn(self, m: int, n: int, *, dtype: str = "float32") -> np.ndarray:
        """
        Generates a random array of shape (m, n) from a standard normal distribution.
        """
        return np.random.randn(m, n).astype(dtype)

    def ones(self, m: int, n: int, *, dtype: str = "float32") -> np.ndarray:
        """
        Creates an array of ones with shape (m, n) and the specified dtype.
        """
        return np.ones((m, n), dtype = dtype)

    def zeros(self, m: int, n: int, *, dtype: str = "float32") -> np.ndarray:
        """
        Creates an array of zeros with shape (m, n) and the specified dtype.
        """
        return np.zeros((m, n), dtype = dtype)

    def eye(self, n: int, *, dtype: str = "float32") -> np.ndarray:
        """
        Creates an identity matrix of size (n, n) with the specified dtype.
        """
        return np.eye(n, dtype = dtype)

    def get_dim(self, x: np.ndarray) -> int:
        """
        Returns the number of dimensions of the input array.
        """
        return x.ndim

    def log(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the natural logarithm of the input array.
        """
        return np.log(x)

    def exp(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the exponential of the input array.
        """
        return np.exp(x)

    def max(self, x: np.ndarray, axis: int = 0, *, keepdims: bool = True) -> np.ndarray:
        """
        Returns the maximum values along the specified axis.
        """
        return np.max(x, axis = axis, keepdims = keepdims)

    def randint(self, s: int, e: int, size: Tuple[int, int]) -> np.ndarray:
        """
        Generates a random array of integers between the specified range [s, e).
        """
        return np.random.randint(s, e, size)

    def random_choice(self, e: int, size: Union[int, Tuple[int, int]], *, replace: bool = False) -> np.ndarray:
        """
        Generates an array of random choices from the range [0, e).
        """
        return np.random.choice(e, size = size, replace = replace)

    def dtype_to(self, data: np.ndarray, dtype: str) -> np.ndarray:
        """
        Converts the input array to the specified dtype.
        """
        return data.astype(dtype)

    def keep_largest_k_element(self, v: np.ndarray, nnz: int) -> np.ndarray:
        """
        Keeps the largest `nnz` elements in the input array and zeroes out the rest.
        """
        v_flat = v.flatten()
        threshold = np.partition(np.abs(v_flat), -nnz)[-nnz]  # Get the threshold value
        mask = np.abs(v_flat) >= threshold  # Create a mask for the largest elements
        result = np.zeros_like(v_flat)
        result[mask] = v_flat[mask]
        return result.reshape(v.shape)

    def norm(self, v: NPArray, order: int = 2):
        return np.linalg.norm(v, order)

    def abs(self, v: NPArray):
        return np.abs(v)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Numpy backend.
        """
        return "default_backend(dev=cpu, api=numpy)"

    @staticmethod
    def assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
        """
        Asserts that two arrays are close within a tolerance.
        """
        np.testing.assert_allclose(actual, expected, rtol = 1.3e-6, atol = 1e-5)


def numpy_backend() -> Numpy:
    """
    Factory function to return a Numpy backend instance.

    :return: Numpy backend instance.
    """
    return Numpy()

from abc import ABC, abstractmethod
from typing import Tuple, Union

from psfit.backend import Array, get_backend
from psfit.exceptions import PsfitException

default_backend = get_backend()


class Operation(ABC):
    """
    An abstract base class for mathematical operations.

    This class defines the interface for implementing forward and backward passes
    of operations in a computation graph.

    Methods:
        forward(self, *args: Tuple[Array]) -> Array:
            Defines the forward computation logic.

        backward(self, adjoint: Union[Array, float], node):
            Defines the backward computation logic for gradient calculation.
    """

    @abstractmethod
    def forward(self, *args: Tuple[Array]):

        pass

    @abstractmethod
    def backward(self, adjoint: Union[Array, float], node):

        pass

    def __repr__(self):

        return f"psfit.{self.__class__.__name__}()"


class TensorOperation(Operation, ABC):


    def __call__(self, *args):
        from psfit.autodiff import Tensor  # Assuming Tensor is defined in autodiff module
        return Tensor.make_tensor_from_op(self, *args)


class TensorAddition(TensorOperation):
    """
    An operation that adds two tensors element-wise.

    """

    def forward(self, x: Array, y: Array):

        return x + y

    def backward(self, adjoint: Array, node):

        return adjoint, adjoint


class TensorScalarAddition(TensorOperation):
    """
    An operation that adds a scalar to a tensor.


    """

    def __init__(self, a: float):

        self.a = a

    def forward(self, x: Array):

        return x + self.a

    def backward(self, adjoint: Array, node):

        return adjoint,


class TensorElementWiseMultiplication(TensorOperation):
    """
    An operation that multiplies two tensors element-wise.

    """

    def forward(self, x: Array, y: Array):

        return x * y

    def backward(self, adjoint: Array, node):

        left_node, right_node = node.input_nodes
        return adjoint * right_node, adjoint * left_node


class TensorScalarMultiplication(TensorOperation):
    """
    An operation that multiplies a tensor by a scalar.

    """

    def __init__(self, a: float):

        self.a = a

    def forward(self, x: Array):

        return x * self.a

    def backward(self, adjoint: Array, node):

        return adjoint * self.a,  # Note that 'a' is not a tensor


class TensorElementWisePower(TensorOperation):
    """
    An operation that raises a tensor to the power of another tensor element-wise.

    """

    def forward(self, x: Array, y: Array):

        return x ** y

    def backward(self, adjoint: Array, node):

        l_node, r_node = node.input_nodes
        from .autodiff import ones, zeros
        return adjoint * r_node * l_node ** (r_node - ones(*r_node.shape)), zeros(*r_node.shape)


class TensorScalarPower(TensorOperation):
    """
    An operation that raises a tensor to the power of a scalar.

    """

    def __init__(self, a: float):

        self.a = a

    def forward(self, x: Array):

        return x ** self.a

    def backward(self, adjoint: Array, node):

        l_node = node.input_nodes[0]
        return adjoint * self.a * l_node ** (self.a - 1.0),


class Transpose(TensorOperation):
    """
    An operation that transposes a tensor.

    """

    def forward(self, x: Array):

        return x.T

    def backward(self, adjoint: Array, node):

        return adjoint.T,


class Reshape(TensorOperation):
    """
    An operation that reshapes a tensor.

    """

    def __init__(self, *axis):

        self.axis = axis

    def forward(self, x: Array):

        return x.reshape(*self.axis)

    def backward(self, adjoint: Array, node):

        raise NotImplementedError()


class Summation(TensorOperation):
    """
    An operation that computes the summation of a tensor along a specified axis.

    """

    def __init__(self, axis: int, keepdims: bool):

        self.axis = axis
        self.keepdims = keepdims
        self.backend = default_backend  # Assuming default_backend is defined elsewhere

    def forward(self, x: Array):

        return self.backend.sum(x, self.axis, keepdims = self.keepdims)

    def backward(self, adjoint: Array, node):

        from .autodiff import ones  # Assuming ones is defined in autodiff
        return adjoint * ones(*node.input_nodes[0].shape),


class MatrixMultiplication(TensorOperation):
    """
    An operation that performs matrix multiplication.

    """

    def forward(self, x: Array, y: Array):

        try:
            return x @ y
        except Exception as exc:
            raise PsfitException(str(exc))

    def backward(self, adjoint: Array, node):

        l_node, r_node = node.input_nodes
        return adjoint @ r_node.T, l_node.T @ adjoint


class Negate(TensorOperation):
    """
    An operation that negates a tensor.

    """

    def forward(self, x: Array):

        return -x

    def backward(self, adjoint: Array, node):

        return -adjoint,


class TensorElementWiseDivision(TensorOperation):
    """
    An operation that divides two tensors element-wise.

    """

    def forward(self, x: Array, y: Array):

        return x / y

    def backward(self, adjoint: Union[Array, float], node):

        raise NotImplementedError("Backward for tensor-tensor element-wise division not implemented yet.")


class TensorScalarDivision(TensorOperation):
    """
    An operation that divides a tensor by a scalar.

    """

    def __init__(self, a: float):

        self.a = a

    def forward(self, x: Array):

        return x / self.a

    def backward(self, adjoint: Union[Array, float], node):

        return adjoint / self.a,


class TensorScalarDivisionRight(TensorOperation):
    """
    An operation that divides a scalar by a tensor.

    """

    def __init__(self, a: float):

        self.a = a

    def forward(self, x: Array):

        return self.a / x

    def backward(self, adjoint: Union[Array, float], node):

        return -adjoint / self.a / node.input_nodes[0] ** 2,


class Exponential(TensorOperation):
    """
    An operation that computes the exponential of a tensor.

    """

    def forward(self, x: Array):

        return default_backend.exp(x)

    def backward(self, adjoint: Union[Array, float], node):

        return adjoint * exp(node.input_nodes[0]),


class Logarithm(TensorOperation):
    """
    An operation that computes the logarithm of a tensor.


    """

    def forward(self, x: Array):

        return default_backend.log(x)

    def backward(self, adjoint: Union[Array, float], node):

        return adjoint / node.input_nodes[0],


class LogSumExp(TensorOperation):
    """
    An operation that computes the log-sum-exp of a tensor.

    """

    def forward(self, logits: Array):

        # Assume logits rows are data and cols are classes so axis = 1
        # TODO: implement numerically stable version
        logits = default_backend.exp(logits)
        logits_sum = default_backend.sum(logits, axis = 1, keepdims = True)
        return default_backend.log(logits_sum)

    def backward(self, adjoint: Union[Array, float], node):

        logits = node.input_nodes[0]
        return adjoint * exp(logits - node),

def add_tensor(x, y):
    return TensorAddition()([x, y])


def add_tensor_scalar(x, a):
    return TensorScalarAddition(a)([x])


def multiply_tensor(x, y):
    return TensorElementWiseMultiplication()([x, y])


def multiply_tensor_scalar(x, a):
    return TensorScalarMultiplication(a)([x])


def power_tensor(x, y):
    return TensorElementWisePower()([x, y])


def power_tensor_scalar(x, a):
    return TensorScalarPower(a)([x])


def transpose(x):
    return Transpose()(x)


def reshape(x, *axis):
    return Reshape(axis)(x)


def summation(x, axis: int, *, keepdims: bool = True):
    tensor = Summation(axis, keepdims)([x])
    return tensor


def matmul(x, y):
    tensor = MatrixMultiplication()([x, y])
    return tensor


def negate(x):
    return Negate()([x])


def tensor_elementwise_division(x, y):
    return TensorElementWiseDivision()([x, y])


def tensor_scalar_division(x, a):
    return TensorScalarDivision(a)([x])


def tensor_scalar_division_right(x, a):
    return TensorScalarDivisionRight(a)([x])


def log(x):
    return Logarithm()([x])


def exp(x):
    return Exponential()([x])


def log_sum_exp(logits):
    return LogSumExp()([logits])

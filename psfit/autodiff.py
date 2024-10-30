from enum import Enum
from typing import List, Optional, Self, Dict, Set

from psfit.backend import Backend, PTArray, NPArray, CPU
from psfit.backend.cpu import Numpy
from psfit.backend.gpu import PyTorch
from psfit.ops import *

default_backend = get_backend()


class ComputationMode(Enum):
    LAZY = 1
    EAGER = 2


COMPUTATION_MODE = ComputationMode.EAGER

RAWDATA = Union[List[List[float]], List[float], NPArray, PTArray]


class Tensor:
    """Represents a tensor in the computational graph.

    Attributes:
        _grad (Optional[Self]): Gradient of the tensor.
        backend (Backend): Backend used for tensor operations.
        _cached_data (Optional[Array]): Cached data for the tensor.
        _inputs (List[Self]): List of input tensors.
        _operation (Optional[Operation]): Operation that computes the tensor.
        dtype (str): Data type of the tensor.
        required_grad (bool): Flag indicating if the tensor requires gradient computation.
    """

    _grad: Optional[Self] = None
    backend: Backend
    _cached_data: Optional[Array] = None
    _inputs: List[Self]  # list of inputs to the self tensor
    _operation: Optional[Operation]  # the operation that computes the self tensor
    dtype: str
    required_grad: bool

    def __init__(self, data: RAWDATA,
                 *,
                 dtype: str = "float32",
                 required_grad: bool = True):

        """Initializes a leaf tensor with the provided data.

        Args:
            data (RAWDATA): Data to initialize the tensor.
            dtype (str, optional): Data type of the tensor. Defaults to "float32".
            required_grad (bool, optional): Flag indicating if gradient computation is required. Defaults to True.
        """
        cached_data = Tensor._make_array_from_backend(backend = default_backend, data = data, dtype = dtype)

        self.init([], None, backend = default_backend, cached_data = cached_data, required_grad = required_grad)

    def init(self,
             inputs: List[Self],
             op: Optional[Operation],
             *,
             backend: Optional[Backend] = None,
             cached_data: Optional[Array] = None,
             required_grad: Optional[bool] = True):
        """Initializes a non-leaf tensor with the provided parameters.

        Args:
            inputs (List['Tensor']): List of input tensors.
            op (Optional[Operation]): Operation that computes this tensor.
            backend (Optional[Backend], optional): Backend to use. Defaults to None.
            cached_data (BackendDataType, optional): Cached data for the tensor. Defaults to None.
            required_grad (Optional[bool], optional): Flag indicating if gradient computation is required. Defaults to None.
        """

        self._cached_data = cached_data
        self._inputs = inputs
        self._operation = op
        self.backend = backend
        self.required_grad = required_grad

    @staticmethod
    def _unpack_data(data: RAWDATA):
        if isinstance(data, (NPArray, PTArray)):
            return data.tolist()

        if isinstance(data, (float, int)):
            return [float(data)]

        if isinstance(data, list):
            return data
        raise TypeError(f"tensor creation for type {type(data)} not supported.")

    @staticmethod
    def _make_array_from_backend(backend: Backend, data: RAWDATA, dtype: str = "float32"):

        raw_data = Tensor._unpack_data(data)

        if isinstance(backend, (Numpy, PyTorch)):
            return backend.make_array(data = raw_data, dtype = dtype)

        raise PsfitException(f"unknown backend.")

    def _populate_cached_data(self) -> Array:
        """Computes and returns the cached data for the tensor, if not already cached.

         Returns:
             Array: The cached data for the tensor.
         """
        if self._cached_data is not None:
            return self._cached_data

        self._cached_data = self._operation.forward(
            *[x._populate_cached_data() for x in self._inputs]
            # forward op is only applied on cached data. no more qrgs
        )  # computes the cache data recursively

        return self._cached_data

    @classmethod
    def make_tensor_from_op(cls, op: Operation, inputs: List[Self]):
        """Creates a tensor from an operation and a list of input tensors.

        Args:
            op (Operation): Operation that computes the tensor.
            inputs (List['Tensor']): List of input tensors.

        Returns:
            Tensor: The created tensor.
        """

        tensor = cls.__new__(cls)

        tensor.init(inputs, op, backend = default_backend)

        if COMPUTATION_MODE == ComputationMode.EAGER:

            if not tensor.required_grad:
                return tensor.detach()

            tensor._populate_cached_data()

            return tensor

        return tensor

    @classmethod
    def _make_detached_tensor(cls, graph: Self):
        """Creates a detached tensor with the same data as the provided tensor.

        Args:
            graph (Tensor): Tensor to detach.

        Returns:
            Tensor: The detached tensor.
        """
        tensor = cls.__new__(cls)

        tensor.init([],
                    None,
                    backend = graph.backend,
                    cached_data = graph._populate_cached_data(),
                    required_grad = False)
        return tensor

    @staticmethod
    def create_tensor_from_array(data: Array, requires_grad: bool) -> "Tensor":
        tensor = Tensor.__new__(Tensor)
        tensor.init([], None,
                    backend = default_backend,
                    cached_data = data,
                    required_grad = requires_grad)
        return tensor

    @property
    def input_nodes(self):
        return self._inputs

    @property
    def operation(self):
        return self._operation

    def detach(self):
        """detach tensor from the computational graph but keep its data"""
        return self._make_detached_tensor(self)

    def _get_default_adjoint(self, adjoint: None):
        if adjoint:
            return adjoint

        return 1.0 if self.dim == 1 else ones(*self.shape, requires_grad = False)

    def backward(self, adjoint = None):
        adjoint = self._get_default_adjoint(adjoint)
        compute_gradients(self, adjoint)

    def array(self):
        return self._populate_cached_data()

    @property
    def is_leaf(self):
        return self._operation is None

    @property
    def device(self):
        return "cpu" if isinstance(self.backend, CPU) else "gpu"

    @property
    def array_api(self):
        return self.backend

    @property
    def shape(self):
        return self._populate_cached_data().shape

    @property
    def dim(self):
        return self.backend.get_dim(self.array())

    @property
    def dtype(self):
        return self._populate_cached_data().dtype

    @property
    def gradient(self):
        if self._grad:
            return self._grad
        raise PsfitException("No computed gradient for this Tensor. make sure backward method is called first.")

    @gradient.setter
    def gradient(self, grad: Self):
        self._grad = grad

    def __add__(self, other: Union[Self, float, int]):
        if isinstance(other, Tensor):
            tensor = add_tensor(self, other)
            return tensor

        if isinstance(other, int):
            other = float(other)

        if isinstance(other, float):
            return add_tensor_scalar(self, other)
        else:
            raise PsfitException(f"tensor operation with {type(other)} is not supported.")

    def __mul__(self, other: Union[Self, float, int]):
        if isinstance(other, Tensor):
            tensor = multiply_tensor(self, other)
            return tensor

        if isinstance(other, (float, int)):
            return multiply_tensor_scalar(self, float(other))
        else:
            raise PsfitException(f"tensor operation with {type(other)} is not supported.")

    def __pow__(self, power: Union[Self, float, int]):
        if isinstance(power, Tensor):
            tensor = power_tensor(self, power)
            return tensor

        if self._is_supported_scalar(power):
            return power_tensor_scalar(self, float(power))

        else:
            raise PsfitException(f"tensor operation with {type(power)} is not supported.")

    def __matmul__(self, other: Self):
        tensor = matmul(self, other)
        return tensor

    def __neg__(self):
        tensor = negate(self)
        return tensor

    def __sub__(self, other: Union[Self, float, int]):
        if isinstance(other, Tensor):
            tensor = add_tensor(self, other.__neg__())
            return tensor

        if self._is_supported_scalar(other):
            return add_tensor_scalar(self, float(-other))

        else:
            raise PsfitException(f"tensor operation with {type(other)} is not supported.")

    @staticmethod
    def _is_supported_scalar(other: Union[int, float]):
        return isinstance(other, (float, int))

    def __truediv__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            tensor = tensor_elementwise_division(self, other)
            return tensor

        if self._is_supported_scalar(other):
            return tensor_scalar_division(self, float(other))

        else:
            raise PsfitException(f"tensor operation with {type(other)} is not supported.")

    def __rtruediv__(self, other: Union[Self, int, float]):
        if isinstance(other, Tensor):
            tensor = tensor_elementwise_division(self, other)
            return tensor

        if self._is_supported_scalar(other):
            return tensor_scalar_division_right(self, float(other))

        else:
            raise PsfitException(f"tensor operation with {type(other)} is not supported.")

    def __getitem__(self, index):
        try:
            data = self._populate_cached_data()[index]
            return Tensor.create_tensor_from_array(data = data, requires_grad = self.required_grad)

        except Exception as exp:
            raise PsfitException(str(exp))

    def __setitem__(self, key, value):
        # todo: check different types of value
        try:
            data = self._populate_cached_data()
            value = value.array().T
            data[key] = value
            return Tensor.create_tensor_from_array(data = data, requires_grad = self.required_grad)

        except Exception as exp:
            raise PsfitException(str(exp))

    @property
    def T(self):
        tensor = transpose([self])
        return tensor

    def reshape(self, *axis):
        tensor = reshape([self], *axis)
        return tensor

    def __repr__(self):
        return f"psfit.tensor({self._populate_cached_data()}, backend={self.backend.__class__.__name__.lower()})"

    def __str__(self):
        return self._populate_cached_data().__str__()

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


def compute_gradients(graph: Tensor, adjoint: Tensor):
    """
    Compute gradients for the nodes in the computational graph using backpropagation.

    Args:
        graph (Tensor): The output tensor of the graph for which gradients are computed.
        adjoint (Tensor): The gradient of the loss with respect to the output tensor.
    """
    # Initialize a mapping from each node to its corresponding gradients
    node_to_grad: Dict[Tensor, List[Array]] = {graph: [adjoint]}

    # Get the nodes in reverse topological order
    reversed_topo_order: List[Tensor] = list(reversed(topological_sort(graph)))

    # Iterate over each node in reverse topological order
    for node in reversed_topo_order:
        # Sum gradients for the current node from all pathways
        node_adjoint = sum(node_to_grad[node])  # for multi-pathways gradients

        # Assign the computed gradient to the node's gradient attribute
        node.gradient = node_adjoint

        # If the node is a leaf node, skip further processing
        if node.is_leaf:
            continue

        # Compute the partial adjoints for the inputs of the current node
        partial_adjoints = node.operation.backward(node_adjoint, node)

        # Distribute the partial adjoints to the input nodes
        for input_node, partial_adjoint in zip(node.input_nodes, partial_adjoints):
            # Initialize the input node's gradient list if it doesn't exist
            if not node_to_grad.get(input_node):
                node_to_grad[input_node] = []
            # Append the partial adjoint to the input node's gradient list
            node_to_grad[input_node].append(partial_adjoint)


def topological_sort(graph: Tensor) -> List[Tensor]:
    """
    Perform a topological sort on the computational graph.

    Args:
        graph (Tensor): The output tensor of the graph.

    Returns:
        List[Tensor]: A list of nodes in topological order.
    """
    topological_order_list = []  # To store the ordered nodes
    visited_nodes = set()  # To keep track of visited nodes

    # Start DFS from the output graph node
    for node in [graph]:
        dfs(node, visited_nodes, topological_order_list)

    return topological_order_list


def dfs(node: Tensor, visited: Set[Tensor], topo_order: List[Tensor]):
    """
    Depth-first search helper function for topological sorting.

    Args:
        node (Tensor): The current node being visited.
        visited (Set[Tensor]): Set of visited nodes to avoid cycles.
        topo_order (List[Tensor]): List to store the topological order.
    """
    # If the node has already been visited, return
    if node in visited:
        return

    # Recursively visit all input nodes
    for input_node in node.input_nodes:
        dfs(input_node, visited, topo_order)

    # Append the current node to the topological order
    topo_order.append(node)
    visited.add(node)  # Mark the current node as visited


def rand(m: int, n: int, *, requires_grad = False) -> Tensor:
    """
    Create a tensor filled with random values.

    Args:
        m (int): Number of rows.
        n (int): Number of columns.
        requires_grad (bool): Flag indicating if the tensor requires gradient tracking.

    Returns:
        Tensor: A tensor filled with random values.
    """
    data = default_backend.rand(m, n)
    return Tensor.create_tensor_from_array(data, requires_grad)


def randn(m: int, n: int, *, requires_grad = False) -> Tensor:
    """
    Create a tensor filled with random values from a normal distribution.

    Args:
        m (int): Number of rows.
        n (int): Number of columns.
        requires_grad (bool): Flag indicating if the tensor requires gradient tracking.

    Returns:
        Tensor: A tensor filled with random values from a normal distribution.
    """
    data = default_backend.randn(m, n)
    return Tensor.create_tensor_from_array(data, requires_grad)


def ones(m: int, n: int, *, requires_grad = False) -> Tensor:
    """
    Create a tensor filled with ones.

    Args:
        m (int): Number of rows.
        n (int): Number of columns.
        requires_grad (bool): Flag indicating if the tensor requires gradient tracking.

    Returns:
        Tensor: A tensor filled with ones.
    """
    data = default_backend.ones(m, n)
    return Tensor.create_tensor_from_array(data, requires_grad)


def zeros(m: int, n: int, *, requires_grad = False) -> Tensor:
    """
    Create a tensor filled with zeros.

    Args:
        m (int): Number of rows.
        n (int): Number of columns.
        requires_grad (bool): Flag indicating if the tensor requires gradient tracking.

    Returns:
        Tensor: A tensor filled with zeros.
    """
    data = default_backend.zeros(m, n)
    return Tensor.create_tensor_from_array(data, requires_grad)


def eye(n: int, *, requires_grad = False) -> Tensor:
    """
    Create an identity tensor.

    Args:
        n (int): Size of the identity matrix.
        requires_grad (bool): Flag indicating if the tensor requires gradient tracking.

    Returns:
        Tensor: An identity tensor of size n x n.
    """
    data = default_backend.eye(n)
    return Tensor.create_tensor_from_array(data, requires_grad)


def one_hot(y: Tensor, c: int, requires_grad = False) -> Tensor:
    """
    Convert class labels to one-hot encoded vectors.

    Args:
        y (Tensor): Tensor containing class labels.
        c (int): Number of classes.
        requires_grad (bool): Flag indicating if the tensor requires gradient tracking.

    Returns:
        Tensor: One-hot encoded tensor.
    """
    y = y.detach().array().reshape(1, -1)  # Detach and reshape labels
    classes = default_backend.dtype_to(data = y, dtype = "int")  # Convert to integer type

    m = classes.shape[1]  # Get the number of samples
    h = default_backend.zeros(m, c)  # Initialize one-hot tensor

    # Set the appropriate indices to 1
    h[range(m), classes] = 1.0

    return Tensor.create_tensor_from_array(h, requires_grad = requires_grad)


def Proximal_L0(y: Tensor, density: int) -> Tensor:
    """
    Keep only the largest 'density' elements from the tensor.

    Args:
        y (Tensor): Input tensor.
        density (int): Number of largest elements to keep.

    Returns:
        Tensor: A tensor containing the largest elements.
    """
    y = y.detach().array()  # Detach the tensor to avoid gradient tracking
    sparsified_y = default_backend.keep_largest_k_element(y, density)  # Keep largest elements

    return Tensor.create_tensor_from_array(sparsified_y, requires_grad = False)  # Return as a new tensor


def Proximal_L1(y: Tensor, penalty: float):
    y = y.detach().array()

    y[abs(y) < penalty] = 0.0
    y[y > penalty] = y[y > penalty] - penalty
    y[y < - penalty] = y[y < - penalty] + penalty

    return Tensor.create_tensor_from_array(y, requires_grad = False)


def Proximal_L2(y: Tensor, rho: float, l2_penalty: float):
    y = y.detach().array()

    y = (rho * y) / (2 * l2_penalty + rho)

    return Tensor.create_tensor_from_array(y, requires_grad = False)

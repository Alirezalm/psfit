from abc import ABC, abstractmethod
from typing import Optional, List, Union, Tuple

from psfit.autodiff import Tensor, zeros, one_hot, PsfitException
from psfit.ops import summation, log_sum_exp

Parameter = Tensor


class Module(ABC):
    """
    Abstract base class for all neural network modules. This class provides a
    callable interface and defines the `forward` method that must be implemented
    by derived classes.

    Methods:
        __call__(self, *args: Union[Tensor, List[Tensor]]) -> Tensor:
            Calls the forward method with the provided input arguments.

        forward(self, *args: List[Tensor]) -> Tensor:
            Defines the forward pass logic.
    """

    def __call__(self, *args: Union[Tensor, List[Tensor]]) -> Tensor:
        """
        Call the forward method with the given input tensors.

        Args:
            *args: Input tensors to the module.

        Returns:
            Tensor: The output tensor produced by the forward pass.
        """
        return self.forward(*args)

    @abstractmethod
    def forward(self, *args: List[Tensor]) -> Tensor:
        """
        Abstract method to define the forward pass logic.

        Args:
            *args: Input tensors for the forward pass.

        Returns:
            Tensor: The output tensor after the forward pass.
        """
        pass


class LinearModel(Module, ABC):
    """
    Abstract base class for linear models. It provides basic functionality to manage model parameters (weights)
    and gradients. Subclasses must implement methods to initialize parameters and define the model size.
    """
    weights: Optional[Parameter] = None

    @property
    def parameters(self) -> Parameter:
        """
        Returns the parameters (weights) of the linear model.

        :return: The weight tensor of the linear model.
        :rtype: Parameter
        """
        return self.weights

    @property
    @abstractmethod
    def size(self) -> Tuple[int, int]:
        """
        Abstract property to return the size of the model, typically (input_features, output_features).
        Subclasses must implement this to define their specific size.

        :return: A tuple representing the size of the model.
        :rtype: Tuple[int, int]
        """
        pass

    @property
    def gradients(self) -> Tensor:
        """
        Returns the gradients of the model's weights, if available.

        :return: The gradient tensor of the weights.
        :rtype: Tensor
        :raises PsfitException: If gradients are not available, i.e., backward() has not been called.
        """
        if self.weights is not None and self.weights.gradient is not None:
            return self.weights.gradient
        raise PsfitException("Gradients not available. Ensure backward() is called before accessing gradients.")

    @abstractmethod
    def init_params(self) -> None:
        """
        Abstract method to initialize the model's parameters. Subclasses must implement this method to initialize
        the model's weights.
        """
        pass


class Linear(LinearModel):
    """
    A fully connected (linear) layer with specified input and output features. Implements a forward pass, parameter
    initialization, and provides size and gradient information.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initializes the linear model with the specified input and output dimensions and initializes the weights.

        :param in_features: Number of input features.
        :type in_features: int
        :param out_features: Number of output features.
        :type out_features: int
        """
        self.in_features = in_features
        self.out_features = out_features
        self._size = (self.in_features, self.out_features)

        # Initialize model parameters (weights)
        self.init_params()

    def init_params(self) -> None:
        """
        Initializes the model's parameters (weights) as a zero tensor with gradient tracking enabled.
        """
        self.weights = zeros(*self.size, requires_grad=True)

    def forward(self, X: Tensor) -> Tensor:
        """
        Performs a forward pass through the linear model by multiplying the input with the model's weights.

        :param X: Input tensor with shape (batch_size, in_features).
        :type X: Tensor
        :return: Output tensor with shape (batch_size, out_features).
        :rtype: Tensor
        """
        return X @ self.weights

    @property
    def size(self) -> Tuple[int, int]:
        """
        Returns the size of the linear model, defined as (input_features, output_features).

        :return: A tuple representing the size of the model.
        :rtype: Tuple[int, int]
        """
        return self._size

    def __repr__(self) -> str:
        """
        Returns a string representation of the linear model.

        :return: A string describing the model with input and output features.
        :rtype: str
        """
        return f"LinearClassifier(in_features={self.in_features}, out_features={self.out_features})"

    def __str__(self) -> str:
        """
        Returns the same representation as __repr__().

        :return: String description of the model.
        :rtype: str
        """
        return self.__repr__()



class LossFunction(Module):
    """
    Abstract base class for loss functions. Inherits from the Module class.

    Subclasses must implement the `forward` method, which defines how the loss is computed
    given one or more input tensors.
    """

    @abstractmethod
    def forward(self, *args: List[Tensor]) -> Tensor:
        """
        Abstract method to compute the forward pass of the loss function.

        :param args: One or more tensors that are required to compute the loss. The specific number
                     and nature of arguments depend on the loss function being implemented.
        :type args: List[Tensor]
        :return: The computed loss as a Tensor.
        :rtype: Tensor
        """
        pass


class SoftmaxLoss(LossFunction):
    """
    Computes the softmax loss function, which is commonly used in multi-class classification problems.

    This class computes the loss based on the logits (raw model outputs) and true labels (y).
    """

    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        """
        Computes the forward pass of the softmax loss function.

        :param logits: The raw outputs of the model (logits), expected to have shape (batch_size, num_classes).
        :type logits: Tensor
        :param y: Ground truth labels, expected to have shape (batch_size,).
        :type y: Tensor
        :return: The computed softmax loss, averaged over the batch.
        :rtype: Tensor
        :raises ValueError: If the logits and labels shapes are incompatible.
        """
        # Ensure the logits and labels have compatible dimensions
        if logits.shape[0] != y.shape[0]:
            raise ValueError("The number of logits does not match the number of labels.")

        m, c = logits.shape  # m: batch size, c: number of classes

        # One-hot encode the labels
        y_one_hot = one_hot(y, c)

        # Compute the component-wise product between logits and one-hot encoded labels, and sum over classes
        h = summation(logits * y_one_hot, axis = 1, keepdims = True)

        # Compute the log-sum-exp for the logits for numerical stability in softmax
        log_sum = log_sum_exp(logits)

        # Calculate the loss for each sample
        f = summation(log_sum - h, axis = 0, keepdims = True)

        # Return the average loss over the batch
        return f / m


class MSELoss(LossFunction):
    """
    Computes the Mean Squared Error (MSE) loss function, which is used for regression tasks.

    This class computes the loss based on the difference between the predicted values (logits)
    and the true labels (y).
    """

    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        """
        Computes the forward pass of the MSE loss function.

        :param logits: Predicted values (logits), expected to have shape (batch_size, num_outputs).
        :type logits: Tensor
        :param y: Ground truth labels, expected to have the same shape as logits.
        :type y: Tensor
        :return: The computed MSE loss, averaged over the batch.
        :rtype: Tensor
        :raises ValueError: If the logits and labels have incompatible shapes.
        """
        # Check if logits and y have the same shape
        if logits.shape != y.shape:
            raise ValueError("Logits and labels must have the same shape for MSE loss.")

        m = logits.shape[0]  # Batch size

        # Compute the residual (difference between predicted and true values)
        r = logits - y

        # Compute the MSE loss as the mean of the squared residuals
        f = r.T @ r

        # Return the averaged loss over the batch size
        return f / m

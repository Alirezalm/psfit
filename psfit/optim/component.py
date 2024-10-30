from abc import ABC, abstractmethod
from typing import List, Optional

from psfit import Proximal_L0, Proximal_L1, Proximal_L2
from psfit import tensor
from psfit.data import DataLoader
from psfit.module import LossFunction, LinearModel, Parameter


class Optimizer(ABC):
    @abstractmethod
    def step(self, *args, **kwargs) -> List[Parameter]:
        """
        Performs a single optimization step, updating the model's parameters.

        Parameters:
        -----------
        additional_grad : Optional[Parameter], optional
            An additional gradient term that can be added to the current gradient of the model's weights.

        Returns:
        --------
        List[Parameter]
            A list of updated parameters after the optimization step. The specific implementation depends on
            the subclass.
        """
        pass


class AdmmOptimizer(Optimizer, ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def primal_residual(self):
        pass

    @abstractmethod
    def reset_params(self):
        pass


class LocalOptimizer(Optimizer, ABC):
    """
    Abstract base class for an optimizer in a machine learning model.

    The `LocalOptimizer` class defines a framework for optimization algorithms that update the model's parameters
    during the training process. It must be subclassed and the `step` method must be implemented by any optimizer
    that inherits from this class.

    Attributes:
    -----------
    model : Linear
        The model whose parameters will be optimized. The model is expected to have a `weights` attribute with
        gradients computed during backpropagation.

    Methods:
    --------
    step(additional_grad: Optional[Parameter] = None) -> List[Parameter]
        Abstract method that must be implemented by subclasses to perform the parameter update.
    """

    def __init__(self, model: LinearModel):
        """
        Initializes the optimizer with the model whose parameters will be optimized.

        Parameters:
        -----------
        model : Linear
            A linear model containing the parameters (weights) that need to be optimized.
        """
        self.model = model


class AdmmComponent(ABC):
    """
    Abstract base class for ADMM modules.
    """

    @abstractmethod
    def step(self, *args):
        """
        Perform one ADMM step.

        :param args: Any necessary arguments for the step.
        """
        pass


class LocalSolver(AdmmComponent, ABC):
    """
    Abstract base class for local solver in ADMM.
    """
    pass


class Aggregator(AdmmComponent, ABC):
    """
    Abstract base class for aggregator in ADMM.
    """
    pass


class LocalTrainer(LocalSolver):
    """
    Trainer for local models with ADMM-based optimization.

    :param dataloader: DataLoader object containing training data.
    :type dataloader: DataLoader
    :param loss: Loss function used for training.
    :type loss: LossFunction
    :param optimizer: Optimizer for updating model parameters.
    :type optimizer: LocalOptimizer
    :param epochs: Number of training epochs, defaults to 5.
    :type epochs: int, optional
    :param verbose: Whether to print training progress, defaults to True.
    :type verbose: bool, optional
    """
    loader: Optional[DataLoader] = None
    loss: Optional[LossFunction] = None
    optimizer: Optional[LocalOptimizer] = None
    model: Optional[LinearModel] = None

    def __init__(self,
                 *,
                 dataloader: DataLoader,
                 loss: LossFunction,
                 optimizer: LocalOptimizer,
                 epochs: int = 5,
                 verbose: bool = True) -> None:
        # Validate input types and initialize attributes
        self.loader = self._validate_type(dataloader, DataLoader, "dataloader")
        self.loss = self._validate_type(loss, LossFunction, "loss")
        self.optimizer = self._validate_type(optimizer, LocalOptimizer, "optimizer")
        self.model = self._validate_type(optimizer.model, LinearModel, "model")
        self.epochs = self._validate_type(epochs, int, "epochs")
        self.verbose = self._validate_type(verbose, bool, "verbose")

    @staticmethod
    def _validate_type(value, expected_type, name: str):
        """
        Validates if the given value matches the expected type.

        :param value: Value to validate.
        :param expected_type: The expected type for the value.
        :param name: The name of the variable being validated (for error message).
        :return: The validated value.
        :raises ValueError: If the value is not of the expected type.
        """
        if not isinstance(value, expected_type):
            raise ValueError(f"Invalid {name}. Expected {expected_type.__name__}.")
        return value

    def step(self, u: tensor, y: tensor, penalty: float = 0.1) -> float:
        """
        Performs the training process for the given number of epochs.

        :param u: Scaled dual variables (ADMM method).
        :type u: tensor
        :param y: ADMM equality constraint variable.
        :type y: tensor
        :param penalty: Penalty parameter for ADMM, defaults to 0.1.
        :type penalty: float, optional
        :return: The final loss after all epochs.
        :rtype: float
        """
        # Validate inputs
        self._validate_tensor(u, "u")
        self._validate_tensor(y, "y")
        if penalty <= 0.0:
            raise ValueError("Penalty parameter must be positive.")

        epoch_loss: float = 1e10  # Initialize to a large loss value

        for epoch in range(self.epochs):
            # Perform one training step (one epoch)
            epoch_loss = self.epoch_step(penalty, y, u)

            # Print progress if verbose is enabled
            if self.verbose:
                print(f"Epoch: {epoch + 1: 1d}, Loss: {epoch_loss:.4f}")

        return epoch_loss

    def epoch_step(self, penalty: float, y: tensor, u: tensor) -> float:
        """
        Performs a single epoch of training, including forward and backward propagation.

        :param penalty: Penalty parameter for ADMM.
        :type penalty: float
        :param y: ADMM equality constraint variable.
        :type y: tensor
        :param u: Scaled dual variables (ADMM method).
        :type u: tensor
        :return: The average loss for the epoch.
        :rtype: float
        """
        epoch_loss = 0.0  # Initialize epoch loss
        batch_idx = 0
        # Iterate through batches of data
        for batch_idx, (x_batch, y_batch) in enumerate(self.loader):
            # Forward pass: compute model output (logits)
            logits = self.model(x_batch)

            # Compute loss
            loss = self.loss(logits, y_batch)

            # Backward pass: compute gradients
            loss.backward()

            # Compute ADMM proximal gradient and add it to model gradients
            proximal_grad = penalty * (self.model.weights - y + u)
            admm_grad = proximal_grad.detach()  # Detach to prevent gradient tracking

            # Update model parameters using the optimizer with ADMM gradient
            self.optimizer.step(additional_grad = admm_grad)

            # Accumulate the loss for the current batch
            epoch_loss += loss.detach().array().squeeze()  # TODO: implement loss.item()

        # Return the average loss over all batches
        return epoch_loss / (batch_idx + 1)

    @staticmethod
    def _validate_tensor(value, name: str) -> None:
        """
        Validates if the given value is a tensor.

        :param value: The tensor to validate.
        :param name: Name of the variable being validated (for error message).
        :raises ValueError: If the value is not a tensor.
        """
        if not isinstance(value, tensor):
            raise ValueError(f"{name} must be an instance of the tensor class.")


class StandardAggregator(Aggregator):
    """
    Standard Aggregator that detaches the given tensor from its computation graph.

    :param v: The tensor to be detached from its computation graph.
    :type v: tensor
    :return: The detached tensor.
    :rtype: tensor
    """
    v: tensor = None

    def step(self, v: tensor) -> tensor:
        """
        Detaches the given tensor from the computation graph, preventing further gradient tracking.

        :param v: The tensor to be detached.
        :type v: tensor
        :return: The detached tensor.
        :rtype: tensor
        """
        # Detach the tensor from the computation graph
        return v.detach()


class L1(Aggregator):

    def __init__(self, admm_penalty: float, *, l1_penalty: float = 1.0):
        assert admm_penalty > 0, "admm penalty must be positive."

        self.proximal_param = l1_penalty / admm_penalty

    def step(self, v: tensor) -> tensor:
        rows, cols = v.shape
        for col in range(cols):
            v[:, col] = self.compute_soft_threshold(v[:, col])
        return v

    def compute_soft_threshold(self, column: tensor):
        return Proximal_L1(column, self.proximal_param)


class L2(Aggregator):

    def __init__(self, admm_penalty: float, l2_penalty: float = 1.0):
        self.admm_penalty = admm_penalty
        self.l2_penalty = l2_penalty

    def step(self, v: tensor) -> tensor:
        rows, cols = v.shape
        for col in range(cols):
            v[:, col] = self.compute_proximal(v[:, col])
        return v

    def compute_proximal(self, column: tensor):
        return Proximal_L2(column, rho = self.admm_penalty, l2_penalty = self.l2_penalty)


class ConstraintAggregator(Aggregator):
    pass


class Sparsifier(Aggregator):
    """
    Abstract class for sparsifying tensors based on a given density level.

    :param density: The density level for sparsification, must be a positive integer.
    :type density: int
    """

    density_level: int  # Stores the density level for sparsification

    def __init__(self, density: int) -> None:
        """
        Initializes the Sparsifier with a specified density level.

        :param density: The density level used for sparsification.
        :type density: int
        :raises ValueError: If the density is not a positive integer or float.
        """
        # Validate the density parameter
        if isinstance(density, (int, float)):
            assert density > 0, "Density must be positive."
            self.density_level = density
        else:
            raise ValueError("Density must be a positive integer or float.")

    @abstractmethod
    def sparsify(self, v: tensor, density: int) -> tensor:
        """
        Abstract method to sparsify a tensor based on a specified density level.

        :param v: The input tensor to be sparsified.
        :type v: tensor
        :param density: The level of sparsity to apply to the tensor.
        :type density: int
        :return: The sparsified tensor.
        :rtype: tensor
        """
        pass  # This method needs to be implemented by subclasses


class MipSparsifier(Sparsifier):
    pass


class DcSparsifier(Sparsifier):
    pass


class ProjectionSparsifier(Sparsifier):
    """
    Projection-based sparsifier that sparsifies the input tensor by keeping the largest elements
    in each column based on the specified density level.

    Inherits from the Sparsifier abstract class.
    """

    def step(self, v: tensor) -> tensor:
        """
        Perform sparsification on the input tensor using the defined density level.

        :param v: The input tensor to be sparsified.
        :type v: tensor
        :return: The sparsified tensor with values modified based on the density level.
        :rtype: tensor
        """
        # Sparsify the input tensor using the stored density level
        if not v.is_leaf:
            v = v.detach()

        return self.sparsify(v, density = self.density_level)

    def sparsify(self, vector: tensor, density: int) -> tensor:
        """
        Sparsifies the given tensor by retaining only the largest elements in each column,
        based on the specified density.

        :param vector: The input tensor to be sparsified.
        :type vector: tensor
        :param density: The density level for sparsification, determining how many of the largest elements to keep.
        :type density: int
        :return: The sparsified tensor with only the largest elements retained in each column.
        :rtype: tensor
        """
        # Get the dimensions of the input tensor
        rows, cols = vector.shape

        # Loop through each column and apply sparsification
        for col in range(cols):
            # Keep only the largest `density` number of elements in the current column
            vector[:, col] = Proximal_L0(vector[:, col], density)

        # Detach the tensor from the computation graph
        return vector.detach() if not vector.is_leaf else vector


class DualSolver(AdmmComponent):
    """
    Solver for the dual update step in ADMM (Alternating Direction Method of Multipliers).

    This class computes the dual variable update in the ADMM algorithm.
    """

    @staticmethod
    def step(U: tensor, W: tensor, Y: tensor) -> tensor:
        """
        Performs the dual variable update step by computing (U + W - Y) and detaching the result from the computation graph.

        :param U: Dual variable tensor.
        :type U: tensor
        :param W: Primal variable tensor (updated weights).
        :type W: tensor
        :param Y: Auxiliary variable tensor (ADMM equality constraint).
        :type Y: tensor
        :return: The updated dual variable tensor, detached from the computation graph.
        :rtype: tensor
        """
        # Perform the dual update computation and detach the result
        return (U + W - Y).detach()


def default_trainer(data_loader: DataLoader,
                    criterion: LossFunction,
                    optimizer: LocalOptimizer,
                    *,
                    epochs: int = 1,
                    verbose: bool = False) -> LocalTrainer:
    """
    Initializes a default LocalTrainer with the provided data loader, loss function, and optimizer.

    :param data_loader: The DataLoader object that provides batches of data for training.
    :type data_loader: DataLoader
    :param criterion: The loss function used to compute the model's error.
    :type criterion: LossFunction
    :param optimizer: The optimizer used for updating model parameters.
    :type optimizer: LocalOptimizer
    :param epochs: The number of training epochs, defaults to 1.
    :type epochs: int, optional
    :param verbose: Whether to print training progress, defaults to False.
    :type verbose: bool, optional
    :return: A LocalTrainer object initialized with the provided parameters.
    :rtype: LocalTrainer
    :raises TypeError: If any of the provided arguments are not of the expected type.
    """
    # Validate the data_loader type
    if not isinstance(data_loader, DataLoader):
        raise TypeError("Invalid data_loader type. Expected DataLoader.")

    # Validate the criterion (loss function) type
    if not isinstance(criterion, LossFunction):
        raise TypeError("Invalid criterion type. Expected LossFunction.")

    # Validate the optimizer type
    if not isinstance(optimizer, LocalOptimizer):
        raise TypeError("Invalid optimizer type. Expected LocalOptimizer.")

    # Return an instance of LocalTrainer with the provided arguments
    return LocalTrainer(
        dataloader = data_loader,
        loss = criterion,
        optimizer = optimizer,
        epochs = epochs,
        verbose = verbose
    )


def default_l0_sparsifier(density: int) -> Sparsifier:
    """
    Initializes a default ProjectionSparsifier with the specified density level.

    :param density: The density level for the sparsification process. Must be a positive integer.
    :type density: int
    :return: A ProjectionSparsifier object initialized with the provided density.
    :rtype: ProjectionSparsifier
    :raises TypeError: If the density is not of type int.
    :raises ValueError: If the density is not positive.
    """
    # Validate the density type
    if not isinstance(density, int):
        raise TypeError("Invalid density type. Expected int.")

    # Ensure that the density is positive
    if density <= 0:
        raise ValueError("Density must be a positive integer.")

    # Return an instance of ProjectionSparsifier with the provided density
    return ProjectionSparsifier(density = density)


def default_dual_solver():
    return DualSolver()

from typing import Optional, List

import numpy as np

from psfit import zeros, Tensor
from psfit.module import LinearModel, Parameter
from psfit.optim import (Aggregator,
                         DualSolver,
                         AdmmOptimizer,
                         LocalTrainer,
                         LocalOptimizer)


class SGD(LocalOptimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer for linear models.

    :param model: The model to be optimized, must be an instance of the Linear class.
    :type model: LinearModel
    :param learning_rate: The learning rate for the optimizer, must be a positive float.
    :type learning_rate: float
    :raises ValueError: If the learning rate is not positive.
    """

    def __init__(self, model: LinearModel, *, learning_rate: float = 0.01) -> None:
        """
        Initializes the SGD optimizer with a model and a learning rate.

        :param model: The model to optimize, expected to have accessible weights and gradients.
        :type model: LinearModel
        :param learning_rate: The learning rate for weight updates, defaults to 0.01.
        :type learning_rate: float
        :raises TypeError: If model is not an instance of Linear.
        :raises ValueError: If learning_rate is not a positive float.
        """
        # Validate the model type
        if not isinstance(model, LinearModel):
            raise TypeError(f"Invalid model type. Expected LinearModel, got {type(model).__name__}")

        # Validate the learning rate
        if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
            raise ValueError(f"Invalid learning rate. Must be a positive float, got {learning_rate}")

        super().__init__(model)
        self._lr = float(learning_rate)  # Ensure learning rate is immutable and stored as a float

    @property
    def learning_rate(self) -> float:
        """
        Returns the learning rate (read-only).

        :return: Learning rate.
        :rtype: float
        """
        return self._lr

    def step(self, additional_grad: Optional[Parameter] = None) -> None:
        """
        Performs one step of weight update using SGD, with an optional additional gradient.

        :param additional_grad: An optional additional gradient to be added (e.g., for regularization).
        :type additional_grad: Optional[Parameter]
        :raises RuntimeError: If model weights or gradients are not properly initialized.
        """
        try:
            # Validate that the model's weights and gradients exist
            if not hasattr(self.model.weights, 'gradient'):
                raise RuntimeError("Model weights or gradients are not initialized.")

            # Retrieve the model's weight gradient, optionally add an additional gradient
            w_grad = self.model.weights.gradient
            if additional_grad is not None:
                w_grad += additional_grad

            # Update the model weights using the gradient and the learning rate
            self.model.weights -= self._lr * w_grad

            # Detach the weights to prevent gradient tracking in future computations
            self.model.weights = self.model.weights.detach()

        except Exception as e:
            raise RuntimeError(f"Error during SGD step: {str(e)}")


class Admm(AdmmOptimizer):
    """
    ADMM (Alternating Direction Method of Multipliers) optimizer that updates model parameters
    based on dual and primal updates. It coordinates the training process using a trainer, aggregator,
    and dual solver.

    :param trainer: The trainer responsible for model updates.
    :type trainer: LocalTrainer
    :param aggregator: The aggregator used for the y-update.
    :type aggregator: Aggregator
    :raises ValueError: If the trainer or aggregator is of an invalid type.
    :raises RuntimeError: If model initialization fails.
    """

    trainer: Optional[LocalTrainer] = None
    aggregators: List[Aggregator] | Aggregator
    dual_solver: Optional[DualSolver] = None

    def __init__(self, *, trainer: LocalTrainer, aggregators: List[Aggregator] | Aggregator) -> None:
        """
        Initializes the ADMM optimizer with the provided trainer and aggregator, and sets up dual and auxiliary variables.

        :param trainer: The LocalTrainer object used for model updates.
        :type trainer: LocalTrainer
        :param aggregators: The Aggregator object used for the y-update step.
        :type aggregators: List[Aggregator] List of aggregators to be applied in the y-update
        :raises ValueError: If the trainer or aggregator is of an invalid type.
        """
        # Validate the trainer and aggregator
        if not isinstance(trainer, LocalTrainer):
            raise ValueError("Invalid trainer: Expected a LocalTrainer instance.")

        if isinstance(aggregators, Aggregator):
            aggregators = [aggregators]

        if not aggregators:
            raise ValueError("Invalid Aggregator: Expected an Aggregator or a List of Aggregators.")

        # Assign the trainer and aggregator
        self.trainer = trainer
        self.aggregators = aggregators

        # Initialize the dual solver
        self.dual_solver = DualSolver()

        # Initialize dual variable u and auxiliary variable y
        try:
            self.u, self.y = self.initialize()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize dual and auxiliary variables: {e}")

    def initialize(self) -> tuple:
        """
        Initializes dual variable `u` and auxiliary variable `y` based on the model type.

        :return: Tuple of initialized dual variable `u` and auxiliary variable `y`.
        :rtype: tuple
        :raises PsfitException: If the model is neither a Classifier nor a Regressor, or has invalid dimensions.
        """
        n, c = self.trainer.model.size

        # Boundary check for dimensions
        if n <= 0 or c <= 0:
            raise ValueError(f"Invalid model dimensions: n = {n}, c = {c}. Dimensions must be positive.")

        # Initialize u and y as zero tensors
        u = zeros(n, c)
        y = zeros(n, c)
        return u, y

    def update_u(self) -> None:
        """
        Updates the dual variable `u` using the dual solver.

        :raises RuntimeError: If the dual solver step fails.
        """
        try:
            w = self.get_params()
            self.u = self.dual_solver.step(self.u, w, self.y)
        except Exception as e:
            raise RuntimeError(f"Failed to update dual variable `u`: {e}")

    def get_params(self) -> Tensor:
        """
        Retrieves the current model parameters (weights) safely.

        :return: The model's weights, detached to prevent unintended modifications.
        :rtype: tensor
        """
        return self.trainer.model.weights.detach()

    def update_y(self) -> None:
        """
        Updates the auxiliary variable `y` using the aggregator.

        :raises RuntimeError: If the aggregator step fails.
        """
        try:
            w = self.get_params()
            v = w + self.u
            for aggregator in self.aggregators:  # apply chain proximal operator
                v = aggregator.step(v)
            self.y = v
        except Exception as e:
            raise RuntimeError(f"Failed to update auxiliary variable `y`: {e}")

    def update_x(self, penalty: float) -> float:
        """
        Performs the x-update step by training the model with the given penalty.

        :param penalty: The penalty parameter used in the ADMM method, must be positive.
        :type penalty: float
        :return: The loss from the training step.
        :rtype: float
        :raises ValueError: If the penalty is not positive.
        :raises RuntimeError: If the trainer fails to update.
        """
        if penalty <= 0:
            raise ValueError("Penalty must be a positive value.")

        try:
            loss = self.trainer.step(self.u, self.y, penalty)
            return float(loss)
        except Exception as e:
            raise RuntimeError(f"Failed to update model parameters in x-update step: {e}")

    def update_params(self) -> None:
        """
        Updates the model's weights with the detached auxiliary variable `y`.

        :raises RuntimeError: If updating model parameters fails.
        """
        try:
            self.trainer.optimizer.model.weights = self.y.detach()
        except Exception as e:
            raise RuntimeError(f"Failed to update model parameters: {e}")

    def primal_residual(self) -> float:
        """
        Computes the primal residual, a measure of the difference between the model's weights and the auxiliary variable `y`.

        :return: The primal residual, computed as the L2 norm squared.
        :rtype: float
        :raises RuntimeError: If residual computation fails.
        """
        try:
            w = self.get_params()
            y = self.y.detach()

            error = w - y

            # Move to CPU if the error tensor is on GPU
            if error.device == "gpu":
                error = error.array().cpu()
                return np.linalg.norm(error.numpy().reshape(-1, 1), 2) ** 2

            # Compute the L2 norm squared of the error
            return np.linalg.norm(error.array().reshape(-1, 1), 2) ** 2

        except Exception as e:
            raise RuntimeError(f"Failed to compute primal residual: {e}")

    def reset_params(self) -> None:
        """
        Resets the model's parameters to their initial values.

        :raises RuntimeError: If resetting parameters fails.
        """
        try:
            self.trainer.optimizer.model.init_params()
        except Exception as e:
            raise RuntimeError(f"Failed to reset model parameters: {e}")

    def step(self, penalty: float = 0.1, *, warm_start: bool = True) -> tuple:
        """
        Performs a full ADMM step including updates for `x`, `y`, and `u`, and computes the primal residual.

        :param penalty: The penalty parameter used for the x-update, defaults to 0.1.
        :type penalty: float, optional
        :param warm_start: Whether to start with the current parameters (True) or reset them (False), defaults to True.
        :type warm_start: bool, optional
        :return: A tuple containing the training loss and primal residual.
        :rtype: tuple
        :raises ValueError: If the penalty is not positive.
        :raises RuntimeError: If any of the update steps fail.
        """
        if penalty <= 0:
            raise ValueError("Penalty must be a positive value.")

        # Reset parameters if not using warm start
        if not warm_start:
            self.reset_params()

        try:
            # Perform x-update (training)
            training_loss = self.update_x(penalty)

            # Perform y-update
            self.update_y()

            # Perform u-update (dual variable)
            self.update_u()

            # Compute the primal residual
            primal_residual = self.primal_residual()

            # Update model parameters with the current `y`
            self.update_params()

            return training_loss, primal_residual
        except Exception as e:
            raise RuntimeError(f"ADMM step failed: {e}")

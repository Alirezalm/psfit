from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score

from psfit.module import Linear, Tensor, PsfitException
from psfit import tensor
from psfit.data import DataLoader, Dataset
from psfit.module import SoftmaxLoss
from psfit.optim import LocalTrainer, SGD, Sparsifier, StandardAggregator
from psfit.optim.optimizer import Admm


class Model(ABC):
    """
    An abstract base class for machine learning models.

    This class defines the basic interface that all machine learning models must implement,
    including methods for training the model and making predictions.

    Methods:
        fit(epochs: int): Train the model for a specified number of epochs.
        predict(x: Tensor): Make predictions on the input data.

    Properties:
        params: Model parameters.
    """

    @abstractmethod
    def fit(self, epochs: int):
        """
        Train the model for a specified number of epochs.

        Args:
            epochs (int): The number of epochs to train the model.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, x: Tensor):
        """
        Make predictions on the input data.

        Args:
            x (Tensor): Input data for making predictions.

        Returns:
            Tensor: The predicted outputs.

        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def params(self):
        """
        Get the model parameters.

        Returns:
            The model parameters.

        Must be implemented by subclasses.
        """
        pass


class SparseSoftmaxClassifier(Model):
    """
    A Sparse Softmax Classifier that implements the Model interface.

    This classifier uses the softmax function for classification tasks and supports
    the Alternating Direction Method of Multipliers (ADMM) for optimization.

    Attributes:
        optimizer (Optional[Admm]): The optimizer used for training the model.
        verbose (bool): Whether to print training progress.
        _local_epochs (int): Number of local epochs for training.
        m (int): Number of samples in the dataset.
        n (int): Number of features in the dataset.
        c (int): Number of classes in the dataset.
        penalty (float): Regularization penalty for training.
        loader (DataLoader): DataLoader instance for managing batches of data.
        x_cv (Tensor): Cross-validation input data.
        y_cv (Tensor): Cross-validation target data.
        training (bool): Indicates whether to calculate training accuracy.
        validation (bool): Indicates whether to calculate validation accuracy.
        iterations (int): Number of ADMM iterations for training.
    """

    optimizer: Optional[Admm] = None

    def __init__(self,
                 dataset: Dataset,
                 *,
                 batch_size: int = 64,
                 learning_rate: float = 0.01,
                 penalty: float = 1.0,
                 local_epochs: int = 1,
                 admm_iterations: int = 10,
                 training_accuracy: bool = True,
                 validation_accuracy: bool = True,
                 verbose: bool = True,
                 local_verbose: bool = False):

        self.verbose = verbose
        self._local_epochs = local_epochs

        if isinstance(dataset, Dataset):
            self.m, self.n, self.c = dataset.number_of_samples, dataset.number_of_features, dataset.number_of_classes
        else:
            raise ValueError("Softmax classifier only supports 'Dataset' type")

        if isinstance(batch_size, (int, float)):
            assert 0 < batch_size < self.m, f"batch size must be between ({1},{self.m})"
            batch_size = int(batch_size)
        else:
            raise ValueError("batch size must be int")

        if isinstance(learning_rate, (int, float)):
            assert 0 < learning_rate, f"learning rate must be positive"
            learning_rate = float(learning_rate)
        else:
            raise ValueError("learning rate must be float or int")

        if isinstance(penalty, (int, float)):
            assert 0 < penalty, f"penalty parameter must be positive"
            self.penalty = float(penalty)
        else:
            raise ValueError("penalty must be float or int")

        self._model = Linear(in_features = self.n, out_features = self.c)

        self.loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, report = verbose)

        if validation_accuracy:
            if self.loader.cv:
                x, y = self.loader.cv_data
                self.x_cv = tensor(x)
                self.y_cv = y
            else:
                raise PsfitException("make sure validation data is available in Dataset")

        self.validation = validation_accuracy
        self.training = training_accuracy

        assert local_epochs > 0, "the number of local epochs must be positive"

        self.trainer = LocalTrainer(
            dataloader = self.loader,
            loss = SoftmaxLoss(),
            optimizer = SGD(model = self.model, learning_rate = learning_rate),
            epochs = self._local_epochs,
            verbose = local_verbose
        )
        assert admm_iterations > 0, "number of iterations must be positive"
        self.iterations = admm_iterations

    def fit(self, sparsifier: Optional[Sparsifier] = None, warm_start: bool = True):
        """
        Train the model using the specified sparsifier.

        Args:
            sparsifier (Optional[Sparsifier]): The sparsifier to be used during training.

            warm_start (bool): The warm start to be used by the admm algorithm
        Returns:
            Tuple: A tuple containing training loss values and primal error values.
        """
        self.model.init_params()

        if not sparsifier:
            sparsifier = StandardAggregator()

        self.setup_optimizer(sparsifier)

        training_loss_values = []
        primal_error_values = []

        for iteration in range(self.iterations):
            training_loss, residual = self.optimizer.step(penalty = self.penalty, warm_start = warm_start)
            training_loss_values.append(training_loss)
            primal_error_values.append(residual)

            training_report = f" Training Accuracy: {self.training_accuracy():.4f}" if self.training else ""
            validation_report = f" Validation Accuracy: {self.validation_accuracy():.4f}" if self.validation else ""

            if (iteration + 1) % 10 == 0 and self.verbose:
                print(f"Epoch [{iteration + 1}/{self.iterations}]"
                      f" - Loss: {training_loss:.4f}",
                      training_report,
                      validation_report,
                      f" Residual: {residual:.4f}")

        return training_loss_values, primal_error_values

    @property
    def local_epoch(self):
        """
        Get the number of local epochs.

        Returns:
            int: The number of local epochs.
        """
        return self._local_epochs

    @local_epoch.setter
    def local_epoch(self, value: int):
        """
        Set the number of local epochs.

        Args:
            value (int): The new number of local epochs.
        """
        self.trainer.epochs = value

    def validation_accuracy(self):
        """
        Calculate the validation accuracy of the model.

        Returns:
            float: The validation accuracy.
        """
        predictions = self.predict(self.x_cv)
        return accuracy_score(predictions, self.y_cv)

    def training_accuracy(self):
        """
        Calculate the training accuracy of the model.

        Returns:
            float: The training accuracy.
        """
        training_acc = 0.0
        for (batch, (x, y)) in enumerate(self.optimizer.trainer.loader):
            y = y.array().cpu() if y.device == "gpu" else y.array()
            predictions = self.predict(x)
            training_acc += accuracy_score(predictions, y)
        return training_acc / (batch + 1)

    def setup_optimizer(self, sparsifier: Sparsifier):
        """
        Set up the optimizer with the given sparsifier.

        Args:
            sparsifier (Sparsifier): The sparsifier to be used.

        Raises:
            ValueError: If the sparsifier is not valid.
        """
        if not isinstance(sparsifier, Sparsifier):
            raise ValueError("invalid sparsifier.")

        self.optimizer = Admm(
            trainer = self.trainer,
            aggregators = sparsifier
        )

    def predict(self, x: Tensor):
        """
        Make predictions for the input data.

        Args:
            x (Tensor): The input data for predictions.

        Returns:
            Tensor: The predicted classes for the input data.
        """
        logits = self.model(x)
        if logits.device == "gpu":
            logits = logits.detach().array().cpu()
        else:
            logits = logits.detach().array()
        predictions = np.argmax(logits, axis = 1, keepdims = True)
        return predictions

    @property
    def params(self):
        """
        Get the model parameters.

        Returns:
            The parameters of the model.
        """
        return self.optimizer.trainer.model.weights

    @property
    def model(self):
        """
        Get the underlying model.

        Returns:
            The underlying model.
        """
        return self._model

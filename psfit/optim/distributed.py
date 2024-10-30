from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import numpy as np
import ray
from ray import remote

from psfit import Tensor, zeros, Linear
from psfit.data import DataLoader, Dataset, get_loader
from psfit.module import LossFunction, LinearModel
from psfit.optim import (Aggregator,
                         AdmmOptimizer,
                         LocalOptimizer,
                         default_trainer,
                         DualSolver,
                         SGD,
                         default_dual_solver)


# Abstract base class representing a generic node (either server or client)
class Node(ABC):
    _id_counter: int = 0  # Class variable to track unique IDs for nodes

    def __init__(self):
        # Assign unique id to each node
        self.id = Node._id_counter
        Node._id_counter += 1

    @abstractmethod
    def init(self):
        """Initialize node-specific parameters."""
        pass

    @abstractmethod
    def reset(self):
        """Reset node-specific parameters."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__.lower()}(id={self.id})"


# Client node class in a federated learning or distributed system
class Client(Node):
    def __init__(self,
                 data_loader: DataLoader,
                 criterion: LossFunction,
                 optimizer: LocalOptimizer,
                 *,
                 epochs: int = 1,
                 verbose: bool = False):
        super().__init__()

        # Set up the local trainer for this client
        self.trainer = default_trainer(data_loader, criterion, optimizer, epochs = epochs, verbose = verbose)

        # Dual variable solver for local updates
        self.dual_solver = DualSolver()

        # Model and local parameters
        self.model = optimizer.model
        self._u: Optional[Tensor] = self.init()  # Initialize dual variable
        self._local_loss: float = float('inf')  # Local loss, initialized to a high value

    def init(self) -> Tensor:
        """Initialize dual variable 'u' based on the model size."""
        n, c = self.model.size
        return zeros(n, c)

    def model_update(self, y: Tensor, penalty: float):
        """Perform model update based on the server's variable `y` and penalty term."""
        self._local_loss = self.trainer.step(self.u, y, penalty = penalty)

    def state_update(self, y: Tensor):
        """Update the dual variable `u` using the dual solver and server's variable `y`."""
        self._u = self.dual_solver.step(self.u, self.w, y)

    @property
    def local_loss(self) -> float:
        """Returns the local loss value."""
        return self._local_loss

    @property
    def u(self) -> Tensor:
        """Returns the current value of the dual variable `u`."""
        return self._u

    @property
    def w(self) -> Tensor:
        """Returns the model's weights, detached from computational graph."""
        return self.model.weights.detach()

    def primal_residual(self, y: Tensor) -> float:
        """Compute the primal residual for convergence checks."""
        error = self.w - y.detach()

        # Ensure data is on CPU for computation
        if error.device == "gpu":
            error = error.array().cpu()
        else:
            error = error.array()

        # Compute L2 norm (squared) of the error
        return np.linalg.norm(error.reshape(-1, 1), 2) ** 2

    def reset(self):
        """Reset the model and dual variables to their initial state."""
        self._u = self.init()  # Reset dual variable
        self.model.init_params()  # Reinitialize model parameters


# Server node class in a federated learning system
class Server(Node):
    def __init__(self, clients: List[Client], aggregator: Aggregator):
        super().__init__()

        if not isinstance(aggregator, Aggregator):
            raise TypeError("invalid aggregator type")

        assert clients, "list of clients cannot be empty"

        self.aggregator = aggregator
        self.clients = clients
        self._y: Optional[Tensor] = self.init()  # Server's state variable

    def init(self) -> Tensor:
        """Initialize the server's state variable `y` based on client model dimensions."""
        n, c = self.clients[0].w.shape
        return zeros(n, c)

    @staticmethod
    def reduce(values: List[Union[Tensor, int, float]], op: str = "avg") -> Union[Tensor, float]:
        """Reduces a list of values using the specified operation (sum or average)."""
        reduced_values = sum(values)

        if op.lower() == "sum":
            return reduced_values.detach() if isinstance(reduced_values, Tensor) else reduced_values
        elif op.lower() == "avg":
            averaged = reduced_values / len(values)
            return averaged.detach() if isinstance(averaged, Tensor) else averaged
        else:
            raise ValueError(f"Unsupported operation '{op}' for reduction")

    def update(self):
        """Update the server's state variable by aggregating client updates."""
        # v is the average of each client's w_i + u_i
        values = [(client.w + client.u).detach() for client in self.clients]
        v = self.reduce(values, op = "avg")
        self._y = self.aggregator.step(v)

    @property
    def y(self) -> Tensor:
        """Returns the server's current state variable."""
        return self._y

    def reset(self):
        """Reset the server's state variable to its initial state."""
        self._y = self.init()


# Distributed optimization class using ADMM (Alternating Direction Method of Multipliers)
class DistributedAdmmSim(AdmmOptimizer):
    def __init__(self, *, server: Server):
        # Initialize clients and server
        self.clients = server.clients
        self.server = server

        # Perform initial parameter setup
        self.initialize()

    def step(self, penalty: float = 0.1, *, warm_start: bool = True):
        """Performs one iteration of ADMM optimization across clients and server."""
        # Update server state
        self.server.update()

        # Update model for each client
        for client in self.clients:
            client.model_update(self.server.y, penalty)

        # Update dual variable for each client
        for client in self.clients:
            client.state_update(self.server.y)

        # Compute and log total loss for the iteration
        total_loss = self.iter_loss()
        self.primal_residual()

    def iter_loss(self) -> float:
        """Compute the average loss across all clients."""
        loss_values = [client.local_loss for client in self.clients]
        return self.server.reduce(loss_values, op = "avg")

    def initialize(self):
        """Initialize the state of all clients and the server."""
        for client in self.clients:
            client.init()
        self.server.init()

    def primal_residual(self):
        """Compute and print the primal residual for each client (for convergence monitoring)."""
        for client in self.clients:
            # Compute error between server state and client weights
            error_norm = np.linalg.norm(self.server.y.array() - client.w.array(), 2)
            print(f"client: {client.id}, loss: {client.local_loss:5f}, error: {error_norm:.5f}")
        print("------------------------------------")

    def reset_params(self):
        """Optional reset function for any additional parameter resetting."""
        pass


@dataclass
class WorkerConfig:
    """
    Configuration settings for a worker in the distributed system.

    :param worker_id: Unique identifier for the worker.
    :type worker_id: int
    :param batch_size: The batch size used in data loading.
    :type batch_size: int
    :param random_seed: Seed for random number generators to ensure reproducibility.
    :type random_seed: int
    :param loader_report: Flag to enable reporting during data loading.
    :type loader_report: bool
    :param model_size: Dimensions of the model parameters.
    :type model_size: Tuple[int, int]
    :param sgd_lr: Learning rate for SGD optimizer.
    :type sgd_lr: float
    :param local_epochs: Number of local epochs for training.
    :type local_epochs: int
    :param local_optim_report: Flag to enable reporting during local optimization.
    :type local_optim_report: bool
    """
    worker_id: int = None
    batch_size: int = 64
    random_seed: int = 101
    loader_report: bool = True
    model_size: Tuple[int, int] = ()
    sgd_lr: float = 0.01
    local_epochs: int = 5
    local_optim_report: bool = False


def default_settings():
    """
    Provides the default settings for a worker.

    :return: A list containing a single WorkerConfig with default values.
    :rtype: List[WorkerConfig]
    """
    return [WorkerConfig()]


@remote
class ParameterServer:
    """
    Parameter server that holds the global parameters and coordinates the workers.

    :ivar _y: The global parameter tensor.
    :vartype _y: Optional[Tensor]
    :ivar _setting: List of worker configurations.
    :vartype _setting: List[WorkerConfig]
    """
    _y: Optional[Tensor] = None
    _setting: List[WorkerConfig] = []

    def __init__(self, aggregators: List[Aggregator] | Aggregator):
        """
        Initializes the ParameterServer with an aggregator.

        :param aggregators: The aggregators used for updating global parameters.
        :type aggregators: List[Aggregator] | Aggregator
        :raises TypeError: If the aggregator is not an instance of Aggregator.
        """
        if isinstance(aggregators, Aggregator):
            aggregators = [aggregators]

        if not aggregators:
            raise ValueError("Invalid Aggregator: Expected an Aggregator or a List of Aggregators.")

        self._setting: List[WorkerConfig] = []
        self.aggregators = aggregators

    def initialize(self, size: Tuple[int, int]):
        """
        Initializes the global parameter tensor with zeros.

        :param size: Size of the parameter tensor.
        :type size: Tuple[int, int]
        """
        self._y = zeros(*size)

    def add_worker_config(self, config: WorkerConfig = None):
        """
        Adds a worker configuration to the server.

        :param config: Configuration settings for a worker.
        :type config: WorkerConfig
        """
        self._setting.append(config)

    def get_param(self) -> Tensor:
        """
        Retrieves the current global parameter tensor.

        :return: The global parameter tensor.
        :rtype: Tensor
        :raises RuntimeError: If the Parameter Server is not properly initialized.
        """
        if self._y:
            return self._y.detach()
        raise RuntimeError("Make sure the Parameter Server is initialized properly")

    def get_setting(self, worker_id: int):
        """
        Retrieves the configuration settings for a specific worker.

        :param worker_id: The unique identifier of the worker.
        :type worker_id: int
        :return: The WorkerConfig for the specified worker.
        :rtype: WorkerConfig
        :raises RuntimeError: If the worker settings are not properly set up.
        """
        for setting in self._setting:
            if setting.worker_id == worker_id:
                return setting
        raise RuntimeError("Make sure worker settings are setup properly.")

    def step(self, params: List[Tensor]):
        """
        Updates the global parameters by aggregating local parameters from workers.

        :param params: List of tensors from workers to be aggregated.
        :type params: List[Tensor]
        """
        v = self.reduce(params, operation = "AVG")

        for aggregator in self.aggregators:
            v = aggregator.step(v)

        self._y = v

    @staticmethod
    def reduce(values: List[Union[Tensor, float, int]], operation: str = "AVG"):
        """
        Reduces a list of values using the specified operation.

        :param values: List of values to reduce.
        :type values: List[Tensor | float | int]
        :param operation: Reduction operation ('SUM' or 'AVG').
        :type operation: str
        :return: The reduced value.
        :rtype: Tensor or float or int
        :raises RuntimeError: If an unsupported operation is specified.
        """
        if operation.upper() == "SUM":
            return sum(values)
        if operation.upper() == "AVG":
            return sum(values) / len(values)
        raise RuntimeError(f"Unsupported operation {operation}.")


@remote
class Worker:
    """
    Represents a worker node in the distributed system that performs local computations.

    :ivar _id: Unique identifier of the worker.
    :vartype _id: int
    :ivar dataset: Local dataset assigned to the worker.
    :vartype dataset: Optional[Dataset]
    :ivar model: Local model instance.
    :vartype model: Optional[LinearModel]
    :ivar server: Reference to the ParameterServer.
    :vartype server: Optional[ParameterServer]
    :ivar settings: Configuration settings for the worker.
    :vartype settings: Optional[WorkerConfig]
    :ivar loss: Loss function used in training.
    :vartype loss: Optional[LossFunction]
    :ivar dual_solver: Solver for the dual optimization problem.
    :vartype dual_solver: Optional[DualSolver]
    :ivar _u: Local state tensor.
    :vartype _u: Optional[Tensor]
    :ivar local_loss_value: The value of the local loss function.
    :vartype local_loss_value: float
    """
    _id: int = Node
    dataset: Optional[Dataset] = None
    model: Optional[LinearModel] = None
    server: Optional[ParameterServer] = None
    settings: Optional[WorkerConfig] = None
    loss: Optional[LossFunction] = None
    dual_solver: Optional[DualSolver] = None
    _u: Optional[Tensor] = None
    local_loss_value: float = 1e10

    def __init__(self,
                 worker_id: int,
                 local_dataset: Dataset,
                 local_loss: LossFunction,
                 server: ParameterServer):
        """
        Initializes a Worker instance.

        :param worker_id: Unique identifier for the worker.
        :type worker_id: int
        :param local_dataset: The dataset assigned to the worker.
        :type local_dataset: Dataset
        :param local_loss: The loss function to be used in training.
        :type local_loss: LossFunction
        :param server: Reference to the ParameterServer.
        :type server: ParameterServer
        """
        self._id = worker_id
        self.server = server
        self.dataset = local_dataset
        self.loss = local_loss

        self.settings = self.download_settings_from_server(worker_id)
        loader = self.setup_data_loader(self.dataset,
                                        batch_size = self.settings.batch_size,
                                        random_seed = self.settings.random_seed,
                                        report = self.settings.loader_report)

        self.model = self.setup_model()

        optimizer = self.setup_local_optimizer(model = self.model)

        self.trainer = self.setup_trainer(loader, optimizer, self.loss)

        self.dual_solver = default_dual_solver()

        self._u = self.initialize()

    def initialize(self):
        """
        Initializes the local state tensor.

        :return: A tensor initialized to zeros with the same size as the model parameters.
        :rtype: Tensor
        """
        size = self.model.size
        return zeros(*size)

    def download_settings_from_server(self, worker_id):
        """
        Downloads the worker's settings from the parameter server.

        :param worker_id: The unique identifier of the worker.
        :type worker_id: int
        :return: The WorkerConfig for the worker.
        :rtype: WorkerConfig
        :raises RuntimeError: If there is an error retrieving settings from the server.
        """
        # this method is explicitly called by the client
        settings_id = self.server.get_setting.remote(worker_id)

        try:
            return ray.get(settings_id)

        except Exception as exp:
            raise RuntimeError(f"Failed reading the settings: {exp}")

    def update_local_model(self, y: Tensor, penalty: float = 5):
        """
        Updates the local model parameters based on the received global parameters.

        :param y: Global parameter tensor downloaded from the server.
        :type y: Tensor
        :param penalty: Penalty parameter for the optimization.
        :type penalty: float
        :return: The value of the local loss function after the update.
        :rtype: float
        """
        u = self.get_local_state()

        self.local_loss_value = self.trainer.step(u, y, penalty)
        return self.local_loss_value

    def update_local_state(self, y: Tensor):
        """
        Updates the local state based on the received global parameters.

        :param y: Global parameter tensor downloaded from the server.
        :type y: Tensor
        """
        u = self.get_local_state()
        w = self.get_local_params()
        self._u = self.dual_solver.step(u, w, y)

    def get_local_params(self):
        """
        Retrieves the local model parameters.

        :return: The local model parameters.
        :rtype: Tensor
        """
        return self.model.parameters.detach()

    def download_param_from_server(self):
        """
        Downloads the global parameters from the parameter server.

        :return: The global parameter tensor.
        :rtype: Tensor
        :raises RuntimeError: If there is an error retrieving parameters from the server.
        """
        param_id = self.server.get_param.remote()

        try:
            return ray.get(param_id)

        except Exception as exp:
            raise RuntimeError(f"Failed reading the parameter from the parameter server: {exp}")

    def param_size(self):
        """
        Retrieves the size of the model parameters.

        :return: The size of the model parameters.
        :rtype: Tuple[int, int]
        """
        return self.model.size

    @staticmethod
    def setup_data_loader(dataset: Dataset,
                          batch_size: int,
                          random_seed: int = 101,
                          report: bool = True) -> DataLoader:
        """
        Sets up the data loader for the worker.

        :param dataset: The dataset to load.
        :type dataset: Dataset
        :param batch_size: The batch size for data loading.
        :type batch_size: int
        :param random_seed: Seed for random number generators.
        :type random_seed: int
        :param report: Flag to enable reporting during data loading.
        :type report: bool
        :return: Configured DataLoader instance.
        :rtype: DataLoader
        """
        return get_loader(dataset, batch_size, random_seed, report)

    def setup_model(self) -> LinearModel:
        """
        Sets up the local model based on the settings.

        :return: The initialized LinearModel.
        :rtype: LinearModel
        :raises RuntimeError: If the model size is not initialized in the settings.
        """
        if self.settings.model_size:
            model_size = self.settings.model_size
            return Linear(*model_size)
        raise RuntimeError("Model size is not initialized in the settings")

    def setup_local_optimizer(self, model: LinearModel):
        """
        Sets up the local optimizer for the model.

        :param model: The local model to optimize.
        :type model: LinearModel
        :return: Configured optimizer instance.
        :rtype: Optimizer
        """
        lr = self.settings.sgd_lr
        return SGD(model, learning_rate = lr)

    def setup_trainer(self, loader: DataLoader, optimizer: LocalOptimizer, loss: LossFunction):
        """
        Sets up the trainer for local optimization.

        :param loader: DataLoader for providing data batches.
        :type loader: DataLoader
        :param optimizer: Optimizer for model training.
        :type optimizer: LocalOptimizer
        :param loss: Loss function for training.
        :type loss: LossFunction
        :return: Configured trainer instance.
        :rtype: Trainer
        """
        epochs = self.settings.local_epochs
        report = self.settings.local_optim_report
        return default_trainer(loader, loss, optimizer, epochs = epochs, verbose = report)

    def get_local_state(self):
        """
        Retrieves the local state tensor.

        :return: The local state tensor.
        :rtype: Tensor
        """
        return self._u

    def compute_primal_residual(self, y: Tensor):
        """
        Computes the primal residual with respect to the global parameters.

        :param y: Global parameter tensor downloaded from the server.
        :type y: Tensor
        :return: The computed primal residual.
        :rtype: float
        """
        # y is downloaded from the server
        w = self.get_local_params()
        err = (w - y).detach().array().reshape(-1, 1)
        # TODO: consider GPU here
        r = np.linalg.norm(err, 2) ** 2
        return r

    def status_report(self, y: Tensor):
        """
        Reports the status of the worker including local loss and error.

        :param y: Global parameter tensor downloaded from the server.
        :type y: Tensor
        """
        r = self.compute_primal_residual(y)
        loss_value = self.local_loss_value
        print(f"Local Loss:{loss_value:.4f} - Local Error: {r:.4f}")


def create_parameter_server(aggregators: List[Aggregator] | Aggregator):
    """
    Creates a remote ParameterServer instance.

    :param aggregators: The aggregators to use in the ParameterServer.
    :type aggregators: List[Aggregator]|Aggregator
    :return: A remote instance of ParameterServer.
    :rtype: ParameterServer
    """
    return ParameterServer.remote(aggregators)


def create_worker(worker_id: int,
                  local_dataset: Dataset,
                  local_loss: LossFunction,
                  server: ParameterServer):
    """
    Creates a remote Worker instance.

    :param worker_id: Unique identifier for the worker.
    :type worker_id: int
    :param local_dataset: Dataset assigned to the worker.
    :type local_dataset: Dataset
    :param local_loss: Loss function used by the worker.
    :type local_loss: LossFunction
    :param server: Reference to the ParameterServer.
    :type server: ParameterServer
    :return: A remote instance of Worker.
    :rtype: Worker
    """
    return Worker.remote(worker_id, local_dataset, local_loss, server)


class DistributedDataParallel(AdmmOptimizer):
    """
    Implements the Distributed Data Parallel optimization using ADMM.

    :ivar workers: List of worker instances.
    :vartype workers: List[Worker]
    :ivar server: Parameter server instance.
    :vartype server: ParameterServer
    """

    def primal_residual(self):
        """
        Computes the primal residual for the optimization problem.
        """
        pass

    def reset_params(self):
        """
        Resets the parameters of the optimizer.
        """
        pass

    workers: List[Worker] = []
    server: Optional[ParameterServer] = None

    def __init__(self, workers: List[Worker], server: ParameterServer):
        """
        Initializes the DistributedDataParallel optimizer.

        :param workers: List of worker instances.
        :type workers: List[Worker]
        :param server: The parameter server instance.
        :type server: ParameterServer
        :raises RuntimeError: If the workers list is empty.
        """
        if not workers:
            raise RuntimeError("workers cannot be an empty list")

        self.workers = workers
        self.server = server

        self.initialize()

    def initialize(self):
        """
        Initializes the optimizer by setting up the model sizes and initializing workers.
        """
        model_size_id = self.workers[0].param_size.remote()
        model_size = ray.get(model_size_id)
        self.server.initialize.remote(model_size)
        [worker.initialize.remote() for worker in self.workers]

    def step(self, penalty: float = 1.0):
        """
        Performs a distributed optimization step.

        :param penalty: Penalty parameter for the optimization.
        :type penalty: float
        """
        # Step 1: Fetch server parameters and update workers' local state and model
        server_params_futures = [worker.download_param_from_server.remote() for worker in self.workers]
        server_params = ray.get(server_params_futures)

        local_aggregates = []
        for worker, server_param in zip(self.workers, server_params):
            # Update worker states and models asynchronously
            worker.update_local_state.remote(server_param)
            worker.update_local_model.remote(server_param)

            # Gather weights and states asynchronously
            local_weights = worker.get_local_params.remote()
            local_state = worker.get_local_state.remote()

            # Aggregate parameters
            aggregate = ray.get(local_weights) + ray.get(local_state)
            local_aggregates.append(aggregate)

            # Status report (run asynchronously)
            worker.status_report.remote(server_param)

        # Step 2: Perform server-side aggregation
        self.server.step.remote(local_aggregates)

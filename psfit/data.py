from abc import ABC, abstractmethod
from random import Random
from typing import Optional, Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split

from psfit import PsfitException, Tensor, get_backend

backend = get_backend()
NDArray = np.ndarray


class Dataset(ABC):
    """
    An abstract base class that defines the interface for datasets used in machine learning tasks.

    This class serves as a blueprint for any dataset implementation, ensuring that all derived
    classes provide the necessary properties and methods to access the dataset's features,
    samples, and data for training and cross-validation.

    """

    @property
    @abstractmethod
    def number_of_features(self):
        """
        int: The number of features in the dataset.

        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def number_of_samples(self):
        """
        int: The total number of samples in the dataset.

        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def train_data(self):
        """
        iterable: The training data for the dataset.

        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def cv_data(self):
        """
        iterable: The cross-validation data for the dataset.

        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def number_of_classes(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        """
        Retrieve a sample from the dataset.

        Args:
            item (int): The index of the sample to retrieve.

        Returns:
            The sample at the specified index.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        Must be implemented by subclasses.
        """
        pass


class Preprocessor(ABC):
    """
    An abstract base class for preprocessing data.

    This class defines a callable interface for preprocessing operations,
    allowing derived classes to implement specific preprocessing techniques.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Perform the preprocessing operation on the input data.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The preprocessed data.

        Must be implemented by subclasses.
        """
        pass


class ImageNormalizer(Preprocessor):
    """
    A concrete implementation of the Preprocessor class for normalizing images.

    This class normalizes image pixel values to the range [0, 1] by dividing
    by 255.0.
    """

    def __call__(self, img):
        """
        Normalize the input image.

        Args:
            img (ndarray): The image to normalize.

        Returns:
            ndarray: The normalized image with pixel values in the range [0, 1].
        """
        return img / 255.0


class WordVectorizer(Preprocessor):
    def __init__(self, vocab_size: int):
        self.size = vocab_size

    def __call__(self, dataset: np.ndarray[List[int]]):
        m = dataset.shape[0]
        vectorized_form = np.zeros((m, self.size))
        for i, data in enumerate(dataset):
            vectorized_form[i, data] = 1.0
        return vectorized_form


class DataLoader:
    """
    A class for loading batches of data from a dataset.

    This class provides functionality to load data in batches, shuffle the dataset,
    and handle cross-validation data.

    Attributes:
        _dataset (Dataset): The dataset object from which data will be loaded.
        _batch_size (Optional[int]): The number of samples per batch.
        seed (int): The seed for random number generation used for shuffling.
        indices (List[int]): The indices for the dataset, used when shuffling.
        counter (int): A counter to track the current batch index.
        report (bool): Whether to print progress reports during loading.
        partitions (List[Tuple[Tensor, Tensor]]): The data partitions for batching.
    """

    _dataset: Dataset
    _batch_size: Optional[int]

    def __init__(self,
                 dataset: Dataset,
                 *,
                 batch_size: Optional[int] = 1,  # Stochastic gradient descent by default
                 shuffle: bool = False,
                 seed: int = 101,
                 report: bool = True):

        self._dataset = dataset
        self._batch_size = batch_size
        self.seed = seed
        self.indices = []

        if shuffle:
            rng = Random()
            rng.seed(seed)
            self._shuffle(rng)

        self.counter = 0
        self.report = report
        self.partitions = self._make_partition()

    def _shuffle(self, rng):
        """
        Shuffle the indices of the dataset.

        Args:
            rng (Random): The random number generator for shuffling.
        """
        data_length = len(self._dataset)
        self.indices = [ix for ix in range(data_length)]
        rng.shuffle(self.indices)

    def __iter__(self):
        """
        Reset the counter and return the iterator object.

        Returns:
            DataLoader: The iterator object.
        """
        self.reset_counter()
        return self

    def reset_counter(self):
        """
        Reset the counter to zero.
        """
        self.counter = 0

    @property
    def cv_data(self):
        """
        Get the cross-validation data from the dataset.

        Returns:
            Iterable: The cross-validation data.
        """
        return self._dataset.cv_data

    def _make_partition(self):
        """
        Create partitions of the dataset for batching.

        Returns:
            List[Tuple[Tensor, Tensor]]: A list of partitions containing
            input data and corresponding labels.
        """
        partitions: List[Tuple[Tensor, Tensor]] = []
        num_batch = len(self._dataset) // self._batch_size
        print(f"Loading batches to {backend.device}")  # Assuming backend is defined elsewhere

        for batch in range(num_batch):
            start_ind = batch * self._batch_size
            end_ind = (batch + 1) * self._batch_size

            samples = [self._dataset[i] for i in self.indices[start_ind:end_ind]]
            x_list = [sample[0] for sample in samples]
            y_list = [sample[1] for sample in samples]

            x = np.concatenate(x_list, axis = 0)
            y = np.stack(y_list, axis = 0).reshape(-1, 1)

            x = Tensor(x)
            y = Tensor(y)
            partition = (x, y)

            partitions.append(partition)

            if (batch + 1) % 100 == 0 and self.report:
                print('.', end = '', flush = True)

            if (batch + 1) % 5000 == 0 and self.report:
                print(end = "\n")

        print("\nNumber of batches loaded:", num_batch)
        return partitions

    @property
    def cv(self):
        """
        Get the cross-validation data.

        Returns:
            Iterable: The cross-validation data.
        """
        return self._dataset.cv_data

    def __next__(self):
        """
        Get the next batch of data.

        Returns:
            Tuple[Tensor, Tensor]: The next batch partition containing
            input data and labels.

        Raises:
            StopIteration: If there are no more batches to return.
        """
        if self.counter > len(self.partitions) - 1:
            raise StopIteration

        partition = self.partitions[self.counter]
        self.counter += 1
        return partition


class MNIST(Dataset):
    x_train: Optional[np.ndarray] = None
    x_cv: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_cv: Optional[np.ndarray] = None

    def __init__(self,
                 filename: str,
                 preprocessor: Optional[Preprocessor] = None,
                 *,
                 cv: bool = False,
                 cv_size: float = 0.2,
                 cv_seed: int = 101
                 ):

        try:
            dataset = np.load(filename)
        except Exception as exp:
            raise PsfitException(str(exp))

        self.num_pixels = self.number_of_pixels

        x = dataset['training_data'].reshape(-1, self.num_pixels)
        y = dataset['training_labels'].reshape(-1, 1)

        self.x_test = dataset["test_data"].reshape(-1, self.num_pixels)

        if preprocessor:
            x = preprocessor(x)
            self.x_test = preprocessor(self.x_test)

        if cv:
            assert cv_size > 0, "cv size cannot be negative."
            assert cv_size < 1, "cv size must be between 0 and 1"
            self.cv_size = cv_size

        if cv:
            x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size = self.cv_size, random_state = cv_seed)
            self.x_train = x_train
            self.x_cv = x_cv
            self.y_train = y_train
            self.y_cv = y_cv
        else:
            self.x_train = x
            self.y_train = y

    def __getitem__(self, index):
        try:
            img = self.x_train[index, :].reshape(1, self.num_pixels)

            label = self.y_train[index]

        except Exception as exp:
            raise PsfitException(str(exp))

        return img, label

    @property
    def img_size(self):
        return 28, 28

    @property
    def number_of_pixels(self):
        return np.prod(self.img_size)

    @property
    def number_of_features(self):
        return int(np.prod(self.x_train[0].shape))

    @property
    def number_of_samples(self):
        return self.__len__()

    @property
    def number_of_classes(self):
        return len(np.unique(self.y_train))

    @property
    def train_data(self):
        return self.x_train, self.y_train

    @property
    def cv_data(self):
        if self.x_cv is not None and self.y_cv is not None:
            return self.x_cv, self.y_cv
        return []

    def __len__(self):
        return self.x_train.shape[0]

    def __repr__(self):
        return f"MNIST()"


class Imdb(Dataset):
    def __init__(self,
                 filename: str,
                 *,
                 cv: bool = False,
                 cv_size: float = 0.2,
                 cv_seed: int = 101
                 ):

        try:
            dataset = np.load(filename, allow_pickle = True)
        except Exception as exp:
            raise PsfitException(str(exp))

        x = dataset['x-train']
        y = dataset['y-train'].reshape(-1, 1)
        self.x_test = dataset["x-test"]
        self.y_test = dataset["y-test"].reshape(-1, 1)
        self.word_index = dataset['index'].item()

        preprocessor = WordVectorizer(vocab_size = 10000)

        x = preprocessor(x)
        self.x_test = preprocessor(self.x_test)

        if cv:
            assert cv_size > 0, "cv size cannot be negative."
            assert cv_size < 1, "cv size must be between 0 and 1"
            self.cv_size = cv_size

        if cv:
            x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size = self.cv_size, random_state = cv_seed)
            self.x_train = x_train
            self.x_cv = x_cv
            self.y_train = y_train
            self.y_cv = y_cv
        else:
            self.x_train = x
            self.y_train = y

    @property
    def number_of_features(self):
        return self.x_train.shape[1]

    @property
    def number_of_samples(self):
        return self.x_train.shape[0]

    @property
    def train_data(self):
        return self.x_train, self.y_train

    @property
    def cv_data(self):
        if self.x_cv is not None and self.y_cv is not None:
            return self.x_cv, self.y_cv
        return []

    @property
    def number_of_classes(self):
        return 2

    def __getitem__(self, index):
        try:
            sample = self.x_train[index, :].reshape(1, self.number_of_features)

            label = self.y_train[index]

        except Exception as exp:
            raise PsfitException(str(exp))

        return sample, label

    def __len__(self):
        return self.x_train.shape[0]

    def get_word_index(self):
        return self.word_index

    def decode_review(self, review: List[int]):
        reverse_index = {value: key for key, value in self.word_index.items()}
        return " ".join([reverse_index.get(i - 3, "?") for i in review])

    def encode_review(self, review: List[str]):
        return [self.word_index[word] + 3 for word in review]

    def __repr__(self):
        return f"IMDB()"


class RandomRegressionDataset(Dataset):

    def __init__(self,
                 m: int,
                 n: int,
                 *,
                 variance: float = 0.0,
                 seed: int = 101,
                 sparsity: bool = False):
        self.n = n
        self.m = m
        self.X = np.random.randn(m, n)
        self.variance = variance
        self.y: Optional[np.ndarray] = None

    def populate(self, w_true):
        noise = np.sqrt(self.variance) * np.random.randn(self.m, 1)
        self.y = self.X @ w_true + noise

    @property
    def number_of_classes(self):
        return 1

    @property
    def number_of_features(self):
        return self.n

    @property
    def number_of_samples(self):
        return self.m

    @property
    def train_data(self):
        return self.X, self.y

    @property
    def cv_data(self):
        return []

    def __getitem__(self, item):
        return self.X[item, :].reshape(1, self.number_of_features), self.y[item]

    def __len__(self):
        return self.m


class Partition(Dataset):
    """
    A class that represents a partition of a dataset, allowing access to
    specific indices of the underlying dataset. This class supports both
    classification and regression datasets.

    Attributes:
        dataset (Dataset): The original dataset from which this partition is created.
        index (List[int]): A list of indices corresponding to the partition of the dataset.
    """

    def __init__(self, dataset: Dataset, index: List[int]):
        """
        Initializes the Partition with a dataset and a list of indices.

        Parameters:
            dataset (Dataset): The dataset to be partitioned.
            index (List[int]): The list of indices representing the subset of the dataset.
        """
        self.dataset = dataset
        self.index = index

    @property
    def number_of_classes(self):
        """
        Returns the number of classes in the dataset if it is a classification dataset.

        Raises:
            PsfitException: If the dataset is a regression dataset,
                            since it does not have classes.
        """
        if isinstance(self.dataset, Dataset):
            return self.dataset.number_of_classes
        raise PsfitException("Invalid dataset type")

    @property
    def number_of_features(self):
        """
        Returns the number of features in the dataset.

        Returns:
            int: The number of features in the dataset.
        """
        return self.dataset.number_of_features

    @property
    def number_of_samples(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.dataset.number_of_samples

    @property
    def train_data(self):
        """
        Returns the training data from the original dataset.

        Returns:
            Any: The training data of the dataset.
        """
        return self.dataset.train_data

    @property
    def cv_data(self):
        """
        Returns the cross-validation data from the original dataset.

        Returns:
            Any: The cross-validation data of the dataset.
        """
        return self.dataset.cv_data

    def __len__(self):
        """
        Returns the number of samples in the partition.

        Returns:
            int: The size of the partition.
        """
        if self.index:
            return len(self.index)
        return 0

    def __getitem__(self, index: int):
        """
        Retrieves a sample from the partition using the given index.

        Parameters:
            index (int): The index of the sample to retrieve.

        Returns:
            Any: The sample corresponding to the given index in the partition.
        """
        data_index = self.index[index]
        return self.dataset[data_index]

    def __repr__(self):
        """
        Returns a string representation of the Partition.

        Returns:
            str: A representation of the Partition with its size.
        """
        return f"DataPartition(size={self.__len__()})"


class DataPartitioner:
    """
    A class that partitions a dataset into N equally sized partitions.
    The partitions are created by shuffling the dataset indices randomly.

    Attributes:
        dataset (Dataset): The original dataset to be partitioned.
        N (int): The number of partitions to create.
        sizes (List[float]): The list of sizes for each partition.
        partition_indices (List[List[int]]): A list of indices for each partition.
    """

    def __init__(self,
                 dataset: Dataset,
                 N,
                 *,
                 seed: int = 101):
        """
        Initializes the DataPartitioner with a dataset, the number of partitions,
        and an optional random seed for reproducibility.

        Parameters:
            dataset (Dataset): The dataset to be partitioned.
            N (int): The number of partitions to create.
            seed (int, optional): The random seed for shuffling the dataset. Default is 101.
        """
        self.N = N
        self.sizes = [1.0 / N for _ in range(self.N)]
        self.dataset = dataset
        self.partition_indices = []

        rng = Random()
        rng.seed(seed)
        data_length = len(dataset)
        indices = [x for x in range(0, data_length)]
        rng.shuffle(indices)

        for size in self.sizes:
            part_len = int(size * data_length)
            self.partition_indices.append(indices[0: part_len])
            indices = indices[part_len:]

    def number_of_partitions(self):
        """
        Returns the number of partitions created by the DataPartitioner.

        Returns:
            int: The number of partitions.
        """
        return len(self.sizes)

    def partition(self):
        """
        Creates and returns the list of Partition objects based on the partition indices.

        Returns:
            List[Partition]: A list of Partition objects created from the dataset.
        """
        return [Partition(self.dataset, self.partition_indices[i]) for i in range(self.N)]


class ArrayDataset(Dataset):
    X: Optional[NDArray] = None  # feature matrix
    y: Optional[NDArray] = None  # response vector

    def __init__(self, X: NDArray, y: NDArray):
        assert X.shape[0] == y.shape[0], "data mismatch dimension"
        self._m, self._n = X.shape
        self.X = X
        self.y = y.reshape(-1, 1)

    @property
    def number_of_features(self):
        return self._n

    @property
    def number_of_samples(self):
        return self._m

    @property
    def train_data(self):
        return self.X, self.y

    @property
    def cv_data(self):
        return []

    @property
    def number_of_classes(self):
        return

    def __getitem__(self, item):
        return self.X[item, :], self.y[item]

    def __len__(self):
        return self._m


def get_loader(dataset: Dataset, batch_size: int, random_seed: int = 101, report: bool = True):
    return DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      shuffle = True,
                      seed = random_seed,
                      report = report)

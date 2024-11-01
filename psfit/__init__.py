"""
psfit - Parallel Sparse Fitting Toolbox for Distributed and Centralized Sparse Regression and Classification.

This package provides tools to solve sparse regression and classification problems with L0-norm sparsity constraints,
 using the distributed Alternating Direction Method of Multipliers (ADMM) algorithm.  The toolbox is designed for both 
 centralized and distributed computing environments and leverages the Ray distributed computing framework.

Key Features:
--------------
- **Sparse Regression and Classification**: Efficiently solve sparse problems with L0-norm sparsity.
- **Distributed ADMM Algorithm**: Supports large-scale distributed computation across multiple nodes.
- **Autodiff Module**: Custom automatic differentiation module for efficient gradient computations.
- **Backend Support**: Flexible backend configuration with support for CPU and GPU environments.
- **Ray Integration**: Uses Ray as the distributed computing framework for scalability.

Modules:
---------
- `autodiff`: Provides tools for automatic differentiation.
- `data`: Handles dataset input and preprocessing for the toolbox.
- `exceptions`: Defines custom exceptions used across the package.
- `model`: Defines the model structures for regression and classification tasks.
- `module`: Houses core functionality and submodules of the toolbox.
- `ops`: Contains operations used internally for computation.
- `optim`: Optimization routines, including distributed and centralized optimizers.



Developer:
---------
Author: Alireza Olama
Github: https://github.com/alirezalm
Email: alireza.lm69@gmail.com


License:
---------
License: MIT

"""

from .autodiff import *
from .autodiff import Tensor as tensor
from .model import SparseSoftmaxClassifier
from .module import Linear, SoftmaxLoss, MSELoss
from .ops import *
from .ops import summation as sum
from .data import *
from .optim.component import *
from .optim.distributed import *
from .optim.optimizer import *

__version__ = "0.1.1"

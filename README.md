#### Parallel Sparse Fitting Toolbox (PsFit)
![PsFit Logo](media/PsFiT.png)

***

PsFiT is an open-source Python framework for distributed sparse model training, optimized for scalability with
GPU support. Using the DISADMM algorithm, PsFiT enables efficient L0 norm constrained learning for tasks like
linear and logistic regression across multiple nodes.


### Features
- **Distributed Sparse Model Training**: Enables efficient L0-norm constrained learning across multiple nodes.
- **Scalability**: Optimized for both single-node and multi-node setups with support for GPU acceleration.
- **Flexible Model Support**: Includes pre-built modules for linear regression, logistic regression, and softmax regression.
- **User-Friendly API**: Provides intuitive APIs for easy integration and customization.


### Architecture
![PsFit Architecture](media/psfit_arch.png)


| **Module**        | **Description**                                                                                             |
|-------------------|-------------------------------------------------------------------------------------------------------------|
| **Model**         | Provides pre-built models for sparse regression tasks, including softmax, logistic, and linear regression, enabling easy model selection for different learning tasks. |
| **Optimizer**     | Contains various optimization algorithms and tools, including centralized and distributed ADMM, SGD, and dual solvers, to manage model training efficiently in different environments. |
| **Data**          | Manages dataset operations, including loading, partitioning, and preprocessing, to facilitate distributed data handling and preparation for training. |
| **Module**        | Contains model-specific components like loss functions for linear, logistic, and softmax regression, allowing for flexible customization of training objectives. |
| **Autodiff**      | Implements automatic differentiation, with custom PsFiT tensors and operations, essential for gradient-based optimization in sparse model training. |
| **Backend**       | Provides support for GPU and CPU computing environments, enabling flexible deployment based on hardware availability. |


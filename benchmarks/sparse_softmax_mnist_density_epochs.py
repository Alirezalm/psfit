import os
import time

os.environ["backend"] = "pytorch"
from psfit.optim import ProjectionSparsifier
from psfit import tensor

from psfit.data import MNIST, ImageNormalizer
import matplotlib.pyplot as plt

from psfit.model import SparseSoftmaxClassifier
from sklearn.metrics import accuracy_score

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True


def main():
    dataset = MNIST("./data/mnist-data.npz", preprocessor = ImageNormalizer(), cv = True)

    iters = 100

    classifier = SparseSoftmaxClassifier(
        dataset = dataset,
        batch_size = 10000,
        learning_rate = 0.5,
        penalty = 0.1,
        local_epochs = 5,
        admm_iterations = iters,
        validation_accuracy = True,
        training_accuracy = True,
        verbose = False
    )
    # Assuming dataset and classifier are already defined
    x_train, y_train = dataset.train_data
    x_cv, y_cv = dataset.cv_data

    x_train = tensor(x_train)
    x_cv = tensor(x_cv)

    densities = [20, 60, 100, 200, 300, 400, 500, 600, 700, 784]
    local_epochs = [5, 10, 20]

    training_accuracies = {epoch: [] for epoch in local_epochs}
    validation_accuracies = {epoch: [] for epoch in local_epochs}
    loss = {epoch: [] for epoch in local_epochs}
    residual = {epoch: [] for epoch in local_epochs}
    execution_times = {epoch: [] for epoch in local_epochs}  # To store execution times

    # Loop over both densities and local_epochs
    for density in densities:
        for epoch in local_epochs:
            classifier.local_epoch = epoch

            # Start timing
            start_time = time.time()

            # Fit the classifier
            training_loss_values, primal_error_values = classifier.fit(
                sparsifier = ProjectionSparsifier(density = density),
                warm_start = False
            )

            # Stop timing
            elapsed_time = time.time() - start_time
            execution_times[epoch].append(elapsed_time)  # Store execution time

            residual[epoch].append(primal_error_values[-1])
            loss[epoch].append(training_loss_values[-1])

            pred_validation = classifier.predict(x_cv)
            pred_training = classifier.predict(x_train)

            train_acc = accuracy_score(y_train, pred_training)
            validation_acc = accuracy_score(y_cv, pred_validation)

            training_accuracies[epoch].append(train_acc)
            validation_accuracies[epoch].append(validation_acc)

            # Print the report for each density and penalty
            print(f"Density: {density} | Local Epoch: {epoch} | "
                  f"Loss: {training_loss_values[-1]:.4f} | "
                  f"Residual: {primal_error_values[-1]:.4f} | "
                  f"Training Accuracy: {train_acc:.4%} | "
                  f"Validation Accuracy: {validation_acc:.4%} | "
                  f"Execution Time: {elapsed_time:.4f} seconds")

        print()

    colors = ['#d62728', '#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'D']
    # Set up a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize = (8, 5))

    # Subplot 1: Loss vs. Densities (for each epoch)
    for i, epoch in enumerate(local_epochs):
        axs[0, 0].plot(densities, loss[epoch], linestyle = '-', color = colors[i % len(colors)],
                       label = f"epochs = {epoch}", marker = markers[i % len(markers)], markersize = 4)

    axs[0, 0].grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
    axs[0, 0].set_xlabel("$\\kappa$", fontsize = 12)
    axs[0, 0].set_ylabel("Loss", fontsize = 12)
    axs[0, 0].legend()

    # Subplot 2: Training Accuracy vs. Densities (for each epoch)
    for i, epoch in enumerate(local_epochs):
        axs[0, 1].plot(densities, training_accuracies[epoch], linestyle = '-', color = colors[i % len(colors)],
                       label = f"epochs = {epoch}", marker = markers[i % len(markers)], markersize = 4)

    axs[0, 1].set_xlabel("$\\kappa$", fontsize = 12)
    axs[0, 1].set_ylabel("Training Accuracy", fontsize = 12)
    axs[0, 1].legend()
    axs[0, 1].grid(True, which = 'both', linestyle = '--', linewidth = 0.5)

    # Subplot 3: Validation Accuracy vs. Densities (for each epoch)
    for i, epoch in enumerate(local_epochs):
        axs[1, 0].plot(densities, validation_accuracies[epoch], linestyle = '-', color = colors[i % len(colors)],
                       label = f"epochs = {epoch}", marker = markers[i % len(markers)], markersize = 4)

    axs[1, 0].set_xlabel("$\\kappa$", fontsize = 12)
    axs[1, 0].set_ylabel("Validation Accuracy", fontsize = 12)
    axs[1, 0].legend()
    axs[1, 0].grid(True, which = 'both', linestyle = '--', linewidth = 0.5)

    # Subplot 4: Residuals vs. Densities (for each epoch)
    for i, epoch in enumerate(local_epochs):
        axs[1, 1].plot(densities, residual[epoch], linestyle = '-', color = colors[i % len(colors)],
                       label = f"epochs = {epoch}", marker = markers[i % len(markers)], markersize = 4)
    axs[1, 1].set_xlabel("$\\kappa$", fontsize = 12)
    axs[1, 1].set_ylabel("Residual", fontsize = 12)
    axs[1, 1].legend()
    axs[1, 1].grid(True, which = 'both', linestyle = '--', linewidth = 0.5)

    # Adjust layout
    plt.tight_layout()

    plt.savefig(f"./figures/local_epochs_density_{dataset.__class__.__name__.lower()}.pdf",
                dpi = 300)
    plt.show()


if __name__ == '__main__':
    main()

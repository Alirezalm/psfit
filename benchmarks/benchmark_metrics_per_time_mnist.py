import os

os.environ['backend'] = 'pytorch'
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

import psfit as pf
from psfit.data import MNIST, ImageNormalizer, DataLoader
from psfit.module import SoftmaxLoss, Linear
from psfit.optim.component import ProjectionSparsifier
from psfit.optim.optimizer import Admm, SGD, LocalTrainer

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

backend = pf.get_backend()

mark_every = 20


def predict(model, x):
    logits = model(x).detach().array().cpu()
    predictions = np.argmax(logits, axis = 1, keepdims = True)
    return predictions


def main():
    data_path = "./data/mnist-data.npz"

    dataset = MNIST(filename = data_path, preprocessor = ImageNormalizer(), cv = True, cv_size = 0.1)

    data_loader = DataLoader(dataset = dataset, batch_size = len(dataset) // 3, shuffle = True)

    print("loading cross-validation and test data...")
    x_train, y_train = pf.tensor(dataset.train_data[0]), dataset.train_data[1]
    x_validation, y_validation = pf.tensor(dataset.cv_data[0]), dataset.cv_data[1]
    x_test = pf.tensor(dataset.x_test)

    print(f"{dataset} loaded successfully on {backend.device} backend")

    n, c = dataset.number_of_features, dataset.number_of_classes
    model = Linear(in_features = n, out_features = c)

    print("number of input features:", n)
    print("number of output features:", c)
    densities = [20, 30, 40]

    training_loss = {density: [] for density in densities}
    training_accuracy = {density: [] for density in densities}
    validation_accuracy = {density: [] for density in densities}
    primal_error_values = {density: [] for density in densities}

    maximum_training_time = 50  # seconds

    admm_penalty = 0.1

    learning_rate = 0.3

    trainer = LocalTrainer(
        dataloader = data_loader,
        loss = SoftmaxLoss(),
        optimizer = SGD(model, learning_rate = learning_rate),
        verbose = False
    )

    for density in densities:

        model.init_params()

        epoch = 0
        optimizer = Admm(trainer = trainer, aggregators = ProjectionSparsifier(density = density))

        start = time.time()
        while time.time() - start < maximum_training_time:
            loss, err = optimizer.step(penalty = admm_penalty)
            training_loss[density].append(loss)

            pred_training = predict(model, x_train)
            pred_validation = predict(model, x_validation)

            train_acc = accuracy_score(y_train, pred_training)
            cv_acc = accuracy_score(y_validation, pred_validation)

            training_accuracy[density].append(train_acc)
            validation_accuracy[density].append(cv_acc)
            primal_error_values[density].append(err)

            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1:3d}: Loss = {loss:.4f}, Error = {err:.6f} "
                    f"Training = {train_acc:.6f}, Validation = {cv_acc:.6f}, Density: {density}")

            epoch += 1
        print()

    minimum = 1e10
    for key, value in training_accuracy.items():
        minimum = min(minimum, len(value))

    colors = ['#d62728', '#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'D']
    xaxis = np.linspace(0, maximum_training_time, minimum)
    # Create a 2x2 subplot grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (8, 5))

    # Training Loss Plot
    for i, density in enumerate(densities):
        ax1.plot(xaxis, training_loss[density][0:minimum], linestyle = '-', color = colors[i % len(colors)],
                 marker = markers[i % len(markers)], label = f'$\\kappa$ = {density}', markersize = 4,
                 markevery = mark_every)

    ax1.set_xlabel('Time (seconds)', fontsize = 12)
    ax1.set_ylabel('Training Loss', fontsize = 12)
    ax1.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
    ax1.legend(loc = 'best', fontsize = 12)
    ax1.set_xscale('log')

    # Primal Error Plot
    for i, density in enumerate(densities):
        ax2.plot(xaxis, primal_error_values[density][0:minimum], linestyle = '-', color = colors[i % len(colors)],
                 marker = markers[i % len(markers)], label = f'$\\kappa$ = {density}', markersize = 4,
                 markevery = mark_every)

    ax2.set_xlabel('Time (seconds)', fontsize = 12)
    ax2.set_ylabel('Primal Error', fontsize = 12)
    ax2.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
    ax2.legend(loc = 'best', fontsize = 12)
    ax2.set_xscale('log')

    # Training Accuracy Plot
    for i, density in enumerate(densities):
        ax3.plot(xaxis, training_accuracy[density][0:minimum], linestyle = '-', color = colors[i % len(colors)],
                 marker = markers[i % len(markers)], label = f'$\\kappa$ = {density}', markersize = 4,
                 markevery = mark_every)

    ax3.set_xlabel('Time (seconds)', fontsize = 12)
    ax3.set_ylabel('Training Accuracy', fontsize = 12)
    ax3.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
    ax3.legend(loc = 'best', fontsize = 12)
    ax3.set_xscale('log')

    # Validation Accuracy Plot
    for i, density in enumerate(densities):
        ax4.plot(xaxis, validation_accuracy[density][0:minimum], linestyle = '-', color = colors[i % len(colors)],
                 marker = markers[i % len(markers)], label = f'$\\kappa$ = {density}', markersize = 4,
                 markevery = mark_every)

    ax4.set_xlabel('Time (seconds)', fontsize = 12)
    ax4.set_ylabel('Validation Accuracy', fontsize = 12)
    ax4.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
    ax4.legend(loc = 'best', fontsize = 12)
    ax4.set_xscale('log')

    # Apply tight layout for better spacing between subplots
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        f"./figures/training_validation_loss_error_time_{dataset.__class__.__name__.lower()}_{backend.device}.pdf",
        dpi = 300)

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()

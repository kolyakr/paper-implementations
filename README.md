# Paper Implementations

This repository contains my implementations of foundational deep learning papers, focusing on classic architectures and experimental variations. The project is designed as a modular framework for training, evaluating, and visualizing these models.

## Project Structure

The repository is organized into several key modules:

* **`models/`**: Implementation of various neural network architectures.
* **`engine/`**: Core logic for experimentation.
  * **`Trainer.py`**: A robust `Trainer` class that manages the training loop, validation, learning rate scheduling, and performance logging with a formatted CLI output.
  * **`DatasetProvider.py`**: Handles data loading and preprocessing.
* **`notebooks/`**: Jupyter notebooks for running experiments on specific datasets like MNIST and CIFAR-10.
* **`custom/`**: Custom layers and modules
* **`papers/`**: Original PDF research papers for reference.
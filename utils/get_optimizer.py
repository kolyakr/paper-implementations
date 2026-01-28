import torch

def get_optimizer(optimizer_name: str, model_parameters, lr=0.001):
    optimizers = {
        "sgd": torch.optim.SGD(
            params=model_parameters,
            lr=lr
        ),
        "adam": torch.optim.Adam(
            params=model_parameters,
            lr=lr
        ),
        "rmsprop": torch.optim.RMSprop(
            params=model_parameters,
            lr=lr
        )
    }

    return optimizers.get(optimizer_name.lower(), torch.optim.Adam(
            params=model_parameters,
            lr=lr
        ))
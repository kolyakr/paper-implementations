import torch

def get_optimizer(optimizer_name: str, model_parameters, lr=0.001, **kwargs):
    params = list(model_parameters)
    if not params:
        raise ValueError("Model parameters are empty.")

    name = optimizer_name.lower()
    
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, **kwargs)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, **kwargs)
    else:
        return torch.optim.Adam(params, lr=lr, **kwargs)
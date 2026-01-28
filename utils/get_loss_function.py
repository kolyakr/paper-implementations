from torch import nn

def get_loss_function(loss_name: str):
    losses = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "mse": nn.MSELoss(),
        "L1": nn.L1Loss()
    }

    return losses.get(loss_name.lower(), nn.CrossEntropyLoss())
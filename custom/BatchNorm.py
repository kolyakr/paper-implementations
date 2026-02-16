import torch
from torch import nn

class BatchNorm(nn.Module):
    def __init__(self, num_features: int, num_dims: int, eps: float = 1e-5, momentum: float | None = 0.1,):
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        if num_dims == 2:
            shape = (1, num_features)
            self.reduction_shape = (0, )
        else:
            shape = (1, num_features, 1, 1)
            self.reduction_shape = (0, 2, 3)

        self.gamma = nn.Parameter(torch.ones(shape), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(shape), requires_grad=True)

        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))

    def forward(self, X):
        if not self.training:
            return self.gamma * ((X - self.running_mean) / 
                                 torch.sqrt(self.running_var + self.eps)) + self.beta
        
        batch_mean = X.mean(dim=self.reduction_shape, keepdim=True)
        batch_var = X.var(keepdim=True, dim=self.reduction_shape, unbiased=False)

        with torch.no_grad():
            unbiased_var = X.var(keepdim=True, dim=self.reduction_shape, unbiased=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean 
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbiased_var 

        norm_X = (X - batch_mean) / torch.sqrt(batch_var + self.eps)
        
        return self.gamma * norm_X + self.beta
    
class BatchNormMean(nn.Module):
    def __init__(self, num_features: int, num_dims: int, eps: float = 1e-5, momentum: float | None = 0.1,):
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        if num_dims == 2:
            shape = (1, num_features)
            self.reduction_shape = (0, )
        else:
            shape = (1, num_features, 1, 1)
            self.reduction_shape = (0, 2, 3)

        self.beta = nn.Parameter(torch.zeros(shape), requires_grad=True)

        self.register_buffer('running_mean', torch.zeros(shape))

    def forward(self, X):
        if not self.training:
            return (X - self.running_mean) + self.beta
        
        batch_mean = X.mean(dim=self.reduction_shape, keepdim=True)

        with torch.no_grad():
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean 

        norm_X = X - batch_mean
        
        return norm_X + self.beta
    
class BatchNormVar(nn.Module):
    def __init__(self, num_features: int, num_dims: int, eps: float = 1e-5, momentum: float | None = 0.1,):
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        if num_dims == 2:
            shape = (1, num_features)
            self.reduction_shape = (0, )
        else:
            shape = (1, num_features, 1, 1)
            self.reduction_shape = (0, 2, 3)

        self.gamma = nn.Parameter(torch.ones(shape), requires_grad=True)

        self.register_buffer('running_var', torch.ones(shape))

    def forward(self, X):
        if not self.training:
            return self.gamma * (X / torch.sqrt(self.running_var + self.eps)) 
        
        batch_var = X.var(keepdim=True, dim=self.reduction_shape, unbiased=False)

        with torch.no_grad():
            unbiased_var = X.var(keepdim=True, dim=self.reduction_shape, unbiased=True)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbiased_var 

        norm_X = X / torch.sqrt(batch_var + self.eps)
        
        return self.gamma * norm_X
    
class BatchNormFixed(BatchNorm):
    def __init__(self, num_features, num_dims):
        super().__init__(num_features, num_dims)
        self.gamma.requires_grad = False
        self.beta.requires_grad = False
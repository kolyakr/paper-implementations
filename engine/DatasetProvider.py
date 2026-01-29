import torch 
import torchvision
import torchvision.transforms as transforms
from enum import Enum

class DatasetName(Enum):
    MNIST = "mnist"
    CIFAR = "cifar10"
    FASHION_MNIST = "fashion_mnist"

class DatasetProvider:
    def __init__(self, dataset_name: DatasetName, batch_size: int, padding: int = 0, resize = 0):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.padding = padding
        self.resize = resize
        
        self.trainloader = None
        self.testloader = None

        self.prepare()

    def prepare(self):
        if self.dataset_name == DatasetName.MNIST:
            ds_class = torchvision.datasets.MNIST
            normalize_params = ((0.5,), (0.5,))
        elif self.dataset_name == DatasetName.CIFAR:
            ds_class = torchvision.datasets.CIFAR10
            normalize_params = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif self.dataset_name == DatasetName.FASHION_MNIST:
            ds_class = torchvision.datasets.FashionMNIST
            normalize_params = ((0.5,), (0.5,))

        transform_list = []

        if self.resize != 0:
            transform_list.append(transforms.Resize((self.resize, self.resize)))
        elif self.padding != 0:
            transform_list.append(transforms.Pad(padding=self.padding))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(*normalize_params))

        transform = transforms.Compose(transform_list)

        train_data = ds_class(root="../data", train=True, download=True, transform=transform)
        test_data = ds_class(root="../data", train=False, download=True, transform=transform)

        self.trainloader = torch.utils.data.DataLoader(
            dataset=train_data, 
            shuffle=True, 
            batch_size=self.batch_size,
            num_workers=2
        )

        self.testloader = torch.utils.data.DataLoader(
            dataset=test_data, 
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=2
        )
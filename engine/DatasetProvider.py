import torch 
import torchvision
import torchvision.transforms as transforms
from enum import Enum
import copy
import numpy as np
class DatasetName(Enum):
    MNIST = "mnist"
    CIFAR = "cifar10"
    FASHION_MNIST = "fashion_mnist"

TRAIN_RATIO = 0.8
class DatasetProvider:
    def __init__(
            self, 
            dataset_name: DatasetName, 
            batch_size: int, 
            padding=None, 
            resize=None,
            random_rotation_degrees=None,
            random_hor_flip_probability=None,
            random_crop_size=None 
        ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.padding = padding
        self.resize = resize
        self.random_rotation_degrees = random_rotation_degrees
        self.random_hor_flip_probability = random_hor_flip_probability
        self.random_crop_size = random_crop_size
        
        self.trainloader = None
        self.testloader = None
        self.validloader = None
        self.classes = None

        self.prepare()

    def prepare(self):
        if self.dataset_name == DatasetName.MNIST:
            ds_class = torchvision.datasets.MNIST
        elif self.dataset_name == DatasetName.CIFAR:
            ds_class = torchvision.datasets.CIFAR10
        elif self.dataset_name == DatasetName.FASHION_MNIST:
            ds_class = torchvision.datasets.FashionMNIST

        preview_dataset = ds_class(
            root="../../data",
            train=True,
            download=True
        )

        data = preview_dataset.data

        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)

        data = data.float() / 255

        means = data.mean(dim=(0, 1, 2)) 
        stds = data.std(dim=(0, 1, 2))

        train_transform_list = []
        test_transform_list = []

        if self.resize is not None:
            train_transform_list.append(transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.BILINEAR))
        if self.padding is not None:
            train_transform_list.append(transforms.Pad(padding=self.padding))
        if self.random_rotation_degrees is not None:
            train_transform_list.append(transforms.RandomRotation(degrees=self.random_rotation_degrees))
        if self.random_hor_flip_probability is not None:
            train_transform_list.append(transforms.RandomHorizontalFlip(p=self.random_hor_flip_probability))
        if self.random_crop_size is not None:
            train_transform_list.append(transforms.RandomCrop(size=(self.random_crop_size, self.random_crop_size)))

        train_transform_list.append(transforms.ToTensor())
        train_transform_list.append(transforms.Normalize(mean=means, std=stds))

        train_transform = transforms.Compose(train_transform_list)

        if self.resize is not None and self.random_crop_size is not None:
            test_transform_list.append(transforms.Resize(size=(self.random_crop_size, self.random_crop_size), interpolation=transforms.InterpolationMode.BILINEAR))
        elif self.resize is not None:
            test_transform_list.append(transforms.Resize(size=(self.resize, self.resize), interpolation=transforms.InterpolationMode.BILINEAR))

        test_transform = transforms.Compose([
            *test_transform_list,
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])

        train_data = ds_class(root="../../data", train=True, download=True, transform=train_transform)
        test_data = ds_class(root="../../data", train=False, download=True, transform=test_transform)

        self.classes = test_data.classes

        n_train_examples = int(len(train_data) * TRAIN_RATIO)
        n_valid_examples = len(train_data) - n_train_examples

        train_data, valid_data = torch.utils.data.random_split(
            train_data, 
            [n_train_examples, n_valid_examples]
        )

        valid_data = copy.deepcopy(valid_data)
        valid_data.dataset.transform = test_transform

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

        self.validloader = torch.utils.data.DataLoader(
            dataset=valid_data, 
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=2
        )
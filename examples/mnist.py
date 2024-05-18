# basics
import os
import tempfile
import numpy as np

# KAN impl
from efficient_kan import KAN

# MNIST and Torch shenanigans
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ray tune
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint

# type hints
from typing import Dict

# partial functions
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes

        # Create a list of layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Not the last layer
                layers.append(nn.SiLU())

        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.layer_sizes[0])  # Flatten the input
        x = self.model(x)
        return x

# taken from https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def mnist_train(config: Dict):
    train_transform = transforms.Compose([
        transforms.ToTensor(), 
        AddGaussianNoise(0, config['std']), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=val_transform
    )
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=False)

    # get model shape via list comprehension
    model_shape = [config['in_dim']] + [config['hidden_dim'] for _ in range(config['num_hidden'])] + [config['out_dim']]
    # initialize model and move to accelerator
    match config['model']:
        case 'mlp':
            model = MLP(model_shape)
        case 'kan':
            model = KAN(model_shape)
        case other:
            raise ValueError(f"unrecognized model type: {other}")

    print("Model:")
    print(model)
    model.to(device)

    # optimizer initialization
    if config['use_adam']:
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay'], 
            betas=[config['beta1'], config['beta2']]
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            nesterov=True
        )
    criterion = nn.CrossEntropyLoss() 

    for i in range(config['num_epochs']):
        model.train()

        # train loop
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, config['in_dim']).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

        # validation loop
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, config['in_dim']).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if (i + 1) % 5 == 0:
                # This saves the model to the trial directory
                torch.save(
                    model.state_dict(),
                    os.path.join(temp_checkpoint_dir, "model.pth")
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Send the current training result back to Tune
            train.report({"val_accuracy": val_accuracy, "val_loss": val_loss}, checkpoint=checkpoint)


if __name__ == "__main__":
    # define hparam search space
    search_space = {
        # MODEL PARAMS
        "model": "mlp",

        # LR PARAMS
        # "lr": tune.grid_search([10**(-i) for i in range(1, 7)]),
        "lr": 0.001,
        "use_adam": True,
        "beta1": 0.9,
        "beta2": 0.99,
        "weight_decay": 1e-4,

        # ARCHITECTURE PARAMS
        "in_dim": 28*28,
        # "hidden_dim": tune.grid_search([2**i for i in range(4, 9)]),
        "hidden_dim": 64,
        "out_dim": 10,
        # "num_hidden": tune.grid_search([i for i in range(1, 6)]),
        "num_hidden": 1,

        # TRAINING PARAMS
        "num_epochs": 10,
        "batch_size": 64,

        # DATA TRANSFORM PARAMS
        "std": 0,
        # "std": tune.grid_search([0.1 * i for i in range(1, 11)]),
    }

    # configure train function
    # train_func = tune.with_parameters(mnist_train, trainloader=trainloader, valloader=valloader)
    train_func = mnist_train
    if torch.cuda.is_available():
        train_func = tune.with_resources(train_func, {"gpu": 1})

    tuner = tune.Tuner(
        train_func,
        param_space=search_space,
    )
    results = tuner.fit() 
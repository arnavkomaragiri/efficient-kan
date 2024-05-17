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

def mnist_train(config: Dict, trainloader: DataLoader = None, valloader: DataLoader = None):
    # get model shape via list comprehension
    model_shape = [config['in_dim']] + [config['hidden_dim'] for _ in range(config['num_hidden'])] + [config['out_dim']]
    # initialize model and move to accelerator
    model = KAN(model_shape)
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
                images = images.view(-1, 28 * 28).to(device)
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
                images = images.view(-1, 28 * 28).to(device)
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
    # Load MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    # define hparam search space
    search_space = {
        "use_adam": True,
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "beta1": 0.9,
        "beta2": 0.99,
        "weight_decay": 1e-4,
        "in_dim": 28*28,
        "hidden_dim": 64,
        "out_dim": 10,
        "num_hidden": 1,
        "num_epochs": 10
    }

    # configure train function
    train_func = partial(mnist_train, trainloader=trainloader, valloader=valloader)
    if torch.cuda.is_available:
        train_func = tune.with_resources(train_func, {"gpu": 1})

    tuner = tune.Tuner(
        train_func,
        param_space=search_space,
    )
    results = tuner.fit() 
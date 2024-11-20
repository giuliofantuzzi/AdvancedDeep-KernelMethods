
import os
import torch as th
from torchvision import datasets, transforms

def FashionMNIST_DataPrep(reduced_trainingset_size=None):

    mnist_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
    os.makedirs("./data/", exist_ok=True)
    train_dataset = datasets.FashionMNIST(
        root="./data/", train=True, transform=mnist_transforms, download=True
    )
    test_dataset = datasets.FashionMNIST(
        root="./data/", train=False, transform=mnist_transforms, download=True
    )

    X_train = train_dataset.data.to(th.float32)
    y_train = train_dataset.targets.to(th.long)
    X_test = test_dataset.data.to(th.float32)
    y_test = test_dataset.targets.to(th.long)

    # Scale values and center the data
    X_train = X_train/255.0
    X_train = X_train - X_train.mean(axis=0)
    X_test = X_test/255.0
    X_test = X_test - X_test.mean(axis=0)
    
    # Compute reduced training set if required
    if reduced_trainingset_size:
        X_train_reduced, y_train_reduced = X_train[:reduced_trainingset_size], y_train[:reduced_trainingset_size]
        # Ensure the reduced training set to be centered
        X_train_reduced = X_train_reduced - X_train_reduced.mean(axis=0)
        
        return X_train, y_train, X_test, y_test, X_train_reduced, y_train_reduced

    else:
        return X_train, y_train, X_test, y_test
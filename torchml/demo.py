"""Image classification demo using TorchML."""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
from torch.utils.data import DataLoader

import torchml as ml
import sklearn.naive_bayes as naive_bayes


def main():
    # Set seed
    torch.manual_seed(42)

    # Load the CIFAR10 dataset
    train_dataset = dsets.CIFAR10(
        root="./data",
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        download=True,
    )
    test_dataset = dsets.CIFAR10(
        root="./data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    # Create a dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=10000, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    # Retrieve the first batch of images and labels
    X_train = next(iter(train_loader))[0]
    y_train = next(iter(train_loader))[1]
    X_test = next(iter(test_loader))[0]
    y_test = next(iter(test_loader))[1]

    # Initialize a pretrained feature extractor (ResNet18)
    feature_extractor = models.resnet18(pretrained=True)

    # Extract features from the dataset
    with torch.no_grad():
        X_train_features = feature_extractor(X_train)
        X_test_features = feature_extractor(X_test)

    # Create the model
    model = ml.naive_bayes.GaussianNB()

    # Train the model
    model.fit(X_train_features, y_train)

    # Test the model
    preds = model.predict(X_test_features)
    accuracy = (preds == y_test).sum().item() / len(preds)
    print("Accuracy: {}".format(accuracy))

    # Compare with sklearn
    ref = naive_bayes.GaussianNB()
    ref.fit(X_train_features, y_train)
    ref_preds = ref.predict(X_test_features)
    ref_accuracy = (ref_preds == y_test.numpy()).sum().item() / len(ref_preds)
    print("Sklearn accuracy: {}".format(ref_accuracy))


if __name__ == "__main__":
    main()

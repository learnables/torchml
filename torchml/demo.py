"""Image classification demo using TorchML."""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchml as ml


def main():
    # Load the MNIST dataset
    train_dataset = dsets.MNIST(
        root="./data",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    test_dataset = dsets.MNIST(
        root="./data",
        train=False,
        transform=transforms.ToTensor(),
    )

    # Create the model
    model = ml.naive_bayes.GaussianNB()

    # Linearize the image data
    X_train = train_dataset.train_data
    X_train = X_train.view(X_train.size(0), -1)
    X_test = test_dataset.test_data
    X_test = X_test.view(X_test.size(0), -1)

    # Train the model
    model.fit(
        X_train.float(),
        train_dataset.train_labels.float(),
    )

    # Test the model
    preds = model.predict(X_test.float())
    accuracy = (preds == test_dataset.test_labels).sum().item() / len(preds)
    print("Accuracy: {}".format(accuracy))


if __name__ == "__main__":
    main()

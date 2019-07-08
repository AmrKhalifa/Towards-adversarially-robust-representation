from torchvision import datasets, transforms

train_set = datasets.MNIST(
    root='./data/MNIST',
    download=True,
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

test_set = datasets.MNIST(
    root='./data/MNIST_test',
    download=True,
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


def get_data():
    return train_set, test_set

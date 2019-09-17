import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import data_loader
from utils.viewer import show_batch
from mnist_classifier import NeuralModel, train_model, test_model
from classifier_attacks import test_attack
import matplotlib.pyplot as plt


def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    # delta = torch.zeros_like(X, requires_grad=True)
    # loss = nn.CrossEntropyLoss()(model(X + delta), y)
    X.requires_grad = True
    output = model(X)
    # output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output, y)

    loss.backward()
    # adv_noise = epsilon * delta.grad.detach().sign()
    adv_noise = epsilon * X.grad.detach().sign()
    return adv_noise


def pgd(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = F.nll_loss(model(X + delta), y)
        loss.backward()
        delta.data = (delta + X.shape[0] * alpha * delta.grad.data).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


def pgd_l2(model, X, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = F.nll_loss(model(X + delta), y)
        loss.backward()
        delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()

    return delta.detach()


def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = F.nll_loss(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


fgsm.__name__ = "FGSM Attack"
pgd.__name__ = "Projected Gradient Attack"
pgd_l2.__name__ = "Deep Fool Attack"
pgd_linf.__name__ = "iFGSM Attack"


def attack(model, device, loader, attack_method, epsilon, *args):
    print(attack_method.__name__ + " : ")

    model.to(device)

    correct = 0
    total_loss = 0

    for batch in loader:
        batch_images, batch_labels = batch

        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        delta = attack_method(model, batch_images, batch_labels, epsilon, *args)
        predictions = model(batch_images + delta)

        loss = F.nll_loss(predictions, batch_labels)
        correct += (predictions.max(dim=1)[1] == batch_labels).sum().item()
        total_loss += loss.item() * batch_images.shape[0]

    accuracy = correct / len(loader.dataset)
    loss = total_loss / len(loader.dataset)

    return accuracy, loss

    pass


def get_adv_examples(model, device, loader, attack_method, epsilon, *args):
    adv_examples = []

    model.to(device)

    for batch in loader:
        batch_images, batch_labels = batch

        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        deltas = attack_method(model, batch_images, batch_labels, epsilon, *args)

        adv_images = torch.clamp(batch_images + deltas, 0, 1)

        adv_examples.append((batch_images, adv_images, batch_labels))

    return adv_examples


def main():
    pass


if __name__ == "__main__":
    train_set, test_set = data_loader.get_data()
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    batch_size = 512
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    model = NeuralModel()

    trained_model = train_model(model, train_loader, epochs=10)

    print(attack(trained_model, device, test_loader, fgsm, 0.3)[0])
    print("=*" * 20)

    print(attack(trained_model, device, test_loader, pgd, 0.3, 1e-2, 40)[0])
    print("=*" * 20)

    print(attack(trained_model, device, test_loader, pgd_linf, 0.1, 1e-3, 40)[0])
    print("=*" * 20)

    print(attack(trained_model, device, test_loader, pgd_l2, 2, 0.3, 40)[0])

    train_data = get_adv_examples(trained_model, device, test_loader, pgd_l2, 2, 0.3, 40)

    print(len(train_data))

    train_iter = iter(train_data)
    a, b, c = next(train_iter)

    show_batch(a.cpu())
    show_batch(b.cpu())
    print(c)

    main()

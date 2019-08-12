import torch
import torch.nn as nn
from vae_models import VAE_CONV_NeuralModel
from mnist_classifier import NeuralModel
from utils import data_loader
import torch.nn.functional as F

train_set, test_set = data_loader.get_data()

batch_size = 1

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


class VAEClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vae = VAE_CONV_NeuralModel()
        self.vae.load_state_dict(torch.load("models/trained_CONV_vae"))

        self.classification_model = NeuralModel()
        self.classification_model.load_state_dict(torch.load("models/trained_model"))

    def forward(self, x):
        after_vae = self.vae(x)
        classification = self.classification_model(after_vae[0])

        return classification
        pass


def test_model(model, test_data):
    model.eval()

    correct = 0
    try:
        for batch in test_data:
            batch_images, batch_labels = batch

            predictions = model(batch_images)
            predictions = predictions.data.max(1, keepdim=True)[1]
            correct += predictions.eq(batch_labels.data.view_as(predictions)).sum()

    except:
        (batch_images, batch_labels) = test_data
        predictions = model(batch_images)

        predictions = predictions.data.max(1, keepdim=True)[1]
        correct += predictions.eq(batch_labels.data.view_as(predictions)).sum()

    accuracy = float(correct.item() / len(test_loader.dataset))

    print("The classifier accuracy is: ", 100 * accuracy)


print("Before attack ")
model = VAEClassifierModel()
test_model(model, test_loader)

print("=*=" * 20)
print("After attack ...")


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)

        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        #print(init_pred)
        # If the initial prediction is wrong, dont bother attacking, just move on
        # if init_pred.item() != target.item():
        #     continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 10000:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


acc, examples = test(model, device, test_loader, epsilon=0.3)


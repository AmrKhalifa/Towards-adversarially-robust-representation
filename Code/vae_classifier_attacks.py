import torch
import torch.nn as nn
from vae_models import VAE_CONV_NeuralModel
from mnist_classifier import NeuralModel
from utils import data_loader

train_set, test_set = data_loader.get_data()

batch_size = 100

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)


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


model = VAEClassifierModel()
test_model(model, test_loader)

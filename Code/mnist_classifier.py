import torch
import torch.nn as nn
from utils import data_loader

train_set, test_set = data_loader.get_data()
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


class NeuralModel(nn.Module):
    def __init__(self):
        super().__init__()

        num_channels = 8

        self.conv = nn.Sequential(

            nn.Conv2d(1, num_channels, kernel_size=5),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_channels, eps=1e-05, momentum=0.5, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels, eps=1e-05, momentum=0.5, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_channels, num_channels, kernel_size=5),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_channels, eps=1e-05, momentum=0.5, affine=True),
            nn.ReLU(inplace=True)

        )
        self.fc1 = nn.Linear(num_channels * 4 ** 2, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        convolved = self.conv(x)
        after_fc1 = self.fc1(convolved.view(convolved.size(0), -1))
        output = self.fc2(after_fc1)
        return output


batch_size = 512
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)


def train_model(model, train_data):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 10
    model.train()
    
    model.to(device)
    
    for epoch in range(n_epochs):

        for batch in train_data:
            batch_images, batch_labels = batch
            
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            batch_output = model(batch_images)
            loss = criterion(batch_output, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            print("the loss after processing this batch is: ", loss.item())
            optimizer.step()

    return model


def test_model(model, test_data):
    model.eval()
    model.to(device)
    correct = 0
    ## stupid and very general try except block, modify it later for specific exceptions
    try:
        for batch in test_data:
            batch_images, batch_labels = batch
            
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            predictions = model(batch_images)

            predictions = predictions.data.max(1, keepdim=True)[1]
            correct += predictions.eq(batch_labels.data.view_as(predictions)).sum()

    except:
        print("executing, the except block") 
        (batch_images, batch_labels) = test_data
        predictions = model(batch_images)

        predictions = predictions.data.max(1, keepdim=True)[1]
        correct += predictions.eq(batch_labels.data.view_as(predictions)).sum()

    accuracy = float(correct.item() / len(test_loader.dataset))

    #print("The classifier accuracy is: ", 100 * accuracy)
    
    return accuracy


def main():
    pass

if __name__ == "__main__":
    neural_model = NeuralModel()

    trained_model = train_model(neural_model, train_loader)
    torch.save(trained_model.state_dict(), "models/trained_model")

    classification_model = NeuralModel()
    classification_model.load_state_dict(torch.load("models/trained_model"))

    test_model(classification_model, test_loader)

    main()

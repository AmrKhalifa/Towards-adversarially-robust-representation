import torch
import torch.nn as nn
from utils import data_loader
from utils.viewer import show_batch

train_set, test_set = data_loader.get_data()


class VAE_CONV_NeuralModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)

        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 16, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True)
        )

        self.fc_mu = nn.Linear(50, 50)
        self.fc_log_var = nn.Linear(50, 50)

        # Sampling vector
        self.latent = nn.Sequential(

            nn.Linear(50, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),

            nn.Linear(50, 7 * 7 * 16),
            nn.BatchNorm1d(7 * 7 * 16),
            nn.ReLU(inplace=True)

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()

        )

    def reparameterize(self, mean, log_var):
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mean)
        else:
            return mean

    def forward(self, x):
        encoded = self.encoder(x)

        encoded = encoded.view(-1, 7 * 7 * 16)

        fc = self.fc(encoded)

        mu = self.fc_mu(fc)
        log_var = self.fc_log_var(fc)

        z = self.reparameterize(mu, log_var)
        latent = self.latent(z)
        latent = latent.view(-1, 16, 7, 7)

        decoded = self.decoder(latent)
        output = decoded.view(-1, 1, 28, 28)

        return output, mu, log_var


# d = 20
#
# class VAE_FC_NeuralModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Linear(784, d ** 2),
#             nn.ReLU(),
#             nn.Linear(d ** 2, d * 2)
#         )
#
#         self.decoder = nn.Sequential(
#             nn.Linear(d, d ** 2),
#             nn.ReLU(),
#             nn.Linear(d ** 2, 784),
#             nn.Sigmoid(),
#         )
#
# def reparameterise(self, mu, logvar):
#     if self.training:
#         std = logvar.mul(0.5).exp_()
#         eps = std.data.new(std.size()).normal_()
#         return eps.mul(std).add_(mu)
#     else:
#         return mu
#
#     def forward(self, x):
#         mu_logvar = self.encoder(x.view(-1, 784)).view(-1, 2, d)
#         mu = mu_logvar[:, 0, :]
#         logvar = mu_logvar[:, 1, :]
#         z = self.reparameterise(mu, logvar)
#         return self.decoder(z), mu, logvar


batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

# vae = VAE_FC_NeuralModel()
vae = VAE_CONV_NeuralModel()


def VAELoss(x_hat, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x.view(-1, 784), reduction='sum'
    )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + 1*KLD


def train_vae(model, train_data):
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_epochs = 10
    model.train()
    for epoch in range(n_epochs):

        for batch in train_data:
            batch_images, _ = batch

            batch_output, mean, log_var = model(batch_images)
            loss = VAELoss(batch_output, batch_images, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            print("the loss after processing this batch is: ", loss.item())
            optimizer.step()

        if epoch % 3 == 0:
            lr /= 2
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model


def main():
    pass


if __name__ == "__main__":
    # vae = VAE_CONV_NeuralModel()
    # vae = train_vae(vae, train_loader)
    #
    # torch.save(vae.state_dict(), "models/trained_CONV_vae_B=1")

    vae = VAE_CONV_NeuralModel()
    vae.load_state_dict(torch.load("models/trained_CONV_vae_B=1"))


    first_batch = next(iter(train_loader))
    first_images, _ = first_batch
    recs, _, _ = vae(first_images)

    recs = recs.reshape(batch_size, 1, 28, 28)
    print(recs.shape)
    # show_batch(first_images)
    show_batch(recs)

    main()

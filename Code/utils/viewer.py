import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils


def show_batch(batch):
    grid = utils.make_grid(batch.detach(), nrow=25)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()

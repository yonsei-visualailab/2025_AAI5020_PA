import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def validate(model, loader, device):
    """
    Your Task) Complete the function in train.py.

    - Define a margin-based ranking loss that encourages query vectors 
      to be closer to positive samples than negative ones.
    - Use the same loss formulation as in training:
          loss = max(0, d(q,p) + margin - d(q,n))
    - Compute the average validation loss across all batches.

    Returns:
        float: Average validation loss.
    """
    total_loss = 0.0

    # TODO: 
    # 1. Set the model to evaluation mode.
    # 2. Disable gradient computation (torch.no_grad).
    # 3. For each batch, extract embeddings for q, p, n.
    # 4. Apply the triplet margin loss.
    # 5. Accumulate the loss and return the average.

    return total_loss


def train(model, train_loader, val_loader, device, epochs=5, lr=1e-4):
    """
    Your Task) Complete the function in train.py.

    - Implement the training loop with a margin-based triplet ranking loss.
    - Loss formulation:
          loss = max(0, d(q,p) + margin - d(q,n))
    - Optimize the model using Adam optimizer.
    - After each epoch:
        * Print the average training loss.
        * Evaluate on the validation set using `validate`.

    """
    # TODO:
    # 1. Define a nn.TripletMarginLoss.
    # 2. Define an optim.Adam
    # 3. For each epoch:
    #    - Set model to train mode.
    #    - Loop over training batches:
    #        * Move q, p, n to the device.
    #        * Compute embeddings using the model.
    #        * Compute triplet loss.
    #        * Backpropagate and update parameters.
    #        * Track the training loss.
    #    - Print average training loss.
    #    - Call validate(model, val_loader, device) to compute validation loss and print it.

    pass

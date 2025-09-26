import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class NetVLADModel(nn.Module):
    """
    Your Task) Complete the function in model.py.

    - Use VGG16 CNN to convert the input image into a high-dimensional feature map.
    - Perform soft-assignment to cluster centroids and aggregate residual vectors into a global descriptor.
    - Normalize the final descriptor for retrieval tasks.

    Returns:
        Tensor: Global image descriptor of shape (N, K*D)
    """
    def __init__(self, num_clusters=16, dim=512):
        super().__init__()

        # TODO:
        # 1. Load a pretrained VGG16 backbone.
        #    * Remove the final pooling/classification layers.
        #    * Keep only convolutional feature extractor.
        # 2. Define a NetVLAD pooling layer with `num_clusters` and feature dimension `dim`.
        #    * Cluster centers as learnable parameters.
        #    * 1x1 convolution for soft-assignment.

        pass

    def forward(self, x):
        """
        Parameters:
            x (Tensor): Input image batch (N, 3, H, W)

        Returns:
            Tensor: Normalized global descriptor (N, K*D)
        """
        # TODO:
        # 1. Extract feature maps using backbone.(use models.vgg16() as a backbone)
        # 2. Apply NetVLAD pooling to aggregate descriptors.
        # 3. L2-normalize the final descriptor.

        return ...
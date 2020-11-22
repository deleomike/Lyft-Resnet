import math
import random
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as f

# Number of entries in Agent State Vector
ASV_DIM = 3

def calculate_backbone_feature_dim(backbone, input_shape: Tuple[int, int, int]) -> int:
    """ Helper to calculate the shape of the fully-connected regression layer. """
    tensor = torch.ones(1, *input_shape)
    output_feat = backbone.forward(tensor)
    return output_feat.shape[-1]

class LyftNet(nn.Module):
    """
    Implementation of Multiple-Trajectory Prediction (MTP) model
    based on https://arxiv.org/pdf/1809.10732.pdf
    """

    def __init__(self, backbone: nn.Module, num_modes: int, num_targets: int, num_kinetic_dim: int = 3,
                 n_hidden_layers: int = 4096, input_shape: Tuple[int, int, int] = (3, 500, 500)):
        """
        Inits the MTP network.
        :param backbone: CNN Backbone to use.
        :param num_modes: Number of predicted paths to estimate for each agent.
        :param seconds: Number of seconds into the future to predict.
            Default for the challenge is 6.
        :param frequency_in_hz: Frequency between timesteps in the prediction (in Hz).
            Highest frequency is nuScenes is 2 Hz.
        :param n_hidden_layers: Size of fully connected layer after the CNN
            backbone processes the image.
        :param input_shape: Shape of the input expected by the network.
            This is needed because the size of the fully connected layer after
            the backbone depends on the backbone and its version.
        Note:
            Although seconds and frequency_in_hz are typed as floats, their
            product should be an int.
        """

        super().__init__()

        self.ASV_DIM = num_kinetic_dim

        self.backbone = backbone
        self.num_modes = num_modes
        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)
        self.fc1 = nn.Linear(backbone_feature_dim + self.ASV_DIM, n_hidden_layers, bias=True)
        # predictions_per_mode = int(seconds * frequency_in_hz) * 2
        predictions_per_mode = num_targets

        self.num_pred = predictions_per_mode * num_modes
        self.future_len = int(num_targets / 2)

        self.norm = nn.BatchNorm1d(self.ASV_DIM)

        self.fc2 = nn.Linear(n_hidden_layers, int(self.num_pred + num_modes))

    def forward(self, image_tensor: torch.Tensor,
                agent_state_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        :param image_tensor: Tensor of images shape [batch_size, n_channels, length, width].
        :param agent_state_vector: Tensor of floats representing the agent state.
            [batch_size, ASV_DIM * num_targets].
        :returns: Tensor of dimension [batch_size, number_of_modes * number_of_predictions_per_mode + number_of_modes]
            storing the predicted trajectory and mode probabilities. Mode probabilities are normalized to sum
            to 1 during inference.
        """

        backbone_features = self.backbone(image_tensor)

        features = torch.cat([backbone_features, self.norm(agent_state_vector)], dim=1)

        predictions = self.fc2(self.fc1(features))

        bs, _ = predictions.shape

        pred, confidences = torch.split(predictions, self.num_pred, dim=1)

        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)

        confidences = torch.softmax(confidences, dim=1)

        # print(self.fc1.weight)
        # print("---------------")

        # nn.utils.clip_grad_norm_()

        if math.isnan(confidences[0][0].item()):
            print("Oh no, not the naan")
            print(agent_state_vector)

        return pred, confidences
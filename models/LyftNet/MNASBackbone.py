import torch
import torch.nn as nn
from torchvision.models import mnasnet1_0


def trim_network_at_index(network: nn.Module, index: int = -1) -> nn.Module:
    """
    Returns a new network with all layers up to index from the back.
    :param network: Module to trim.
    :param index: Where to trim the network. Counted from the last layer.
    """
    assert index < 0, f"Param index must be negative. Received {index}."
    return nn.Sequential(*list(network.children())[:index])


class MnasBackbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.

    Allowed versions: mobilenet_v2.
    """

    def __init__(self, num_in_channels: int = 3):
        """
        Inits MobileNetBackbone.
        :param version: mobilenet version to use.
        """
        super().__init__()

        model = mnasnet1_0(pretrained=True)
        layer = model.layers[0]
        model.layers[0] = nn.Conv2d(
            in_channels=num_in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=False,
        )

        self.backbone = trim_network_at_index(model, -1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width].
        :return: Tensor of shape [batch_size, n_convolution_filters]. For mobilenet_v2,
            the shape is [batch_size, 1280].
        """
        backbone_features = self.backbone(input_tensor)
        return backbone_features.mean([2, 3])


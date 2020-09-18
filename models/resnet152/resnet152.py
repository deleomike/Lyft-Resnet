from torchvision.models import resnet152
import torch.nn as nn
from l5kit.configs import load_config_data


class Model(nn.Module):

    def __init__(self, config=None, checkpoint_path=None):
        super().__init__()

        if config is None:
            # Load in the config
            self.cfg = load_config_data("./models/resnet152/config.yaml")
        else:
            self.cfg = config

        model = resnet152(pretrained=True)

        # change input channels number to match the rasterizer's output
        num_history_channels = (self.cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        object_methods = [method_name for method_name in dir(model.conv1)
                          if callable(getattr(model.conv1, method_name))]

        print(num_in_channels)
        print(model.conv1.out_channels)
        print(model.conv1.kernel_size)
        print(model.conv1.stride)
        print(model.conv1.padding)
        model.conv1 = nn.Conv2d(
            in_channels=num_in_channels,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False,
        )

        # change output size to (X, Y) * number of future states
        num_targets = 2 * self.cfg["model_params"]["future_num_frames"]
        model.fc = nn.Linear(in_features=2048, out_features=num_targets)

        self.model = model

    def forward(self, data, device, criterion):
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)
        # Forward pass
        outputs = self.model(inputs).reshape(targets.shape)
        loss = criterion(outputs, targets)
        # not all the output steps are valid, but we can filter them out from the loss using availabilities
        loss = loss * target_availabilities
        loss = loss.mean()
        return loss, outputs


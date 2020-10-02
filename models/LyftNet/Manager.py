from nuscenes.prediction.models.covernet import CoverNet
from nuscenes.prediction.models.mtp import MTP, MTPLoss
from nuscenes.prediction.models.backbone import MobileNetBackbone, ResNetBackbone
from torchvision.models import mnasnet1_0, mnasnet1_3, resnet152

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer
from l5kit.dataset import AgentDataset

from torch.utils.data import DataLoader

import os
import torch

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np

import pandas as pd


class MTPManager:

    def __init__(self, config, data_path, device, num_modes=1):

        self.cfg = config
        self.data_path = data_path
        self.device = device

        self.backbone = MnasBackbone()
        # self.backbone = ResNetBackbone("resnet50")

        # change input channels number to match the rasterizer's output
        num_history_channels = (self.cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        # self.backbone = mnasnet1_3(pretrained=False)
        # firstLayer = self.backbone.layers[0]
        # print(num_in_channels)
        # print(firstLayer.out_channels)
        # print(firstLayer.kernel_size)
        # print(firstLayer.stride)
        # print(firstLayer.padding)
        # self.backbone.layers[0] = nn.Conv2d(
        #     in_channels=num_in_channels,
        #     out_channels=firstLayer.out_channels,
        #     kernel_size=firstLayer.kernel_size,
        #     stride=firstLayer.stride,
        #     padding=firstLayer.padding,
        #     bias=False,
        # )

        # lastConv = self.backbone.layers[14]

        # self.backbone.layers[14] = nn.Conv2d(
        #     in_channels=lastConv.in_channels,
        #     out_channels=lastConv.out_channels,
        #     kernel_size=(5, 7),
        #     stride=lastConv.stride,
        #     padding=lastConv.padding,
        #     bias=False,
        # )

        # change output size to (X, Y) * number of future states
        num_targets = 2 * self.cfg["model_params"]["future_num_frames"]
        self.future_len = self.cfg["model_params"]["future_num_frames"]
        self.num_targets = num_targets
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        # model.fc = nn.Linear(in_features=2048, out_features=self.num_preds + num_modes)

        # print(self.backbone.layers)
        print("++++++++++++++++++++++++++++++++++++++++++")


        future_seconds = self.cfg["model_params"]["future_num_frames"] * self.cfg["model_params"]["future_delta_time"]
        frequency = 1 / self.cfg["model_params"]["future_delta_time"]
        self.model = MTP(self.backbone, num_modes=num_modes)#, input_shape=(num_in_channels, 300, 300))

        self.lossModel = MTPLoss(num_modes=num_modes)
        self.model.to(device=self.device)

    def train(self, iterations, lr=1e-3, file_name="mtp.pth"):

        # set env variable for data
        os.environ["L5KIT_DATA_FOLDER"] = self.data_path
        dm = LocalDataManager(None)
        # get config
        cfg = self.cfg
        print(cfg)

        # ===== INIT DATASET
        train_cfg = cfg["train_data_loader"]

        # Rasterizer
        rasterizer = build_rasterizer(cfg, dm)

        # Train dataset/dataloader
        train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
        train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

        agents = pd.DataFrame.from_records(train_zarr.agents,
                                           columns=['centroid', 'extent', 'yaw', 'velocity', 'track_id',
                                                    'label_probabilities'])

        #
        # print(agents.head())

        train_dataloader = DataLoader(train_dataset,
                                      shuffle=train_cfg["shuffle"],
                                      batch_size=train_cfg["batch_size"],
                                      num_workers=train_cfg["num_workers"])

        print(train_dataset)

        # ==== INIT MODEL parameters
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction="none")

        # ==== TRAIN LOOP

        tr_it = iter(train_dataloader)
        progress_bar = tqdm(range(iterations))
        losses_train = []
        rolling_avg = []
        # torch.save(model.state_dict(), "/home/michael/Workspace/Lyft/model/resnet_base.pth")
        for i in progress_bar:
            try:
                data = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data = next(tr_it)
            self.model.train()
            torch.set_grad_enabled(True)

            ####################
            # this is where it gets real funky

            ids = data["track_id"]
            position_tensor = data["target_positions"]
            vel_tensor, accel_tensor = self._track_kinetics_(target_id_tensor=ids, target_position_tensor=position_tensor)

            yaw_tensor = data["target_yaws"]

            # agent_vector = torch.cat([data["target_positions"], data["target_yaws"]], 2).to(device=self.device)
            # history_vector = torch.cat([data["history_positions"], data["history_yaws"]], 2).to(device=self.device)

            # state_vector = torch.cat([agent_vector, history_vector], 1)
            # state_vector = torch.cat([[position_tensor, vel_tensor, accel_tensor, yaw_tensor]], 2).to(device=self.device)

            # image_tensor = torch.Tensor(data["image"]).permute(2, 0, 1).unsqueeze(0).to(device=self.device)

            inputData = data["image"].to(self.device)
            inputData = torch.ones(1, 3, 500, 500).to(self.device)
            print(inputData.shape)

            pred = self.model.forward(inputData, torch.ones(1, 6).to(device=self.device))
            prob = 1

            print("Prediction: ", pred.shape)
            print("Probability: ", prob.shape)

            loss = self.lossModel(pred, data["target_positions"])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_train.append(loss.item())
            rolling_avg.append(np.mean(losses_train))
            progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

            # if i == 10000:
            #     torch.save(model.state_dict(), "/home/michael/Workspace/Lyft/model/resnet" + str(i) + ".pth")

        print("Done Training")
        torch.save(self.model.state_dict(), f"/home/michael/Workspace/Lyft/model/{file_name}")


    def _track_kinetics_(self, target_id_tensor, target_position_tensor):
        vel = target_position_tensor
        accel = target_position_tensor

        return vel, accel




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

    def __init__(self):
        """
        Inits MobileNetBackbone.
        :param version: mobilenet version to use.
        """
        super().__init__()

        self.backbone = trim_network_at_index(mnasnet1_3(), -1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width].
        :return: Tensor of shape [batch_size, n_convolution_filters]. For mobilenet_v2,
            the shape is [batch_size, 1280].
        """
        backbone_features = self.backbone(input_tensor)
        return backbone_features.mean([2, 3])
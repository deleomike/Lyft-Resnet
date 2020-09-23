from nuscenes.prediction.models.covernet import CoverNet
from nuscenes.prediction.models.mtp import MTP
from nuscenes.prediction.models.backbone import MobileNetBackbone, ResNetBackbone

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


class MTPManager:

    def __init__(self, config, data_path, device, num_modes=3):

        self.cfg = config
        self.data_path = data_path
        self.device = device

        self.backbone = MobileNetBackbone('mobilenet_v2')
        self.model = MTP(self.backbone, num_modes=num_modes)

    def train(self, iterations, lr=1e-3, file_name="resnet_mtp.pth"):

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
            agent_vector = torch.cat([data["target_positions"], data["target_yaws"]], 2)
            loss, _, _ = self.model.forward(data["image"].to(self.device), agent_vector)

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



# class CoverNetManager:
#
#     def __init__(self, num_modes=3):
#
#         self.backbone = MobileNetBackbone('MobileNetV1.0')
#         self.CoverNet = CoverNet(self.backbone, num_modes=num_modes)
#
#     def train(self):
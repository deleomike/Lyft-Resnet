from models.LyftNet.LyftNet import LyftNet, LyftLoss
from models.LyftNet.MNASBackbone import MnasBackbone
from models.resnet152.loss_functions import pytorch_neg_multi_log_likelihood_batch

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer
# from l5kit.dataset import AgentDataset
from models.LyftNet.KineticDataset import KineticDataset

from torch.utils.data import DataLoader

import os
import torch

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np

import pandas as pd


class LyftManager:

    def __init__(self, config, data_path, device, num_modes=3, verbose=False):

        self.cfg = config
        self.data_path = data_path
        self.device = device

        self.verbose = verbose

        # change input channels number to match the rasterizer's output
        num_history_channels = (self.cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        # Create backbone
        self.backbone = MnasBackbone(num_in_channels=num_in_channels)

        # Raster Size
        raster_size = self.cfg["raster_params"]["raster_size"]

        # change output size to (X, Y) * number of future states
        num_targets = 2 * self.cfg["model_params"]["future_num_frames"]
        self.future_len = self.cfg["model_params"]["future_num_frames"]
        print("Number of Targets = ", self.future_len)
        self.num_targets = num_targets
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.model = LyftNet(self.backbone, num_modes=num_modes, num_kinetic_dim=7 * self.future_len,
                             num_targets=num_targets, input_shape=(num_in_channels, raster_size[0], raster_size[1]))

        self.lossModel = LyftLoss(num_modes=num_modes)
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
        if self.verbose:
            print("Dataset Chunked")
        train_dataset = KineticDataset(cfg, train_zarr, rasterizer)
        if self.verbose:
            print("Agent Dataset Retrieved")

        # agents = pd.DataFrame.from_records(train_zarr.agents,
        #                                    columns=['centroid', 'extent', 'yaw', 'velocity', 'track_id',
        #                                             'label_probabilities'])

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
            position_tensor = data["target_positions"].to(self.device)
            vel_tensor, accel_tensor = self._track_kinetics_(target_id_tensor=ids, target_position_tensor=position_tensor)

            yaw_tensor = data["target_yaws"].to(self.device)

            inputData = data["image"].to(self.device)
            if self.verbose:
                print("Image Tensor: ", inputData.shape)

            state_vector = torch.cat([position_tensor, vel_tensor, accel_tensor, yaw_tensor], 2).to(self.device)
            state_vector = torch.flatten(state_vector, 1).to(self.device)
            if self.verbose:
                print("State Vector: ", state_vector.shape)

            pred, conf = self.model.forward(inputData, state_vector)

            if self.verbose:
                print("Prediction: ", pred.shape)
                print("Probability: ", conf.shape)

            target_availabilities = data["target_availabilities"].unsqueeze(-1).to(self.device)

            flattenedTargets = torch.flatten(target_availabilities, 1, 2)

            loss = pytorch_neg_multi_log_likelihood_batch(position_tensor, pred, conf, flattenedTargets)

            # loss = self.lossModel(pred, data["target_positions"])


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


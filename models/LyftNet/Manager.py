from models.LyftNet.LyftNet import LyftNet
from models.LyftNet.MNASBackbone import MnasBackbone
from models.resnet152.loss_functions import pytorch_neg_multi_log_likelihood_batch, pytorch_neg_multi_log_likelihood_single
from l5kit.evaluation import write_pred_csv

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer
from l5kit.dataset import AgentDataset
from models.LyftNet.Kinetic.KineticDataset import KineticDataset

from torch.utils.data import DataLoader

import os
import torch

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np

import gc

import pandas as pd

import horovod.torch as hvd


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

        self.model = LyftNet(self.backbone, num_modes=num_modes, num_kinetic_dim=2 * (self.future_len + 1),
                             num_targets=num_targets, input_shape=(num_in_channels, raster_size[0], raster_size[1]))

        self.model.to(device=self.device)

    def train(self, iterations, lr=1e-3, file_name="lyft-net.pth"):

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
        criterion = nn.PoissonNLLLoss()

        # Add Horovod Distributed Optimizer
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

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

            # ids = data["track_id"]
            position_tensor = data["target_positions"].to(self.device)
            # velocity_tensor = data["target_velocities"].to(self.device)
            # acceleration_tensor = data["target_accelerations"].to(self.device)
            # yaw_tensor = data["target_yaws"].to(self.device)

            history_position_tensor = data["history_positions"].to(self.device)
            estimated_future_positions = data["estimated_future_positions"].to(self.device)
            # history_velocity_tensor = data["history_velocities"].to(self.device)
            # history_acceleration_tensor = data["history_accelerations"].to(self.device)
            # history_yaw_tensor = data["history_yaws"].to(self.device)
            # history_availability = data["history_availabilities"].to(self.device)

            imageTensor = data["image"].to(self.device)
            if self.verbose:
                print("Image Tensor: ", imageTensor.shape)

            # state_vector = torch.cat([history_position_tensor, history_velocity_tensor, history_acceleration_tensor,
            #                           history_yaw_tensor], 2).to(self.device)

            state_vector = torch.cat([estimated_future_positions, history_position_tensor], 1).to(self.device)

            state_vector = torch.flatten(state_vector, 1).to(self.device)
            if self.verbose:
                print("State Vector: ", state_vector.shape)

            pred, conf = self.model.forward(imageTensor, state_vector)

            # loss2 = criterion(pred, position_tensor)

            # print(loss2)

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

            # Save once a day
            if i % 86000 == 0:
                torch.save(self.model.state_dict(), f"./model/{file_name[:-4]}_backup-{i}.pth")

        print("Done Training")
        torch.save(self.model.state_dict(), f"./model/{file_name}")

    def evaluate(self, data_path, file_name="submission.csv"):

        # set env variable for data
        os.environ["L5KIT_DATA_FOLDER"] = data_path
        dm = LocalDataManager(None)

        cfg = self.cfg

        # ===== INIT DATASET
        test_cfg = cfg["test_data_loader"]

        # Rasterizer
        rasterizer = build_rasterizer(cfg, dm)

        # Test dataset/dataloader
        test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
        test_mask = np.load(f"{data_path}/scenes/mask.npz")["arr_0"]
        # test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
        test_dataset = KineticDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
        test_dataloader = DataLoader(test_dataset,
                                     shuffle=test_cfg["shuffle"],
                                     batch_size=test_cfg["batch_size"],
                                     num_workers=test_cfg["num_workers"])
        test_dataloader = test_dataloader
        print(test_dataloader)

        # ==== EVAL LOOP
        self.model.eval()
        torch.set_grad_enabled(False)
        criterion = nn.MSELoss(reduction="none")

        # store information for evaluation
        future_coords_offsets_pd = []
        timestamps = []
        pred_coords =  []
        confidences_list = []

        agent_ids = []
        progress_bar = tqdm(test_dataloader)
        for data in progress_bar:

            # ids = data["track_id"]
            # position_tensor = data["target_positions"].to(self.device)
            # velocity_tensor = data["target_velocities"].to(self.device)
            # acceleration_tensor = data["target_accelerations"].to(self.device)
            # yaw_tensor = data["target_yaws"].to(self.device)

            history_position_tensor = data["history_positions"].to(self.device)
            estimated_future_positions = data["estimated_future_positions"].to(self.device)
            # history_velocity_tensor = data["history_velocities"].to(self.device)
            # history_acceleration_tensor = data["history_accelerations"].to(self.device)
            # history_yaw_tensor = data["history_yaws"].to(self.device)
            # history_availability = data["history_availabilities"].to(self.device)

            imageTensor = data["image"].to(self.device)
            if self.verbose:
                print("Image Tensor: ", imageTensor.shape)

            # state_vector = torch.cat([history_position_tensor, history_velocity_tensor, history_acceleration_tensor,
            #                           history_yaw_tensor], 2).to(self.device)

            state_vector = torch.cat([estimated_future_positions, history_position_tensor], 1).to(self.device)

            state_vector = torch.flatten(state_vector, 1).to(self.device)
            # print(state_vector)
            if self.verbose:
                print("State Vector: ", state_vector.shape)

            pred, confidences = self.model.forward(imageTensor, state_vector)

            # future_coords_offsets_pd.append(outputs.cpu().numpy().copy())
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())
            #
            # pred, confidences = predictor(image)

            pred_coords.append(pred.cpu().numpy().copy())
            confidences_list.append(confidences.cpu().numpy().copy())

        # ==== Save Results
        pred_path = f"{os.getcwd()}/{file_name}"
        write_pred_csv(pred_path,
                       timestamps=np.concatenate(timestamps),
                       track_ids=np.concatenate(agent_ids),
                       coords=np.concatenate(pred_coords),
                       confs=np.concatenate(confidences_list))

        torch.cuda.empty_cache()

    def load(self, checkpoint_path:str):

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    # def clean(self):  # DOES WORK
    #     self._optimizer_to(torch.device('cuda:0'))
    #     del self.optimizer
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #
    # def _optimizer_to(self, device):
    #     for param in self.optimizer.state.values():
    #         # Not sure there are any global tensors in the state dict
    #         if isinstance(param, torch.Tensor):
    #             param.data = param.data.to(device)
    #             if param._grad is not None:
    #                 param._grad.data = param._grad.data.to(device)
    #         elif isinstance(param, dict):
    #             for subparam in param.values():
    #                 if isinstance(subparam, torch.Tensor):
    #                     subparam.data = subparam.data.to(device)
    #                     if subparam._grad is not None:
    #                         subparam._grad.data = subparam._grad.data.to(device)


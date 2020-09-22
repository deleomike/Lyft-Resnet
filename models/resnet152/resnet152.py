from torchvision.models import resnet152
import torch.nn as nn
import torch
from torch import tensor
from l5kit.configs import load_config_data
from l5kit.evaluation.metrics import neg_multi_log_likelihood
import pytorch_pfn_extras as ppe

from models.resnet152.loss_functions import pytorch_neg_multi_log_likelihood_batch, pytorch_neg_multi_log_likelihood_single

import numpy as np
import os
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.evaluation import write_pred_csv, create_chopped_dataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv

from pathlib import Path


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
        self.num_targets = num_targets
        self.num_pred = num_history_channels
        model.fc = nn.Linear(in_features=2048, out_features=self.num_targets)

        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

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


class MultiModalModel(nn.Module):

    def __init__(self, config=None, num_modes=1, checkpoint_path=None):
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
        self.future_len = self.cfg["model_params"]["future_num_frames"]
        self.num_targets = num_targets
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        model.fc = nn.Linear(in_features=2048, out_features=self.num_preds + num_modes)

        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, data, criterion):
        inputs = data["image"].to(self.device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(self.device)
        targets = data["target_positions"].to(self.device)
        # Forward pass
        x = self.model(inputs)
        bs, _ = x.shape

        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)

        confidences = torch.softmax(confidences, dim=1)

        flattenedTargets = torch.flatten(target_availabilities, 1, 2)

        loss = pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, flattenedTargets)

        metrics = {
            "loss": loss.item(),
            "nll": pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, flattenedTargets).item(),
        }
        ppe.reporting.report(metrics, self)

        return loss, pred, confidences

        #return loss, confidences, pred, metrics

    def evaluate(self, data_path):

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
        test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
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
            _, pred, confidences = self.forward(data, criterion)

            # future_coords_offsets_pd.append(outputs.cpu().numpy().copy())
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())
            #
            # pred, confidences = predictor(image)

            pred_coords.append(pred.cpu().numpy().copy())
            confidences_list.append(confidences.cpu().numpy().copy())

        # ==== Save Results
        pred_path = "./submission.csv"
        write_pred_csv(pred_path,
                       timestamps=np.concatenate(timestamps),
                       track_ids=np.concatenate(agent_ids),
                       coords=np.concatenate(pred_coords),
                       confs=np.concatenate(confidences_list))



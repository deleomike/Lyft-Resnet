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

        model = resnet152(pretrained=False)

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


config = {
    'format_version': 4,
    'model_params': {
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },

    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 4
    }

}

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


def evaluate(model, device, data_path, output_path="./"):

    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = data_path
    dm = LocalDataManager(None)

    cfg = model.cfg

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

    print(test_dataloader)

    # ==== EVAL LOOP
    model.eval()
    torch.set_grad_enabled(False)
    criterion = nn.MSELoss(reduction="none")

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []

    agent_ids = []
    progress_bar = tqdm(test_dataloader)
    for data in progress_bar:
        _, outputs = model.forward(data, device, criterion)
        future_coords_offsets_pd.append(outputs.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())

    # ==== Save Results
    pred_path = f"{output_path}submission.csv"
    write_pred_csv(pred_path,
                   timestamps=np.concatenate(timestamps),
                   track_ids=np.concatenate(agent_ids),
                   coords=np.concatenate(future_coords_offsets_pd))


data_path = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
checkpoint_path = "/kaggle/input/lyftmd/resnet.pth"
config_path = "/kaggle/input/lyftmodel/config.yaml"
output_path = "/kaggle/working/"

#config = load_config_data(config_path)
print("config:")
print(config)
print("eof")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(config=config, checkpoint_path=checkpoint_path).to(device)

if checkpoint_path is not None:
    state = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state)

evaluate(model=model, device=device, data_path=data_path, output_path=output_path)


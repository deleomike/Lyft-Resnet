from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.resnet152.resnet152 import Model


from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import os

def train(model, device):

    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = "/home/michael/Workspace/Lyft/data/"
    dm = LocalDataManager(None)
    # get config
    cfg = model.cfg
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
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss(reduction="none")

    # ==== TRAIN LOOP
    tr_it = iter(train_dataloader)
    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    losses_train = []
    rolling_avg = []
    torch.save(model.state_dict(), "/home/michael/Workspace/Lyft/model/resnet_base.pth")
    for i in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)
        model.train()
        torch.set_grad_enabled(True)
        loss, _ = model.forward(data, device, criterion)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())
        rolling_avg.append(np.mean(losses_train))
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

        # if i == 10000:
        #     torch.save(model.state_dict(), "/home/michael/Workspace/Lyft/model/resnet" + str(i) + ".pth")

    torch.save(model.state_dict(), "/home/michael/Workspace/Lyft/model/resnet.pth")
    plt.plot(rolling_avg)

    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model("./models/resnet152/config.yaml").to(device)

train(model, device)

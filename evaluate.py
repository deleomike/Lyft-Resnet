from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50, resnet152
from torchvision.models import googlenet
from tqdm import tqdm

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

def evaluate(model):


    # ==== EVAL LOOP
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []

    agent_ids = []
    progress_bar = tqdm(eval_dataloader)
    for data in progress_bar:
        _, ouputs = forward(data, model, device, criterion)
        future_coords_offsets_pd.append(ouputs.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())

    # ==== Save Results
    pred_path = f"{gettempdir()}/pred.csv"
    print(pred_path)

    write_pred_csv(pred_path,
                   timestamps=np.concatenate(timestamps),
                   track_ids=np.concatenate(agent_ids),
                   coords=np.concatenate(future_coords_offsets_pd),
                  )

    # ==== Perform Evaluation
    print(eval_gt_path)
    metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)

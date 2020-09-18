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


def evaluate(model, device, data_path):

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
    test_dataloader = test_dataloader
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
    pred_path = "./submission.csv"
    write_pred_csv(pred_path,
                   timestamps=np.concatenate(timestamps),
                   track_ids=np.concatenate(agent_ids),
                   coords=np.concatenate(future_coords_offsets_pd))

    # ===== GENERATE AND LOAD CHOPPED DATASET
    num_frames_to_chop = 56
    test_cfg = cfg["test_data_loader"]
    test_base_path = create_chopped_dataset(zarr_path=dm.require(test_cfg["key"]),
                                            th_agent_prob=cfg["raster_params"]["filter_agents_threshold"],
                                            num_frames_to_copy=num_frames_to_chop,
                                            num_frames_gt=cfg["model_params"]["future_num_frames"],
                                            min_frame_future=MIN_FUTURE_STEPS)

    eval_zarr_path = str(Path(test_base_path) / Path(dm.require(test_cfg["key"])).name)
    print(eval_zarr_path)
    test_mask_path = str(Path(test_base_path) / "mask.npz")
    test_gt_path = str(Path(test_base_path) / "gt.csv")

    test_zarr = ChunkedDataset(eval_zarr_path).open()
    test_mask = np.load(test_mask_path)["arr_0"]


    # ===== INIT DATASET AND LOAD MASK
    test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
    test_dataloader = DataLoader(test_dataset, shuffle=test_cfg["shuffle"], batch_size=test_cfg["batch_size"],
                                 num_workers=test_cfg["num_workers"])
    print(test_dataset)

    # ==== Perform Evaluation
    print(test_gt_path)
    metrics = compute_metrics_csv(test_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)

from torchvision.models import resnet152
import torch.nn as nn
from l5kit.configs import load_config_data

import numpy as np

import torch
from torch import Tensor
import pytorch_pfn_extras as ppe

def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)

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

        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=2048, out_features=4096),
        )

        # change output size to (X, Y) * number of future states
        num_targets = 2 * self.cfg["model_params"]["future_num_frames"]
        self.future_len = self.cfg["model_params"]["future_num_frames"]
        self.num_targets = num_targets
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        model.fc = nn.Linear(in_features=2048, out_features=self.num_preds + num_modes)

        self.model = model

    def forward(self, data, device, criterion):
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)
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

        return loss, confidences, metrics



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
        _, outputs, _ = model.forward(data, device, criterion)
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
checkpoint_path = "/kaggle/input/lyftmulti/resnet.pth"
config_path = "/kaggle/input/lyftmulti/config.yaml"
output_path = "/kaggle/working/"

config = load_config_data(config_path)
print("config:")
print(config)
print("eof")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MultiModalModel(config=config, checkpoint_path=checkpoint_path).to(device)

if checkpoint_path is not None:
    state = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state)

evaluate(model=model, device=device, data_path=data_path, output_path=output_path)


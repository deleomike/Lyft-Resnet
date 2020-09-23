from models.resnet152.resnet152 import MultiModalModel as ResnetModel
from train import train
from l5kit.configs import load_config_data
from evaluate import evaluate

from CoverNetManager import MTPManager

import torch

data_path = "/home/michael/Workspace/Lyft/data/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = load_config_data("/home/michael/Workspace/Lyft/models/resnet152/config_multi.yaml")

model = MTPManager(config=config,data_path=data_path, device=device)

model.train(iterations=1, lr=1e-3)
import torch
from l5kit.configs import load_config_data

from MTPManager import MTPManager

data_path = "/home/michael/Workspace/Lyft/data/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = load_config_data("/home/michael/Workspace/Lyft/models/config_mtp.yaml")

model = MTPManager(config=config, data_path=data_path, device=device)

model.train(iterations=1, lr=1e-3, file_name="mtp.pth")


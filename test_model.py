from models.resnet152.resnet152 import MultiModalModel, Model
from train import train
from evaluate import evaluate
from l5kit.configs import load_config_data

import torch

data_path = "/home/michael/Workspace/Lyft/data/"

config = load_config_data("/home/michael/Workspace/Lyft/models/resnet152/config_multi.yaml")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MultiModalModel(num_modes=3, config=config).to(device)


singleModel = Model().to(device)

model.load_state_dict(torch.load("/home/michael/Workspace/Lyft/model/resnet.pth", map_location=device))

#evaluate(model=singleModel, device=device, data_path=data_path)
model.evaluate(data_path=data_path)

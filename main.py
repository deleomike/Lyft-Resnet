from models.resnet152.resnet152 import MultiModalModel as ResnetModel
from train import train
from l5kit.configs import load_config_data
from evaluate import evaluate
import os

import torch

data_path = f"{os.getcwd()}/data/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = load_config_data(f"{os.getcwd()}/models/resnet152/config_multi.yaml")
model = ResnetModel(num_modes=3, config=config).to(device)

#model.load_state_dict(torch.load("/home/michael/Workspace/Lyft/model/resnet.pth", map_location=device))

# get the model familiar with the whole dataset
model = train(model=model, device=device, data_path=data_path, lr=1e-4, force_iters=10)
model.evaluate(data_path=data_path, file_name="submission1.csv")
#model = train(model=model, device=device, data_path=data_path, lr=1e-4, force_iters=20000)
#model.evaluate(data_path=data_path, file_name="submission2.csv")

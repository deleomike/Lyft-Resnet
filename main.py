from models.resnet152.resnet152 import MultiModalModel as ResnetModel
from train import train
from l5kit.configs import load_config_data
from evaluate import evaluate

import torch

data_path = "/home/michael/Workspace/Lyft/data/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = load_config_data("/home/michael/Workspace/Lyft/models/resnet152/config_multi.yaml")
model = ResnetModel(num_modes=3, config=config).to(device)

#model.load_state_dict(torch.load("/home/michael/Workspace/Lyft/model/resnet.pth", map_location=device))

# get the model familiar with the whole dataset
model = train(model=model, device=device, data_path=data_path, lr=1e-5, force_iters=10000)
model.evaluate(data_path=data_path)
model = train(model=model, device=device, data_path=data_path, lr=1e-6, force_iters=10000)

model.evaluate(data_path=data_path)

from models.resnet152.resnet152 import MultiModalModel, Model
from train import train
from evaluate import evaluate

import torch

data_path = "/home/michael/Workspace/Lyft/data/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MultiModalModel(num_modes=3).to(device)

singleModel = Model().to(device)

model.load_state_dict(torch.load("/home/michael/Workspace/Lyft/model/resnet.pth", map_location=device))

#evaluate(model=singleModel, device=device, data_path=data_path)
model.evaluate(device=device, data_path=data_path)

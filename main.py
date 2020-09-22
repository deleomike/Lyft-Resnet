from models.resnet152.resnet152 import MultiModalModel as ResnetModel
from train import train
from evaluate import evaluate

import torch

data_path = "/home/michael/Workspace/Lyft/data/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResnetModel(num_modes=3).to(device)

model.load_state_dict(torch.load("/home/michael/Workspace/Lyft/model/resnet.pth", map_location=device))

# get the model familiar with the whole dataset
model = train(model=model, device=device, data_path=data_path, lr=1e-4, force_iters=10000)

model.evaluate(data_path=data_path)

from models.resnet152.resnet152 import Model as ResnetModel
from train import train
from evaluate import evaluate

import torch

data_path = "/home/michael/Workspace/Lyft/data/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResnetModel().to(device)

model = train(model=model, device=device, data_path=data_path)
evaluate(model=model, device=device, data_path=data_path)





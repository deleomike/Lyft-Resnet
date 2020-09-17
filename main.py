from models.resnet152.resnet152 import Model as ResnetModel
from train import train

import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResnetModel("./models/resnet152/config.yaml").to(device)

model = train(model, device)




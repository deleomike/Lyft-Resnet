import torch
from l5kit.configs import load_config_data

from models.LyftNet.Manager import LyftManager as Manager

data_path = "/home/michael/Workspace/Lyft/data/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = load_config_data("/home/michael/Workspace/Lyft/models/LyftNet/config_lyftnet.yaml")

checkpoint_path = "/home/michael/Workspace/Lyft/model/lyft-net.pth"


manager = Manager(config=config, data_path=data_path, device=device, num_modes=3, verbose=False)

# manager.load(checkpoint_path)

#toto train for 52000
# manager.train(iterations=44000, lr=1e-4, file_name="lyft-net.pth")
manager.train(iterations=264000, lr=1e-4, file_name="lyft-net.pth")

manager.evaluate(data_path=data_path)

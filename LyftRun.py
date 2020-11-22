import torch
from l5kit.configs import load_config_data
import horovod.torch as hvd

hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())

from models.LyftNet.Manager import LyftManager as Manager

data_path = "./data/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = load_config_data("./models/LyftNet/config_lyftnet.yaml")


manager = Manager(config=config, data_path=data_path, device=device, num_modes=3, verbose=False)

# checkpoint_path = "./model/lyft-net-base-448px_backup-344000.pth"
# manager.load(checkpoint_path)

# 604800 is one week approximately
# manager.train(iterations=44000, lr=1e-4, file_name="lyft-net.pth")
manager.train(iterations=352000, lr=1e-4, file_name="lyft-net-base-448px.pth")

manager.evaluate(data_path=data_path)

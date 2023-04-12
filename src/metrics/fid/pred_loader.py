import torch
from torch.utils.data import Dataset

from cfg.cfg import Config

class LoadPred(Dataset):
    def __init__(self, config: Config, model, size = 256):
        super(LoadPred, self).__init__()
        self.config = config
        self.model = model
        self.size = size

    def __getitem__(self, index):
        inputs = torch.randn([1, self.config.input_dims]).to(self.config.device)
        image = self.model(inputs)[0]
        return image


    def __len__(self):
        return self.size
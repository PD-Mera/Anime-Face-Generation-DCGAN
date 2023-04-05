import os, sys
from pathlib import Path
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from cfg.cfg import Config
from src.model import Generator
from src.plot import draw_plot




def infer(config: Config):
    device = config.device

    net_G = Generator(in_features=config.input_dims).to(device)
    net_G.load_state_dict(torch.load(config.best_checkpoint))

    inputs = torch.randn(config.num_col * config.num_row, config.input_dims).to(device)
    outputs = net_G(inputs) 

    draw_plot(outputs, config)

if __name__ == "__main__":
    config = Config(phase = 'valid')
    infer(config)
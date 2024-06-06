from torch import Tensor
import numpy as np
import yaml
import pandas as pd
import os
import matplotlib.pyplot as plt


def operate(a: bool, b, c) -> any:
    return b if a is True else c


def tensor_to_numpy(tensor: Tensor) -> np.float32:
    return tensor.detach().cpu().numpy().astype(np.float32)


def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def show_plot(path: str, name: str) -> None:
    pass
    # data = pd.read_csv(os.path.join(path, name))
    # x = data[COLUMNS[0]]

    # plt.figure()

    # for col in COLUMNS[1:]:
    #     plt.plot(x, data[col], label=col)

    # plt.xlabel(COLUMNS[0])
    # plt.legend()
    # plt.show()


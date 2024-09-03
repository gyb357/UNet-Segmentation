import torch.nn as nn


class VGG(nn.Module):
    def __init__(
            self
    ) -> None:
        super(VGG, self).__init__()

        
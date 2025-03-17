import logging
import os
from typing import Optional, Tuple


class Augmentation():
    def __init__(
            self,
            channels: int,
            resize: Optional[Tuple[int, int]] = None,
            hflip: bool = False,
            vflip: bool = False,
            rotate: float = 0,
            saturation: float = 0,
            brightness: float = 0,
            contrast: float = 0,
            factor: float = 1,
            p: float = 0.5
    ) -> None:
        
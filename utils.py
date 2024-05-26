import os


def makedirs(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


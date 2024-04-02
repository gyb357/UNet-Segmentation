import json
import os


def load_config(path: str) -> json:
    with open(path, 'r') as f:
        return json.load(f)


def make_folder(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


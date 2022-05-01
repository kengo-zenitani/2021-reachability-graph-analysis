import os
import pickle
import gc
from typing import Any


class Problem:
    def __init__(self, name: str):
        self.name = name


def save_object(filename: str, obj: Any):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp, protocol=4)
        return obj


def load_object(filename: str) -> Any:
    with open(filename, 'rb') as fp:
        try:
            gc.disable()
            return pickle.load(fp)
        finally:
            gc.enable()

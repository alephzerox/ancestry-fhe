from __future__ import annotations

import pickle

import numpy as np
import torch


class ModelParameters:

    @classmethod
    def load_from_pickle(cls, path):
        with open(path, "rb") as f:
            smoother_weights: list[float] = pickle.load(f)

        window_size = 200
        model = ModelParameters(window_size, smoother_weights)
        return model

    def __init__(self, window_size, smoother_weights):
        self._window_size = window_size
        self._smoother_weights: np.ndarray = np.array(smoother_weights)
        self._smoother_weights_as_tensor = torch.tensor(smoother_weights)

    @property
    def window_size(self):
        return self._window_size

    @property
    def smoother_weights(self):
        return self._smoother_weights

    @property
    def smoother_weights_as_tensor(self):
        return self._smoother_weights_as_tensor

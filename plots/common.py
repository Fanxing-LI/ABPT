from tensorboard.backend.event_processing import event_accumulator
import torch as th
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
import os, sys
from typing import Dict


@dataclass
class log:
    t: np.ndarray
    v: np.ndarray
    step: np.ndarray
    is_interpolated: bool = False

    def __mul__(self, other):
        return log(self.t, self.v * other, self.step)

    def __add__(self, other):
        return log(self.t, self.v + other, self.step)

    def __truediv__(self, other):
        return log(self.t, self.v / other, self.step)

    def interpolate(self, max_interp_step=None):
        max_interp_step = round(self.step[-1], -6) if max_interp_step is None else max_interp_step
        new_step = np.arange(self.step[0], max_interp_step, 10000)
        self.v = np.interp(new_step, self.step, self.v)
        self.t = np.interp(new_step, self.step, self.t)
        self.step = new_step
        self.is_interpolated = True

        return self

    def append(self, other):
        assert self.is_interpolated

        # self.t = np.stack([self.t, other.t])
        self.v = np.vstack([np.atleast_2d(self.v), np.atleast_2d(other.v)])
        # self.step = np.stack([self.step, other.step])

        return self

    def max(self):
        return np.max(self.v, 0)

    def min(self):
        return np.min(self.v, 0)

    def mean(self):
        return np.mean(self.v, 0)


def load_tensorboard_log(path, max_interp_step=None):
    # child file in this path folder
    path = os.path.join(path, os.listdir(path)[0])
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    tags = ea.Tags()['scalars']
    tag = tags[0]  # Replace with the tag you are interested in
    events = ea.Scalars(tag)
    # Get the list of tags
    tags = ea.Tags()['scalars']
    events = ea.Scalars(tag)

    # convert log file data which tag includes "rollout/" to self defined class log
    data = {}
    for tag in tags:
        if "rollout/" in tag:
            events = ea.Scalars(tag)
            t = np.array([e.wall_time for e in events])
            t = t - t[0]
            v = np.array([e.value for e in events])
            v = gaussian_filter1d(v, sigma=2)
            step = np.array([e.step for e in events])
            data[tag] = log(t, v, step).interpolate(max_interp_step=max_interp_step)

    return data


def load_average_data(root_path, tag="std", max_interp_step=None):
    i = 5
    all_data = []
    for i in range(i):
        path = root_path + f"_{tag}_{i + 1}"
        data = load_tensorboard_log(path, max_interp_step=max_interp_step)
        all_data.append(data)

    sum_data: Dict = all_data[0]
    for key in sum_data.keys():
        for i in range(1, len(all_data)):
            sum_data[key].append(all_data[i][key])
    return sum_data

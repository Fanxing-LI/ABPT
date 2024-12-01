# load tensorboard data and plot
from common import load_average_data
import os, sys
from typing import Dict
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from VisFly.utils.FigFashion.FigFashion import FigFon

FigFon.set_fashion("IEEE")
fig, axeses = FigFon.get_figure_axes(SubFigSize=(2, 2),share_legend=True, Column=1,
                                     Border=(0, 0, 1, 1))

types = ["std", "velR_robust"]
types_alias_list = ["Position-based Reward", "Velocity-based Reward"]
algs = [
    "MTD",
    "PPO",
    "SHAC",
    "BPTT"
]
algs_alias = [
    "ABPT",
    "PPO",
    "SHAC",
    "BPTT"
]

from VisFly.utils.FigFashion.colors import colorsets

colors = colorsets["Modern Scientific"]
env = "racing"
root_path = f"examples/{env}/saved/"

for i, tp in enumerate(types):
    for j, alg in enumerate(algs):
        path = f"{root_path}/{alg}"
        if alg == "SHAC":
            data = load_average_data(path, tag=tp, max_interp_step=7e6)
        else:
            data = load_average_data(path, tag=tp)
        r = data["rollout/ep_rew_mean"]
        s = data["rollout/success_rate"]
        num_gate = data["rollout/ep_past_gate_mean"]
        alpha = 0.2
        axeses[0,i].plot(num_gate.step, num_gate.mean(), color=colors[j], label=alg)
        axeses[0,i].fill_between(num_gate.step, num_gate.min(), num_gate.max(), color=colors[j], alpha=alpha)
        axeses[0,i].set_xlabel("Time-steps")
        axeses[1, i].plot(num_gate.t, num_gate.mean(), color=colors[j], label=alg)
        axeses[1, i].fill_between(num_gate.t, num_gate.min(), num_gate.max(), color=colors[j], alpha=alpha)
        axeses[1, i].set_xlabel("Wall-time (s)")

        axeses[0, i].set_title(types_alias_list[i].capitalize())

    if i == 0:
        axeses[0,i].set_ylabel("Number of Past Gates")
        axeses[1,i].set_ylabel("Number of Past Gates")

    # set legends
    # axeses[0,i].legend()
    FigFon.set_shared_legend(axeses[0,i].lines, algs_alias)

    test = 1

plt.show()
current_file_path = os.path.abspath(__file__)
# Get the directory name of the current file
current_file_dir = os.path.dirname(current_file_path)
fig.savefig(current_file_dir+"/reward_robust.png")
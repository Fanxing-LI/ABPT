# load tensorboard data and plot
from common import load_average_data
import os, sys
from typing import Dict
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from VisFly.utils.FigFashion.FigFashion import FigFon

FigFon.set_fashion("IEEE")
fig, axeses = FigFon.get_figure_axes(SubFigSize=(1, 4),share_legend=True, Column=2,
                                     Border=(0, 0, 1, 1))

types = ["lr0.01", "lr0.003","lr0.001","lr0.0003","lr0.0001"]
types_alias_list = [r"$\alpha=0.01$", r"$\alpha=0.003$",r"$\alpha=0.001$",r"$\alpha=0.0003$",r"$\alpha=0.0001$"]
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
env = "hovering"
root_path = f"examples/{env}/saved/"

for j, alg in enumerate(algs):
    for i, tp in enumerate(types):
        path = f"{root_path}/{alg}"
        data = load_average_data(path, tag=tp)
        r = data["rollout/ep_rew_mean"]
        s = data["rollout/success_rate"]
        # num_gate = data["rollout/ep_past_gate_mean"]
        alpha = 0.2
        axeses[j].plot(r.step, r.mean(), color=colors[i], label=alg)
        axeses[j].fill_between(r.step, r.min(), r.max(), color=colors[i], alpha=alpha)
        axeses[j].set_xlabel("Time-steps")
        # axeses[1, i].plot(num_gate.t, num_gate.mean(), color=colors[j], label=alg)
        # axeses[1, i].fill_between(num_gate.t, num_gate.min(), num_gate.max(), color=colors[j], alpha=alpha)
        # axeses[1, i].set_xlabel("Wall-time (s)")

    axeses[j].set_title(algs_alias[j])

axeses[0].set_ylabel("Reward")
        # axeses[1,i].set_ylabel("Number of Past Gates")

    # set legends
    # axeses[0,i].legend()
FigFon.set_shared_legend(axeses[0].lines, types_alias_list)


plt.show()
current_file_path = os.path.abspath(__file__)
# Get the directory name of the current file
current_file_dir = os.path.dirname(current_file_path)
fig.savefig(current_file_dir+"/lr_robust.png")
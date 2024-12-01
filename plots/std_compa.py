# load tensorboard data and plot
from common import load_average_data
import os, sys
from typing import Dict
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from VisFly.utils.FigFashion.FigFashion import FigFon

FigFon.set_fashion("IEEE")
fig, axeses = FigFon.get_figure_axes(SubFigSize=(2, 4),share_legend=True,
                                     Border=(0, 0, 1, 1))

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
envs = [
    "hovering",
    "tracking",
    "landing",
    "racing"
]


from VisFly.utils.FigFashion.colors import colorsets

colors = colorsets["Modern Scientific"]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors)


for i, env in enumerate(envs):
    root_path = f"examples/{env}/saved/"
    for j, alg in enumerate(algs):
        path = f"{root_path}/{alg}"
        data = load_average_data(path)
        r = data["rollout/ep_rew_mean"]
        s = data["rollout/success_rate"]
        alpha=0.2
        axeses[0, i].plot(r.step, r.mean(), color=colors[j], label=alg)
        axeses[0, i].fill_between(r.step, r.min(), r.max(), color=colors[j], alpha=alpha)
        axeses[0, i].set_xlabel("Time-steps")

        axeses[1, i].plot(r.t, r.mean(), color=colors[j], label=alg)
        axeses[1, i].fill_between(r.t, r.min(), r.max(), color=colors[j], alpha=alpha)
        axeses[1, i].set_xlabel("Wall-time (s)")

        axeses[0, i].set_title(env.capitalize())
    if i == 0:
        axeses[0, i].set_ylabel("Reward")
        axeses[1, i].set_ylabel("Reward")
        # set legends
        # axeses[0,i].legend()
        FigFon.set_shared_legend(axeses[0, i].lines, algs_alias)

        test = 1

plt.show()
current_file_path = os.path.abspath(__file__)
# Get the directory name of the current file
current_file_dir = os.path.dirname(current_file_path)
fig.savefig(current_file_dir+"/std_compa.png")

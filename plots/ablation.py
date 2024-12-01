from common import load_average_data
import os, sys
sys.path.append(os.getcwd())
from VisFly.utils.FigFashion.FigFashion import FigFon
import matplotlib.pyplot as plt


types = ["std", "no_ent", "no_start", "no_start_no_ent", "no_start_no_ent_no_reset"]
types_alias = {
    "std": "Standard",
    "no_ent": "Without Entropy",
    "no_start": "Without 0-step Return",
    "no_start_no_ent": "Without 0-step Return & Entropy",
    "no_start_no_ent_no_reset": "Without Samping Initial States"
}
types_alias_list = [types_alias[tp] for tp in types]

envs = ["hovering","tracking","landing", "racing"]

alg = "MTD"


FigFon.set_fashion("IEEE")
fig, axeses = FigFon.get_figure_axes(SubFigSize=(1, 4), Column=2, share_legend=True,
                                     Border=(0, 0, 1, 1))

from VisFly.utils.FigFashion.colors import colorsets
colors = colorsets["Modern Scientific"]


for i, env in enumerate(envs):
    root_path = f"examples/{env}/saved/"
    for j, tp in enumerate(types):
        if env in ["hovering", "tracking"] and tp in ["no_ent", "no_start"]:
            continue
        if env in ["landing", "racing"] and tp == "no_start_no_ent_no_reset":
            continue
        path = f"{root_path}/{alg}"
        data = load_average_data(path, tag=tp)
        r = data["rollout/ep_rew_mean"]
        s = data["rollout/success_rate"]
        alpha=0.2
        axeses[i].plot(r.step, r.mean(), color=colors[j], label=tp)
        axeses[i].fill_between(r.step, r.min(), r.max(), color=colors[j], alpha=alpha)
        axeses[i].set_xlabel("Time-steps")
        axeses[i].set_title(env.capitalize())

axeses[0].set_ylabel("Reward")

    # set legends
    # axeses[0,i].legend()
FigFon.set_shared_legend(list(axeses[3].lines)+(list(axeses[0].lines)[-1:]), types_alias_list)

test = 1

axeses[0].set_xlim([0, 1e6])
axeses[1].set_xlim([0, 1e6])
# axeses[2].set_xlim([0, 1e7])
# axeses[3].set_xlim([0, 1e7])

plt.show()
current_file_path = os.path.abspath(__file__)
# Get the directory name of the current file
current_file_dir = os.path.dirname(current_file_path)
fig.savefig(current_file_dir+"/ablation.png")


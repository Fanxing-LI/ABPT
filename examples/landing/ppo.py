#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import torch
import time
import yaml
from VisFly.utils.policies import extractors
from VisFly.utils.algorithms.ppo import ppo
from VisFly.utils import savers
import torch as th
from VisFly.utils.launcher import rl_parser, training_params
from VisFly.utils.type import Uniform
from VisFly.envs.LandingEnv import LandingEnv2

args = rl_parser().parse_args()
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"


def main():
    with open(os.path.dirname(os.path.abspath(__file__))+'/cfg/ppo.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # if train mode, train the model
    if args.train:
        env = LandingEnv2(
            **config["env"]
        )

        if args.weight is not None:
            model = ppo.load(save_folder + args.weight, env=env)
        else:
            model = ppo(
                env=env,
                seed=args.seed,
                comment=args.comment,
                tensorboard_log=save_folder,
                **config["algorithm"]
            )

        model.learn(**config["learn"])
        model.save()

    # Testing mode with a trained weight
    else:
        test_model_path = save_folder + args.weight
        from test import Test
        env = LandingEnv2(**config["eval_env"])
        model = ppo.load(test_model_path, env=env)

        test_handle = Test(
            model=model,
            save_path=os.path.dirname(os.path.realpath(__file__)) + "/saved/test",
            name=args.weight)
        test_handle.test(**config["test"])


if __name__ == "__main__":
    main()

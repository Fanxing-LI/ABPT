#!/usr/bin/env python3

import sys
import os
import torch
import time

sys.path.append(os.getcwd())
from VisFly.utils.policies import extractors
import torch as th
from VisFly.envs.LandingEnv import LandingEnv2
from VisFly.utils.launcher import rl_parser, training_params
import stable_baselines3 as sb3
from VisFly.utils.algorithms.sac import SAC

args = rl_parser().parse_args()
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"
scene_path = "datasets/spy_datasets/configs/garage_empty"

random_kwargs = {
    "state_generator":
        {
            "class": "Uniform",
            "kwargs": [
                {"position": {"mean": [2., 0., 2.5], "half": [1.0, 1.0, 1.0]}},
            ]
        }
}
dynamics_kwargs = {
    "dt": 0.02,
    "ctrl_dt": 0.02,
    "action_type": "thrust",
}


def main():
    # if train mode, train the model
    if args.train:
        env = LandingEnv2(num_agent_per_scene=training_params["num_envs"],
                          random_kwargs=random_kwargs,
                          visual=False,
                          max_episode_steps=training_params["max_episode_steps"],
                          dynamics_kwargs=dynamics_kwargs,
                          requires_grad=False,
                          )
        model = SAC(
            policy="MultiInputPolicy",
            policy_kwargs=dict(
                features_extractor_class=extractors.StateExtractor,
                features_extractor_kwargs={
                    "net_arch": {
                        "state": {
                            # "mlp_layer": [128, 64],
                            "mlp_layer": [192,192],
                            "bn": False,
                        },
                        # "target": {
                        #     "mlp_layer": [128, 64],
                        # }
                    },
                },
                net_arch=dict(
                    pi=[192, 96],
                    qf=[192, 96]),
                activation_fn=torch.nn.ReLU,
                optimizer_kwargs=dict(weight_decay=1e-5),
                share_features_extractor=False,
                # deterministic=False,

            ),
            env=env,
            learning_rate=training_params["learning_rate"],
            batch_size=training_params["batch_size"],
            gamma=training_params["gamma"],
            device="cuda",
            seed=training_params["seed"],
            train_freq=training_params["train_freq"],
            verbose=1,
            tensorboard_log=save_folder,
            gradient_steps=training_params["n_epochs"] * 20,
            commit=args.comment,

        )
        if args.weight is not None:
            model.load(path=save_folder + args.weight)

        start_time = time.time()
        model.learn(training_params["learning_step"]*5, log_interval=100,
                    reset_num_timesteps=False,
                    # progress_bar=True
                    )
        # model.save()
        training_params["time"] = time.time() - start_time

    # Testing mode with a trained weight
    else:
        test_model_path = save_folder + args.weight
        from test import Test
        env = LandingEnv2(num_agent_per_scene=4, visual=True,
                          random_kwargs=random_kwargs,
                          scene_kwargs={
                              "path": scene_path,
                              "render_settings": {
                                  "mode": "fix",
                                  "view": "custom",
                                  "resolution": [1080, 1920],
                                  # "position": th.tensor([[6., 6.8, 5.5], [6,4.8,4.5]]),
                                  "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
                                  "trajectory": True,
                              },
                          },
                          )
        model = SAC(
            env=env,
            policy=sb3.td3.policies.MultiInputPolicy,
        )
        model.load(test_model_path)

        test_handle = Test(
            model=model,
            save_path=os.path.dirname(os.path.realpath(__file__)) + "/saved/test",
            name=args.weight)
        test_handle.test(is_fig=True, is_fig_save=True, is_render=True, is_video=True, is_video_save=True,
                         render_kwargs={})


if __name__ == "__main__":
    main()

import torch as th


class PredictiveEnv:
    def __init__(self, env_class, env_kwargs, batch_size):
        env_kwargs["num_scene"] = 1
        env_kwargs["num_agent_per_scene"] = batch_size
        env_kwargs["visual"] = False
        env_kwargs["requires_grad"] = True
        self.envs = env_class(**env_kwargs)

    def step(self, action):
        return self.envs.step(action)

    def reset_by_id(self, index, state):
        return self.envs.reset_by_id(index, state)

    def reset(self):
        self.envs.reset()

    def predict(self):
        pass


def main():
    from VisFly.envs.LandingEnv import LandingEnv
    env_class = LandingEnv
    env_kwargs = {
        "random_kwargs": {
            "state_generator": {
                "class": "Uniform",
                "kwargs": [
                    {"position": {"mean": [2., 0., 2.5], "half": [1.0, 1.0, 1.0]}},
                ]
            }
        },
        "dynamics_kwargs": {
            "dt": 0.02,
            "ctrl_dt": 0.02,
            "action_type": "thrust",
        }
    }
    batch_size = 1
    predictive_env = PredictiveEnv(env_class, env_kwargs, batch_size=1000)


if __name__ == "__main__":
    main()

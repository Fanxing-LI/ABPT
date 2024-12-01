from typing import Optional, Dict, Type

from gym.vector.utils import spaces

from VisFly.utils.policies.extractors import StateExtractor, set_mlp_feature_extractor
import torch as th
import torch.nn as nn


class StateGateExtractor(StateExtractor):
    def __init__(self, observation_space: spaces.Dict,
                 net_arch: Optional[Dict] = {},
                 activation_fn: Type[nn.Module] = nn.ReLU, ):
        super(StateGateExtractor, self).__init__(observation_space=observation_space, net_arch=net_arch, activation_fn=activation_fn)

    def _build(self, observation_space, net_arch, activation_fn):
        super()._build(observation_space, net_arch, activation_fn)
        gate_feature_dim = set_mlp_feature_extractor(self, "gate", observation_space["gate"], net_arch["gate"], activation_fn)
        self._features_dim = self._features_dim + gate_feature_dim

    def extract(self, observations) -> th.Tensor:
        state_features = self.state_extractor(observations['state'])
        gate_features = self.gate_extractor(observations['gate'])
        return th.cat([state_features, gate_features], dim=1)

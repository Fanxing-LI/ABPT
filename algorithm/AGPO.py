import time
from collections import deque
from typing import Type, Optional, Dict, ClassVar, Any, Union, List

from stable_baselines3.common import logger
import os, sys
from gymnasium import spaces
import random

import numpy as np
import torch as th
from VisFly.utils.policies.td_policies import CnnPolicy, BasePolicy, MultiInputPolicy
# from torch.distributions import
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_schedule_fn, safe_mean, update_learning_rate
from tqdm import tqdm
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
from VisFly.utils.algorithms.lr_scheduler import transfer_schedule
from copy import deepcopy
from algorithm.common import compute_td_returns, DataBuffer3, SimpleRolloutBuffer
from VisFly.utils.common import set_seed

from .shac import TemporalDifferBase


def get_weight_grad(model, losses: th.Tensor):
    grads_all = []
    for loss in losses:
        model.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        grads_all.append(th.cat(grads))
    return th.stack(grads_all)


def compute_vector_variance(vectors):
    if vectors.ndim != 2:
        raise ValueError("Input tensor must have shape (N, d), where N is the number of vectors and d is their dimension.")

    N = vectors.size(0)  # Number of vectors
    if N < 2:
        raise ValueError("At least two vectors are required to compute variance.")

    # Step 1: Compute the mean vector
    mean_vector = vectors.mean(dim=0)  # Shape: (d,)

    # Step 2: Compute squared differences
    squared_diffs = th.sum((vectors - mean_vector) ** 2, dim=1)  # Shape: (N,)

    # Step 3: Compute the variance
    variance = squared_diffs.sum() / (N - 1)

    return variance


def compute_model_grad_variance(model, loss):
    vectors = get_weight_grad(model, loss)
    return compute_vector_variance(vectors)


def compute_alpha(sigma0, sigma1):
    return sigma0 ** 2 / (sigma0 ** 2 + sigma1 ** 2)


class AGPO(TemporalDifferBase):
    def __init__(
            self,
            env,
            policy: Union[Type, str],
            policy_kwargs: Optional[Dict] = None,
            learning_rate: Union[float, Schedule] = 1e-3,
            logger_kwargs: Optional[Dict[str, Any]] = None,
            comment: Optional[str] = None,
            save_path: Optional[str] = None,
            dump_step: int = 1e4,
            horizon: float = 32,
            tau: float = 0.005,
            gamma: float = 0.99,
            gradient_steps: int = 5,
            buffer_size: int = int(1e6),
            batch_size: int = int(2e5),
            clip_range_vf: float = 0.1,
            pre_stop: float = 0.1,
            policy_noise: float = 0.,
            device: Optional[str] = "cpu",
            seed: int = 42,
            # **kwargs
    ):
        super().__init__(
            env=env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            logger_kwargs=logger_kwargs,
            comment=comment,
            save_path=save_path,
            dump_step=dump_step,
            horizon=horizon,
            tau=tau,
            gamma=gamma,
            gradient_steps=gradient_steps,
            buffer_size=buffer_size,
            batch_size=batch_size,
            clip_range_vf=clip_range_vf,
            pre_stop=pre_stop,
            policy_noise=policy_noise,
            device=device,
            seed=seed,
        )

    def _build(self):
        self.name = "AGPO"
        super()._build()

    def learn(
            self,
            total_timesteps: int,
    ):
        # assert self.H >= 1, "horizon must be greater than 1"
        self.policy.train()
        self._logger = self._create_logger(**self.logger_kwargs)

        eq_buffer_len = 100
        # initialization
        eq_rewards_buffer, eq_len_buffer, eq_success_buffer, eq_info_buffer = \
            deque(maxlen=eq_buffer_len), deque(maxlen=eq_buffer_len), deque(maxlen=eq_buffer_len), deque(maxlen=eq_buffer_len)

        for _ in range(eq_buffer_len):
            eq_success_buffer.append(False)

        current_step, previous_step, previous_time = 0, 0, 0
        try:
            with (tqdm(total=total_timesteps) as pbar):
                while current_step < total_timesteps:
                    self._update_current_progress_remaining(num_timesteps=current_step, total_timesteps=total_timesteps)
                    optimizers = [self.actor.optimizer, self.critic.optimizer]
                    # Update learning rate according to lr schedule
                    self._update_learning_rate(optimizers)

                    actor_loss, critic_loss = 0., 0.  # th.tensor(0, device=self.device), th.tensor(0, device=self.device)

                    fail_cache = th.zeros((self.num_envs,), dtype=th.bool, device=self.device)
                    discount_factor = th.ones((self.num_envs,), dtype=th.float32, device=self.device)
                    episode_done = th.zeros((self.num_envs,), device=self.device, dtype=th.bool)
                    inner_step = 0
                    beta = th.ones((self.num_envs,), dtype=th.float32, device=self.device)
                    for inner_step in range(self.H):
                        # dream a horizon of experience
                        obs = self.env.get_observation()
                        pre_obs = obs
                        # iteration
                        actions, _, h = self.policy.actor.action_log_prob(obs)
                        clipped_actions = th.clip(
                            actions, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
                        )

                        # compute Q backpropagation weight grads
                        values_bp, _ = th.cat(self.policy.critic(obs, actions), dim=1).min(dim=1)  # retain the gradient
                        cache_actor_loss = -values_bp
                        # self.actor.optimizer.zero_grad()
                        # cache_actor_loss.backward()
                        Q0_grad_variance = compute_model_grad_variance(self.policy.actor, cache_actor_loss)

                        # compute F1 backpropagation weight grads
                        # step
                        obs, reward, done, info = self.env.step(clipped_actions)
                        for i in range(len(episode_done)):
                            episode_done[i] = info[i]["episode_done"]

                        reward, done = reward.to(self.device), done.to(self.device)
                        current_step += self.num_envs

                        # compute the temporal difference
                        next_actions, _ = self.policy.actor(obs)
                        next_actions = next_actions.clip(
                            th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
                        )
                        next_values, _ = th.cat(self.policy.critic_target(obs.detach(), next_actions.detach()), dim=1).min(dim=1)
                        cache_actor_loss = -reward - self.gamma * next_values
                        # self.actor.optimizer.zero_grad()
                        # cache_actor_loss.backward()
                        F1_grad_variance = compute_model_grad_variance(self.policy.actor, cache_actor_loss)

                        alpha = compute_alpha(sigma0=Q0_grad_variance, sigma1=F1_grad_variance)
                        actor_loss = actor_loss - discount_factor * ((1 - alpha) * beta * values_bp + beta * alpha * reward)

                        done_but_not_episode_end = ((done) | (inner_step == self.H - 1)) & ~episode_done
                        if done_but_not_episode_end.any():
                            actor_loss = actor_loss - next_values * discount_factor * self.gamma * done_but_not_episode_end

                        beta = beta * alpha * ~done + done
                        discount_factor = discount_factor * self.gamma * ~done + done
                        self.rollout_buffer.add(obs=pre_obs.clone().detach(),
                                                reward=reward.clone().detach(),
                                                action=clipped_actions.clone().detach(),
                                                next_obs=obs.clone().detach(),
                                                done=done.clone().detach(),
                                                episode_done=episode_done.clone().detach(),
                                                value=next_values.clone().detach()
                                                )
                    # update
                    actor_loss = (actor_loss).mean()
                    self.policy.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
                    # record grad
                    # get_network_statistics(self.actor, self._logger, is_record=pbar.n - previous_step >= self._dump_step)
                    self.policy.actor.optimizer.step()
                    self.rollout_buffer.compute_returns()
                    self.env.detach()

                    # update critic
                    for i in range(self.gradient_steps):
                        values, _ = th.cat(self.policy.critic(self.rollout_buffer.obs, self.rollout_buffer.action), dim=1).min(dim=1)
                        target = self.rollout_buffer.returns
                        critic_loss = th.nn.functional.mse_loss(target, values)
                        self.policy.critic.optimizer.zero_grad()
                        critic_loss.backward()
                        th.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
                        self.policy.critic.optimizer.step()

                        polyak_update(params=self.policy.critic.parameters(), target_params=self.policy.critic_target.parameters(), tau=self.tau)
                        polyak_update(params=self.critic_batch_norm_stats, target_params=self.critic_batch_norm_stats_target, tau=1.)

                    self.rollout_buffer.clear()

                    # evaluate
                    if pbar.n - previous_step >= self._dump_step:
                        with th.no_grad():
                            eval_info_id_list = [i for i in range(self.num_envs)]
                            self.eval_env.reset_by_id()
                            obs = self.eval_env.get_observation()
                            while True:
                                actions, _ = self.policy.actor(obs, deterministic=True)
                                clipped_actions = th.clip(
                                    actions, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
                                )
                                obs, reward, done, info = self.eval_env.step(clipped_actions, is_test=True)
                                for index in reversed(eval_info_id_list):
                                    if done[index]:
                                        eval_info_id_list.remove(index)
                                        eq_rewards_buffer.append(info[index]["episode"]["r"])
                                        eq_len_buffer.append(info[index]["episode"]["l"])
                                        eq_success_buffer.append(info[index]["is_success"])
                                        eq_info_buffer.append(info[index]["episode"])
                                if done.all():
                                    break

                    if pbar.n - previous_step >= self._dump_step and len(eq_rewards_buffer) > 0:
                        self._logger.record("time/fps", (current_step - previous_step) / (time.time() - previous_time))
                        self._logger.record("rollout/ep_rew_mean", sum(eq_rewards_buffer) / len(eq_rewards_buffer))
                        self._logger.record("rollout/ep_len_mean", sum(eq_len_buffer) / len(eq_len_buffer))
                        self._logger.record("rollout/success_rate", sum(eq_success_buffer) / len(eq_success_buffer))
                        self._logger.record("train/actor_loss", actor_loss.item())
                        self._logger.record("train/critic_loss", critic_loss.item() if isinstance(critic_loss, th.Tensor) else critic_loss)
                        if len(eq_info_buffer[0]["extra"]) >= 0:
                            for key in eq_info_buffer[0]["extra"].keys():
                                self.logger.record(
                                    f"rollout/ep_{key}_mean",
                                    safe_mean(
                                        [ep_info["extra"][key] for ep_info in eq_info_buffer]
                                    ),
                                )
                        self._logger.dump(current_step)
                        previous_time, previous_step = time.time(), current_step
                    pbar.update((inner_step + 1) * self.num_envs)

        except KeyboardInterrupt:
            pass

        return self.policy

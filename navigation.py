"""
Deep Q-Network (DQN) trainer for the Unity Banana Navigation environment.

Key characteristics
- Single seed only (DEFAULT_SEED).
- Unity environment is hosted in a separate worker process (isolates native runtime, similar to p2 Reacher script).
- Supports optional hyperparameter search (--search) across a predefined grid.
- Per-trial, per-seed plotting only:
  • Evaluation per-episode rewards from 1-episode evaluations run after each training episode.
  • 100-episode moving average of those evaluation rewards (overlay).
  * A dotted vertical line marks the start of the first 100-episode window meeting the goal mean.
- A CSV summary is produced for the per-seed plots, capturing early stopping info.
- Exactly one log call per episode, written to both console and run.log (includes episode_steps per episode).

Exploration schedule (updated)
- Epsilon-greedy schedule is configurable: 'linear' or 'exp' (exponential).
- Decay is gated by a warmup phase: epsilon stays high until replay has enough samples (warmup_steps).
- Practical defaults aim to reach min_epsilon around 30–60% of total expected steps.

Outputs
- results/navigation/run_YYYYMMDD_HHMMSS/
    run.log
    hparam_search.csv  (when --search)
    best_hparams.json  (when --search)
    plots/
      evaluation_mean100_{trial_or_default}_seed_{SEED}.png
    plots_summary.csv
    trials/
      {trial_or_default}/seed_{SEED}/checkpoints/model.<ep_idx>.tar
"""

import warnings ; warnings.filterwarnings('ignore')
import os
import sys
import io
import time
import glob
import json
import random
import argparse
from dataclasses import dataclass
from datetime import datetime
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from unityagents import UnityEnvironment

import logging
from logging import handlers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp

# ----------------------------
# Minimal "spaces" shims (avoid gym dependency)
# ----------------------------

@dataclass
class DiscreteSpace:
    """
    Simple discrete action space with a Gym-like interface.

    Attributes:
        n (int): Number of discrete actions.
    """
    n: int
    def sample(self):
        """Sample a random action in [0, n)."""
        return np.random.randint(self.n)
    def seed(self, seed):
        """Seed numpy RNG for reproducibility."""
        np.random.seed(seed)

@dataclass
class BoxSpace:
    """
    Simple box observation space placeholder carrying only shape information.

    Attributes:
        shape (tuple): Observation shape (e.g., (37,) for Banana).
    """
    shape: tuple
    def seed(self, seed):
        """Seed numpy RNG for reproducibility."""
        np.random.seed(seed)

# ----------------------------
# Global constants
# ----------------------------

DEFAULT_SEED = 22
EPS = 1e-6  # used in divisions to avoid zero-division
RESULTS_DIR = os.path.join('results')

np.set_printoptions(suppress=True)

# ----------------------------
# Unity environment worker (single-agent Banana)
# ----------------------------

def _banana_worker(child_end, env_exe_path: str, seed: int, worker_id: int):
    """
    Worker process hosting the UnityEnvironment (single-agent Banana).

    Commands:
    - 'reset' with kwargs {'train_mode': bool}
    - 'step' with kwargs {'action': int}
    - 'close'

    Returns:
    - reset(train_mode): np.ndarray state (37,)
    - step(action): (next_state (37,), reward (float), done (bool))
    """
    try:
        logging.info(f'[Worker {worker_id}] Starting Banana worker (seed={seed}). Opening Unity env: {env_exe_path}')
    except Exception:
        pass

    # Initialize Unity in the child process
    env = UnityEnvironment(file_name=env_exe_path, seed=seed, no_graphics=True, worker_id=worker_id)
    brain_name = env.brain_names[0]

    try:
        logging.info(f'[Worker {worker_id}] Unity env opened. brain_name="{brain_name}"')
    except Exception:
        pass

    def _reset(train_mode: bool = True):
        info = env.reset(train_mode=train_mode)[brain_name]
        return info.vector_observations[0].astype(np.float32)  # (37,)

    def _step(action: int):
        info = env.step(int(action))[brain_name]
        next_state = info.vector_observations[0].astype(np.float32)  # (37,)
        reward = float(info.rewards[0])
        done = bool(info.local_done[0])
        return next_state, reward, done

    try:
        while True:
            cmd, kwargs = child_end.recv()
            if cmd == 'reset':
                train_mode = bool(kwargs.get('train_mode', True))
                child_end.send(_reset(train_mode=train_mode))
            elif cmd == 'step':
                action = int(kwargs.get('action', 0))
                child_end.send(_step(action))
            elif cmd == 'close':
                try:
                    logging.info(f'[Worker {worker_id}] Closing Unity env.')
                except Exception:
                    pass
                env.close(); child_end.close(); break
            else:
                try:
                    logging.info(f'[Worker {worker_id}] Unknown command "{cmd}". Closing.')
                except Exception:
                    pass
                env.close(); child_end.close(); break
    finally:
        try:
            env.close()
            logging.info(f'[Worker {worker_id}] Unity env closed. Worker exiting.')
        except Exception:
            try:
                logging.info(f'[Worker {worker_id}] Worker exiting (env close raised).')
            except Exception:
                pass


class UnityBananaWorkerEnv:
    """
    Lightweight wrapper that runs Unity Banana in a separate process and communicates via a Pipe.

    Why a worker:
    - Isolates native Unity runtime; avoids hangs/crashes impacting the main process.
    - Clean shutdowns, no zombie Banana.exe processes.

    API (Gym-like):
    - observation_space.shape -> (37,)
    - action_space.n -> 4
    - reset(eval_mode: bool=False) -> np.ndarray state
    - step(action: int) -> (next_state, reward, done, info)
    - close()
    """
    def __init__(self, env_exe_path: str, seed: int, worker_id: int = 0):
        self.env_exe_path = env_exe_path
        self.seed = int(seed)
        self.worker_id = int(worker_id)

        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        ctx = mp.get_context('spawn')
        parent_end, child_end = ctx.Pipe()
        self.pipe = parent_end

        self.worker = ctx.Process(target=_banana_worker, args=(child_end, self.env_exe_path, self.seed, self.worker_id), daemon=True)
        self.worker.start()
        time.sleep(0.15)  # small delay to let Unity start

        # Fixed spaces for Banana
        self.observation_space = BoxSpace(shape=(37,))
        self.action_space = DiscreteSpace(n=4)

    def reset(self, eval_mode: bool = False):
        # Unity's train_mode=True corresponds to training (non-eval); eval_mode flips it
        self.pipe.send(('reset', {'train_mode': (not eval_mode)}))
        return self.pipe.recv()

    def step(self, action: int):
        self.pipe.send(('step', {'action': int(action)}))
        next_state, reward, done = self.pipe.recv()
        return next_state, float(reward), bool(done), {}

    def close(self):
        try:
            self.pipe.send(('close', {}))
        except Exception:
            pass
        try:
            self.worker.join(timeout=2.0)
        except Exception:
            pass

# ----------------------------
# Models and RL components (CPU-only)
# ----------------------------

class FCQ(nn.Module):
    """
    Fully-connected Q-network: maps state -> Q-values over discrete actions.

    Args:
        input_dim (int): State dimensionality.
        output_dim (int): Number of discrete actions.
        hidden_dims (tuple): Hidden layer sizes, e.g., (256, 256).
        activation_fc (callable): Activation function, e.g., F.relu.

    Notes:
        Device is forced to CPU for stability and portability in worker setups.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=(64,64), activation_fc=F.relu):
        super(FCQ, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.device = torch.device("cpu")  # CPU-only
        self.to(self.device)

    def _format(self, state):
        """Ensure state is a torch.Tensor on the correct device with batch dimension."""
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        """Forward pass: state -> Q-values."""
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)

    def load(self, experiences):
        """
        Convert a numpy experience batch to device tensors suitable for training.

        Args:
            experiences (tuple): (states, actions, rewards, next_states, is_terminals)

        Returns:
            tuple[Tensor, ...]: (states, actions, rewards, next_states, is_terminals)
        """
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, new_states, is_terminals


class GreedyStrategy():
    """
    Greedy evaluation policy: selects argmax Q(s,·). No exploration.
    """
    def __init__(self):
        self.exploratory_action_taken = False
    def select_action(self, model, state):
        """Return greedy action for the given state using the provided model."""
        with torch.no_grad():
            q_values = model(state).cpu().detach().numpy().squeeze()
            return int(np.argmax(q_values))

class EGreedyStrategy():
    """
    Epsilon-greedy training policy with configurable decay schedule and warmup gating.

    Args:
        init_epsilon (float): Starting epsilon (exploration probability).
        min_epsilon (float): Lower bound after decay completes.
        decay_steps (int): Horizon for decay.
            - Linear: steps to go from init_epsilon to min_epsilon (after warmup).
            - Exponential: characteristic scale; epsilon drops by ~63% of (init-min) over this horizon.
        schedule (str): 'linear' or 'exp' (exponential).
        tau_ratio (float): Only for schedule='exp'. If provided, tau = max(1, int(decay_steps * tau_ratio)).
                           This lets epsilon reach near floor around 3–5 tau.
        warmup_steps (int): Steps to hold epsilon at init_epsilon before starting decay.
                            Recommended: replay_buffer.batch_size * n_warmup_batches.

    Schedule:
        - Linear: e_t = min + (init-min) * max(0, 1 - (t-warmup)/decay_steps)
        - Exponential: e_t = min + (init-min) * exp( - max(0,t-warmup) / tau )

    Note:
        - Epsilon is applied to action selection first, then schedule advances one step.
        - exploratory_action_taken is recorded per call to select_action.
    """
    def __init__(self,
                 init_epsilon=1.0,
                 min_epsilon=0.05,
                 decay_steps=300_000,
                 schedule: str = 'exp',
                 tau_ratio: float = 0.3,
                 warmup_steps: int = 0):
        self.init_epsilon = float(init_epsilon)
        self.min_epsilon = float(min_epsilon)
        self.decay_steps = max(1, int(decay_steps))
        self.schedule = str(schedule).lower()
        self.tau_ratio = float(tau_ratio)
        self.warmup_steps = max(0, int(warmup_steps))

        # internal counters/state
        self.t = 0
        self._epsilon = self.init_epsilon
        self.exploratory_action_taken = False

        # derive tau for exponential schedule
        if self.schedule not in ('linear', 'exp'):
            raise ValueError(f"Unsupported epsilon schedule: {schedule}")
        if self.schedule == 'exp':
            # Default tau uses tau_ratio to reach near floor around 3–5 taus
            self.tau = max(1, int(self.decay_steps * (self.tau_ratio if self.tau_ratio > 0 else 0.3)))
        else:
            self.tau = None

    @property
    def epsilon(self) -> float:
        return float(self._epsilon)

    def _compute_epsilon(self):
        # Hold during warmup
        if self.t < self.warmup_steps:
            return self.init_epsilon
        t_eff = self.t - self.warmup_steps
        if self.schedule == 'linear':
            frac = min(1.0, max(0.0, t_eff / self.decay_steps))
            return max(self.min_epsilon, self.init_epsilon - (self.init_epsilon - self.min_epsilon) * frac)
        # exponential
        return self.min_epsilon + (self.init_epsilon - self.min_epsilon) * np.exp(- t_eff / float(self.tau))

    def _advance(self):
        self._epsilon = float(self._compute_epsilon())
        self.t += 1
        return self._epsilon

    def select_action(self, model, state):
        """
        Choose an epsilon-greedy action and record whether it was exploratory.
        """
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach().numpy().squeeze()
            greedy = int(np.argmax(q_values))
        if np.random.rand() > self.epsilon:
            action = greedy
        else:
            action = int(np.random.randint(len(q_values)))
        self.exploratory_action_taken = (action != greedy)
        self._advance()
        return action

    def on_episode_end(self, mean_100_eval_score, episode_idx):
        """
        Hook reserved for adaptive schedules (no-op for static schedules).
        """
        return

class ReplayBuffer():
    """
    Fixed-size experience replay buffer with uniform random sampling.

    Args:
        max_size (int): Maximum number of transitions stored.
        batch_size (int): Default batch size used by sample().
    """
    def __init__(self, max_size=50000, batch_size=64):
        self.ss_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    def store(self, sample):
        """
        Store a transition (s, a, r, s', done_flag_as_float).
        """
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = np.array(s, dtype=np.float32)
        self.as_mem[self._idx] = np.array([a], dtype=np.int64)
        self.rs_mem[self._idx] = np.array([r], dtype=np.float32)
        self.ps_mem[self._idx] = np.array(p, dtype=np.float32)
        self.ds_mem[self._idx] = np.array([d], dtype=np.float32)
        self._idx = (self._idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    def sample(self, batch_size=None):
        """
        Uniformly sample a batch of transitions.

        Returns:
            tuple of np.ndarray batches aligned along the first dimension.
        """
        if batch_size is None:
            batch_size = self.batch_size
        idxs = np.random.choice(self.size, batch_size, replace=False)
        experiences = (
            np.vstack(self.ss_mem[idxs]),
            np.vstack(self.as_mem[idxs]),
            np.vstack(self.rs_mem[idxs]),
            np.vstack(self.ps_mem[idxs]),
            np.vstack(self.ds_mem[idxs]),
        )
        return experiences
    def __len__(self):
        return self.size

# ----------------------------
# DQN agent
# ----------------------------

class DQN():
    """
    Deep Q-Network trainer encapsulating model, optimizer, exploration/exploitation
    strategies, and a reproducible training/evaluation workflow.

    Overview
    - Maintains two networks:
      • online_model: the network being optimized every training step
      • target_model: a lagged copy used to build stable TD targets (hard-sync)
    - Collects transitions in a ReplayBuffer and samples uniform mini-batches.
    - Uses an epsilon-greedy training policy and a purely greedy evaluation policy.
    - Logs exactly one line per episode that includes episode_steps and rolling means.
    - Saves one checkpoint per episode (prunes most via get_cleaned_checkpoints).
    - Produces per-episode single-episode eval scores, then aggregates a 100-ep MA.

    Changes in this version
    - Environment is spawned inside a separate worker process (UnityBananaWorkerEnv).
    - CPU-only models for stability and portability with subprocess workers.
    - Epsilon schedule supports 'linear' and 'exp' with warmup gating.
    """
    def __init__(self,
                 replay_buffer_fn,
                 value_model_fn,
                 value_optimizer_fn,
                 value_optimizer_lr,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps):
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.trial_id = None
        self.hparams = None
        self.log_prefix = ""
        self.final_eval_episodes = 10

    def optimize_model(self, experiences):
        """
        Perform one TD(0) update on the online network.

        Steps
        1) Compute the bootstrapped target:
           y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        2) Gather Q_online(s, a) for the sampled actions.
        3) Minimize 0.5 * ||Q_online(s, a) - y||^2 via backprop.
        4) Clip the global gradient norm to improve stability.

        Args
        - experiences: tuple of device tensors
          (states, actions, rewards, next_states, is_terminals)

        Returns
        - None. Updates self.online_model in-place.
        """
        states, actions, rewards, next_states, is_terminals = experiences

        # Compute bootstrap value on s'
        if (self.hparams is not None) and bool(self.hparams.get('double_dqn', True)):
            # Double DQN: action from online, value from target
            with torch.no_grad():
                next_q_online = self.online_model(next_states)
                next_actions = next_q_online.argmax(1, keepdim=True)                    # [B, 1]
                next_q_target = self.target_model(next_states).gather(1, next_actions)  # [B, 1]
        else:
            # Vanilla DQN: max over target network
            with torch.no_grad():
                next_q_target = self.target_model(next_states).max(1, keepdim=True)[0]  # [B, 1]

        target_q_sa = rewards + (self.gamma * next_q_target * (1 - is_terminals))      # [B, 1]

        # Q(s, a) from online network
        q_sa = self.online_model(states).gather(1, actions)                             # [B, 1]

        # Loss: Huber (smooth L1) preferred for robustness unless disabled
        use_huber = bool((self.hparams or {}).get('use_huber', True))
        if use_huber:
            value_loss = F.smooth_l1_loss(q_sa, target_q_sa)
        else:
            td_error = q_sa - target_q_sa
            value_loss = td_error.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()

        # Gradient clipping (global norm)
        clip_norm = float((self.hparams or {}).get('grad_clip', 10.0))
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), max_norm=clip_norm)

        self.value_optimizer.step()

    def _env_reset(self, env, eval_mode=False):
        """
        Reset the environment and return the initial observation.

        Worker reset uses train_mode = not eval_mode.
        """
        s = env.reset(eval_mode=eval_mode)
        if isinstance(s, tuple) and len(s) == 2:
            obs, _ = s
            return obs
        return s

    def _env_step(self, env, action):
        """
        Step the environment and normalize the return signature.

        For the worker wrapper:
        - Returns (obs, reward, done, info)
        - The 'is_failure' flag mirrors bool(done) (no truncation info from Unity API).
        """
        out = env.step(action)
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return obs, float(reward), bool(done), info, float(bool(done))
        # Fallback (should not trigger)
        obs, reward, done, info = out
        return obs, float(reward), bool(done), info, float(bool(done))

    def interaction_step(self, state, env):
        """
        Take one training interaction step:
        - Choose action via epsilon-greedy policy on self.online_model.
        - Step the environment.
        - Store transition (s, a, r, s', done_float) in the replay buffer.
        - Update per-episode counters.
        """
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, info, is_failure = self._env_step(env, action)
        experience = (state, action, reward, new_state, is_failure)
        self.replay_buffer.store(experience)
        # Accumulate per-episode stats
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        return new_state, is_terminal

    def update_network(self):
        """
        Hard-sync the target network with the online network.

        Copies parameters from self.online_model to self.target_model. This is
        triggered at a fixed step interval to stabilize TD targets.
        """
        for target, online in zip(self.target_model.parameters(),
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

    def train(self, env_exe_path, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward, base_worker_id=0):
        """
        Train the agent using a worker-hosted Unity Banana environment.

        Workflow per episode
        1) Launch worker env and reset (training mode).
        2) Interact until terminal; accumulate reward/steps/exploration.
        3) Start optimizing once n_warmup_batches are available; then one
           optimization per env step using uniform replay sampling.
        4) Hard-sync target network every update_target_every_steps env steps.
        5) Run one greedy evaluation episode and append its return.
        6) Save a checkpoint for the episode, log one INFO line, and test
           stopping conditions (minutes/episodes/goal mean_100 eval).

        Returns
        - result (np.ndarray): shape (max_episodes, 5)
          columns:
            [0] total_steps (cumulative across episodes)
            [1] mean_100_train_reward
            [2] mean_100_eval_reward
            [3] training_time_seconds (cumulative)
            [4] wallclock_seconds (elapsed since training start)
          Tail may be NaN when stopped early.
        - final_eval_score (float): mean return over final_eval_episodes greedy trials
        - training_time (float): sum of episode durations in seconds
        - wallclock_time (float): total elapsed wall time in seconds
        - stop_episode (int): zero-based episode index of stop
        - stop_cause (str): 'max_minutes' | 'max_episodes' | 'goal_mean_100_reward'
        - stop_mean_100_eval (float|None): mean_100_eval at stop when goal reached, else None
        """
        training_start = time.time()
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.seed = seed
        self.gamma = gamma

        prefix = f"[{self.trial_id} | seed {self.seed}] " if self.trial_id is not None else f"[seed {self.seed}] "
        self.log_prefix = prefix
        logging.info(f"{prefix}=== Starting run for seed {self.seed} ===")
        if self.hparams is not None:
            logging.info(f"{prefix}Hyperparameters: {json.dumps(self.hparams, sort_keys=True)}")

        # Re-seed libraries for reproducibility
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)

        # Start worker environment
        env = UnityBananaWorkerEnv(env_exe_path, seed=self.seed, worker_id=int(base_worker_id))

        # Infer state/action dimensions from the wrapper
        obs_space = env.observation_space
        if hasattr(obs_space, 'shape') and obs_space.shape is not None:
            nS = int(np.prod(obs_space.shape))
        else:
            env.close()
            raise ValueError("Unsupported observation space shape")
        nA = int(env.action_space.n)

        # Per-episode trackers (lists grow with episodes)
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        # Build networks/optimizer/buffer/strategies (CPU-only models)
        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        self.update_network()
        self.value_optimizer = self.value_optimizer_fn(self.online_model, self.value_optimizer_lr)
        self.replay_buffer = self.replay_buffer_fn()

        # Compute exploration warmup in steps (keep epsilon high until replay has enough samples)
        warmup_steps = int(self.replay_buffer.batch_size) * int(self.n_warmup_batches)

        # Strategy: schedule + warmup gating
        self.training_strategy = self.training_strategy_fn(warmup_steps=warmup_steps)
        self.evaluation_strategy = self.evaluation_strategy_fn()

        # Pre-allocate metrics array; rows may remain NaN if early stopped
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0.0
        total_steps_accum = 0

        # Early stopping bookkeeping
        stop_episode = None
        stop_cause = None
        stop_mean_100_eval = None

        try:
            for episode in range(1, max_episodes + 1):
                episode_start = time.time()
                state = self._env_reset(env, eval_mode=False)
                is_terminal = False
                # Start episode accumulators
                self.episode_reward.append(0.0)
                self.episode_timestep.append(0.0)
                self.episode_exploration.append(0.0)

                # --------- Interaction loop ----------
                while not is_terminal:
                    state, is_terminal = self.interaction_step(state, env)
                    # Start optimizing once we have enough samples
                    min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                    if len(self.replay_buffer) > min_samples:
                        experiences = self.replay_buffer.sample()
                        experiences = self.online_model.load(experiences)
                        self.optimize_model(experiences)
                    total_steps_accum += 1
                    # Hard target sync
                    if total_steps_accum % self.update_target_every_steps == 0:
                        self.update_network()

                # Episode timing and cumulative training time
                episode_elapsed = time.time() - episode_start
                self.episode_seconds.append(episode_elapsed)
                training_time += episode_elapsed

                # Greedy evaluation (single episode) to build the eval curve
                eval_score, _ = self.evaluate(self.online_model, env)

                # Save checkpoint for this episode
                self.save_checkpoint(episode-1, self.online_model)

                # Aggregate scalars for logging/plots
                total_step = int(np.sum(self.episode_timestep))
                self.evaluation_scores.append(eval_score)

                mean_100_reward = np.mean(self.episode_reward[-100:]) if len(self.episode_reward) >= 1 else np.nan
                std_100_reward = np.std(self.episode_reward[-100:]) if len(self.episode_reward) >= 1 else np.nan
                mean_100_eval_score = np.mean(self.evaluation_scores[-100:]) if len(self.evaluation_scores) >= 1 else np.nan
                std_100_eval_score = np.std(self.evaluation_scores[-100:]) if len(self.evaluation_scores) >= 1 else np.nan
                if len(self.episode_exploration) >= 1:
                    lst_100_exp_rat = np.array(self.episode_exploration[-100:]) / np.maximum(np.array(self.episode_timestep[-100:]), EPS)
                    mean_100_exp_rat = np.mean(lst_100_exp_rat)
                    std_100_exp_rat = np.std(lst_100_exp_rat)
                else:
                    mean_100_exp_rat = np.nan
                    std_100_exp_rat = np.nan

                wallclock_elapsed = time.time() - training_start
                # Store one row (episode-1 is zero-based index)
                result[episode-1] = total_step, mean_100_reward, mean_100_eval_score, training_time, wallclock_elapsed

                # --------- Epsilon hook (reserved for adaptive schedules) ----------
                if hasattr(self.training_strategy, 'on_episode_end'):
                    self.training_strategy.on_episode_end(
                        mean_100_eval_score if not np.isnan(mean_100_eval_score) else None,
                        episode_idx=episode
                    )

                # --------- Logging (every episode, exactly once) ----------
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
                episode_steps = int(self.episode_timestep[-1])  # number of env steps in this episode
                debug_message = (f'{self.log_prefix}elapsed_time {elapsed_str}, episode {episode-1:04}, '
                                 f'episode_steps {episode_steps}, '
                                 f'mean_reward_over_100_episodes_from_training {mean_100_reward:05.2f}±{std_100_reward:05.2f}, '
                                 f'mean_exploration_ratio_over_100_episodes_from_training {mean_100_exp_rat:.4f}±{std_100_exp_rat:.4f}, '
                                 f'epsilon {self.training_strategy.epsilon:.3f}, '
                                 f'mean_reward_over_100_episodes_from_eval {mean_100_eval_score:05.2f}±{std_100_eval_score:05.2f}')
                # Single logging call; handlers route to console and file
                logging.info(debug_message)

                # --------- Stopping conditions ----------
                reached_max_minutes = wallclock_elapsed >= max_minutes * 60
                reached_max_episodes = episode >= max_episodes
                reached_goal_mean_reward = (not np.isnan(mean_100_eval_score)) and (mean_100_eval_score >= goal_mean_100_reward)
                training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward

                if training_is_over:
                    # Record early stopping details
                    stop_episode = episode - 1
                    if reached_goal_mean_reward:
                        stop_cause = 'goal_mean_100_reward'
                        stop_mean_100_eval = float(mean_100_eval_score) if not np.isnan(mean_100_eval_score) else None
                        logging.info(f'{self.log_prefix}--> reached_goal_mean_100_reward OK')
                    elif reached_max_minutes:
                        stop_cause = 'max_minutes'
                        stop_mean_100_eval = None
                        logging.info(f'{self.log_prefix}--> reached_max_minutes x')
                    elif reached_max_episodes:
                        stop_cause = 'max_episodes'
                        stop_mean_100_eval = None
                        logging.info(f'{self.log_prefix}--> reached_max_episodes x')
                    else:
                        stop_cause = 'unknown'
                        stop_mean_100_eval = None
                    break
        finally:
            # Ensure the worker is closed even on exceptions
            try:
                env.close()
            except Exception:
                pass

        # --------- Final evaluation (multi-episode) ----------
        # Re-open env for final eval to avoid residual episode state issues
        env_final = UnityBananaWorkerEnv(env_exe_path, seed=self.seed+1, worker_id=int(base_worker_id)+1)
        try:
            n_eval_eps = int(getattr(self, 'final_eval_episodes', 10))
            final_eval_score, score_std = self.evaluate(self.online_model, env_final, n_episodes=n_eval_eps)
        finally:
            env_final.close()

        wallclock_time = time.time() - training_start
        logging.info(f'{self.log_prefix}Training complete.')
        logging.info(f'{self.log_prefix}Final evaluation score {final_eval_score:.2f}±{score_std:.2f} in {training_time:.2f}s training, {wallclock_time:.2f}s wall.')

        # Cleanup old checkpoints (keep a handful spread across episodes)
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time, stop_episode, stop_cause, stop_mean_100_eval

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        """
        Evaluate a policy greedily for n_episodes (uses the worker env).

        Args
        - eval_policy_model: model used to select greedy actions
        - eval_env: environment wrapper (eval_mode=True resets)
        - n_episodes (int): number of evaluation episodes

        Returns
        - mean_reward (float)
        - std_reward (float)
        """
        rs = []
        for _ in range(n_episodes):
            s = self._env_reset(eval_env, eval_mode=True)
            d = False
            rs.append(0.0)
            while not d:
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        """
        Keep at most n_checkpoints files, spaced roughly evenly across episodes.

        Implementation
        - Read all checkpoint episode indices from filenames model.<ep_idx>.tar.
        - Compute a set of evenly spaced indices including the last episode.
        - Delete files not in the retained set; cache kept paths in self.checkpoint_paths.

        Args
        - n_checkpoints (int): maximum number of checkpoints to keep

        Returns
        - dict[int, str]: episode_idx -> filepath for retained checkpoints
        """
        try:
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}
        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        if not paths:
            return self.checkpoint_paths
        paths_dic = {int(path.split('.')[-2]): path for path in paths}
        last_ep = max(paths_dic.keys())
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int) - 1
        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                try: os.unlink(path)
                except Exception: pass
        return self.checkpoint_paths

    def save_checkpoint(self, episode_idx, model):
        """
        Save the online model weights for a given episode index.

        Args
        - episode_idx (int): zero-based episode index at which the snapshot is taken
        - model (nn.Module): source network whose state_dict is saved
        """
        torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, f'model.{episode_idx}.tar'))

# ----------------------------
# Utilities
# ----------------------------

def _setup_logging(run_dir):
    """
    Initialize root logger to write both to console and to run.log.

    To prevent duplicate lines in run.log and console:
      - Clear any pre-existing handlers.
      - Add a de-dup filter to the file handler (drops consecutive identical messages).
      - If stderr is redirected to the same run.log file (e.g., using shell
        redirection/tee), skip adding the file handler to avoid double writes.
      - Avoid logger propagation to ancestor loggers.

    Args:
        run_dir (str): Output directory for the current run.

    Returns:
        str: Path to the created log file.
    """
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'run.log')

    class _DedupFilter(logging.Filter):
        """Drop consecutive identical messages (level+message) to avoid double writes."""
        def __init__(self):
            super().__init__()
            self._last = None
        def filter(self, record: logging.LogRecord) -> bool:
            key = (record.levelno, record.getMessage())
            if key == self._last:
                return False
            self._last = key
            return True

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers (avoid duplicates across re-inits or IDE-attached handlers)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Console handler (stderr)
    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(ch)

    # Decide whether to also add a file handler
    add_file_handler = True
    try:
        stderr_name = getattr(sys.stderr, 'name', None)
        if isinstance(sys.stderr, io.TextIOBase) and stderr_name and os.path.exists(log_path) and os.path.exists(stderr_name):
            try:
                add_file_handler = not os.path.samefile(stderr_name, log_path)
            except Exception:
                add_file_handler = (os.path.abspath(stderr_name) != os.path.abspath(log_path))
    except Exception:
        pass

    if add_file_handler:
        fh = handlers.WatchedFileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        fh.addFilter(_DedupFilter())
        logger.addHandler(fh)

    # Avoid propagating to ancestor loggers that might add handlers
    logger.propagate = False

    logging.info(f'Logging initialized. Log file: {log_path}')
    return log_path

def _default_env_settings():
    """
    Default environment and training stop conditions.

    Returns:
        dict with keys:
            gamma (float): Discount factor for TD target.
            max_minutes (int): Wall-clock budget (minutes).
            max_episodes (int): Episode budget (cap).
            goal_mean_100_reward (float): Early stop threshold on eval mean_100.
            final_eval_episodes (int): Episodes for final evaluation at the end.
    """
    return {
        'gamma': 0.99,
        'max_minutes': 2000,
        'max_episodes': 2000,
        'goal_mean_100_reward': 13,
        'final_eval_episodes': 100,  # number of episodes for final evaluation per seed
    }

def _default_hparams():
    """
    Default DQN hyperparameters.

    Returns:
        dict: hyperparameter set used when --search is not specified.
    """
    return {
        'hidden_dims': (128, 128),
        # Optimizer
        'optimizer': 'Adam',      # Supported: 'Adam', 'RMSprop'
        'lr': 1e-4,
        'buffer_size': 50_000,
        'batch_size': 64,

        # Epsilon schedule (updated)
        'init_epsilon': 1.0,
        'min_epsilon': 0.05,
        'schedule': 'exp',        # 'linear' or 'exp'
        'decay_steps': 300_000,   # reach most of decay ~ first half of a 600k-step run
        'tau_ratio': 0.3,         # only for 'exp'; tau = decay_steps * tau_ratio

        # Replay warmup / targets
        'n_warmup_batches': 5,    # epsilon decay is gated by warmup: batch_size * n_warmup_batches
        'target_update_every_steps': 100,

        # Optimization and loss stabilizers
        'use_huber': True,        # smooth L1 loss
        'double_dqn': True,       # online argmax with target network value
        'grad_clip': 10.0,        # clip global norm

        # RMSprop knobs (if optimizer='RMSprop')
        'rms_alpha': 0.95,
        'rms_eps': 1e-2,
        'rms_momentum': 0.9,
        'rms_centered': False,
        'weight_decay': 0.0,
    }

def _optimizer_from_name(name: str):
    """
    Resolve optimizer name into a torch optimizer class.

    Supported:
        - 'adam'    -> torch.optim.Adam
        - 'rmsprop' -> torch.optim.RMSprop
    """
    name = name.lower()
    if name == 'adam': return optim.Adam
    if name == 'rmsprop': return optim.RMSprop
    raise ValueError(f'Unsupported optimizer: {name}')

def _build_search_space():
    """
    Construct a grid of hyperparameter configurations for search.

    Returns:
        list[dict]: Each dict is a configuration tested in --search mode.
    """
    grid = {
        'hidden_dims': [(64,64), (128,128), (256,256)],
        'optimizer': ['RMSprop','Adam'],
        'lr': [1e-4, 3e-4, 5e-4],
        'buffer_size':  [50_000],
        'batch_size': [64],

        # Exploration schedule search
        'init_epsilon': [1.0],
        'min_epsilon': [0.01, 0.05],
        'schedule': ['linear', 'exp'],
        'decay_steps': [100_000, 300_000, 600_000],  # tie to 30–60% of total ~600k steps
        'tau_ratio': [0.25, 0.35],                   # only used for 'exp'

        'n_warmup_batches': [5, 10],
        'target_update_every_steps': [100],
    }
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    configs = []
    for combo in product(*values):
        cfg = {k: v for k, v in zip(keys, combo)}
        configs.append(cfg)
    return configs

def _format_hparams(hp: dict):
    """
    Compact JSON string for logging hyperparameter dictionaries.
    """
    return json.dumps(hp, separators=(',', ':'), sort_keys=True)

def _save_json(path, obj):
    """
    Save a Python object as pretty-printed JSON.

    Ensures the parent directory exists prior to writing.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _resolve_banana_exe(arg_path=None) -> str:
    """
    Resolve Unity executable path for the navigation environment.

    Strategy:
        - This function is currently hard-wired to a fixed Windows path to Banana.exe.
        - Update the path if your local layout differs.

    Raises:
        FileNotFoundError: If the hard-coded path does not exist.

    Returns:
        str: Absolute path to Banana.exe.
    """
    fixed_path = r"path\to\Banana_Windows_x86_64\Banana.exe"
    if not os.path.isfile(fixed_path):
        raise FileNotFoundError(f"Unity executable not found: {fixed_path}")
    return fixed_path

# ----------------------------
# Training orchestration
# ----------------------------

def run_single_training(run_dir, seeds, env_settings, hparams, env_exe_path, trial_id=None, trial_index=0):
    """
    Train a DQN agent on the Unity Banana environment for one or more seeds.

    High-level behavior
    - Builds factories (model, optimizer, replay buffer, policies) from hparams.
    - For each seed:
      * Instantiates a DQN agent and trains it, spawning its own worker env.
      * Logs exactly one line per episode from inside DQN.train().
      * Saves per-episode checkpoints to trials/<trial_or_default>/seed_<SEED>/checkpoints/.
      * Runs a 1-episode greedy evaluation after each training episode to build an eval curve.
      * Produces a per-seed evaluation plot (raw eval + 100-episode moving average).
      * Appends a per-seed row to plots_summary.csv with early stopping info.

    Args:
        run_dir (str): Base run directory.
        seeds (Iterable[int]): One or more integer seeds.
        env_settings (dict): Stop conditions and evaluation settings.
        hparams (dict): DQN hyperparameters.
        env_exe_path (str): Path to Unity Banana executable.
        trial_id (str|None): Grouping identifier under run_dir/trials/.
        trial_index (int): Index of the trial (used to derive a unique worker_id).

    Returns:
        tuple:
            - results_per_seed (list[np.ndarray])
            - final_scores (list[float])
            - best_eval_score (float)
            - summary_rows (list[dict])
    """
    results_per_seed = []
    final_scores = []
    best_eval_score = float('-inf')
    summary_rows = []

    # Build agent factories from hyperparameters
    value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=tuple(hparams['hidden_dims']))

    # Build optimizer factory with correct kwargs per optimizer
    opt_name = str(hparams['optimizer']).lower()
    if opt_name == 'rmsprop':
        value_optimizer_fn = lambda net, lr: optim.RMSprop(
            net.parameters(),
            lr=lr,
            alpha=hparams.get('rms_alpha', 0.99),
            eps=hparams.get('rms_eps', 1e-8),
            momentum=hparams.get('rms_momentum', 0.0),
            centered=hparams.get('rms_centered', False),
            weight_decay=hparams.get('weight_decay', 0.0),
        )
    elif opt_name == 'adam':
        value_optimizer_fn = lambda net, lr: optim.Adam(
            net.parameters(),
            lr=lr,
            betas=hparams.get('betas', (0.9, 0.999)),
            eps=hparams.get('adam_eps', 1e-5),
            weight_decay=hparams.get('weight_decay', 0.0),
            amsgrad=hparams.get('amsgrad', False),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {hparams['optimizer']}")
    value_optimizer_lr = float(hparams['lr'])

    # Training strategy factory with warmup gating
    # warmup_steps will be provided later inside DQN.train(), but we pass a lambda that accepts it.
    training_strategy_fn = lambda warmup_steps=0: EGreedyStrategy(
        init_epsilon=float(hparams['init_epsilon']),
        min_epsilon=float(hparams['min_epsilon']),
        decay_steps=int(hparams['decay_steps']),
        schedule=str(hparams.get('schedule', 'exp')),
        tau_ratio=float(hparams.get('tau_ratio', 0.3)),
        warmup_steps=int(warmup_steps),
    )
    evaluation_strategy_fn = lambda: GreedyStrategy()

    replay_buffer_fn = lambda: ReplayBuffer(max_size=int(hparams['buffer_size']),
                                            batch_size=int(hparams['batch_size']))
    n_warmup_batches = int(hparams['n_warmup_batches'])
    update_target_every_steps = int(hparams['target_update_every_steps'])

    # Unpack environment stopping criteria
    gamma = env_settings['gamma']
    max_minutes = env_settings['max_minutes']
    max_episodes = env_settings['max_episodes']
    goal_mean_100_reward = env_settings['goal_mean_100_reward']
    final_eval_episodes = int(env_settings.get('final_eval_episodes', 10))

    # Prepare CSV path and header (appended later)
    plots_summary_csv = os.path.join(run_dir, 'plots_summary.csv')
    if not os.path.isfile(plots_summary_csv):
        with open(plots_summary_csv, 'w', encoding='utf-8') as f:
            f.write('trial_id,seed,stop_episode,stop_cause,stop_mean_100_eval\n')

    for seed in seeds:
        # Each seed gets its own checkpoint directory
        agent_run_dir = os.path.join(run_dir, 'trials', str(trial_id) if trial_id else 'default', f'seed_{seed}')
        os.makedirs(agent_run_dir, exist_ok=True)

        # Instantiate the DQN agent
        agent = DQN(replay_buffer_fn,
                    value_model_fn,
                    value_optimizer_fn,
                    value_optimizer_lr,
                    training_strategy_fn,
                    evaluation_strategy_fn,
                    n_warmup_batches,
                    update_target_every_steps)
        agent.run_dir = agent_run_dir
        agent.trial_id = trial_id if trial_id else 'default'
        agent.hparams = hparams
        agent.final_eval_episodes = final_eval_episodes

        logging.info(f"[{agent.trial_id} | seed {seed}] Start training with hparams: {_format_hparams(hparams)}")

        # Derive a unique worker_id (avoid port conflicts across trials/seeds)
        base_worker_id = 1000*int(trial_index) + int(seed)

        # --- Train for this seed ---
        result, final_eval_score, training_time, wallclock_time, stop_episode, stop_cause, stop_mean_100_eval = agent.train(
            env_exe_path, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward, base_worker_id=base_worker_id
        )

        # Collect outputs
        results_per_seed.append(result)
        final_scores.append(final_eval_score)
        logging.info(f"[{agent.trial_id} | seed {seed}] Final eval score: {final_eval_score:.2f}")
        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score

        # --- Per-seed eval plot: per-episode raw rewards + 100-episode moving average ---
        try:
            plots_dir = os.path.join(run_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)

            # Column 2 is mean_100_eval_reward (per episode)
            eval_mean_100_curve = result[:, 2].astype(np.float32)
            mask = ~np.isnan(eval_mean_100_curve)

            # Raw per-episode evaluation rewards (each eval uses n_episodes=1)
            raw_eval = np.array(agent.evaluation_scores, dtype=np.float32)
            xs_raw = np.arange(raw_eval.shape[0])

            fig, ax = plt.subplots(1, 1, figsize=(15, 6), sharex=True)

            # Plot raw per-episode eval first
            if raw_eval.size > 0:
                ax.plot(xs_raw, raw_eval, color='steelblue', alpha=0.65, linewidth=1.25, label='Eval (per-episode)')

            # Overlay 100-episode moving average from result[:,2]
            xs = np.arange(len(eval_mean_100_curve))[mask]
            ys = eval_mean_100_curve[mask]
            if ys.size > 0:
                ax.plot(xs, ys, color='orange', linewidth=2.0, label='Eval (100-episode MA)')

            ax.set_title(f"Evaluation Performance (seed {seed}, trial='{agent.trial_id}')")
            ax.set_xlabel('Episodes'); ax.set_ylabel('Reward')
            ax.grid(True, linestyle=':', alpha=0.5)

            # Vertical line at the start of the first 100-episode window that reaches the goal
            goal = goal_mean_100_reward
            all_idxs = np.arange(len(eval_mean_100_curve))
            hit_idxs = np.where(mask & (all_idxs >= 99) & (eval_mean_100_curve >= goal))[0]
            if hit_idxs.size > 0:
                hit_idx = int(hit_idxs[0])
                start_idx = max(0, hit_idx - 99)
                ax.axvline(x=start_idx, linestyle=':', color='gray', linewidth=1.5, label='Goal window start')

            # Aesthetic: x ticks and limits based on both series
            if np.any(mask) or raw_eval.size > 0:
                last_idx_ma = int(np.where(mask)[0].max()) if np.any(mask) else -1
                last_idx_raw = int(xs_raw[-1]) if raw_eval.size > 0 else -1
                last_idx = max(last_idx_ma, last_idx_raw)
                ax.tick_params(axis='x', which='both', labelsize=10)
                ax.margins(x=0.02)
                step = max(1, last_idx // 10) if last_idx > 0 else 1
                ax.set_xticks(np.arange(0, last_idx + 1, step))
                ax.set_xlim(0, max(1, last_idx))
            else:
                logging.warning(f'[{agent.trial_id} | seed {seed}] No valid points to plot for this seed.')

            ax.legend(loc='upper left')
            fig.tight_layout(rect=[0, 0.03, 1, 0.98])

            plot_path = os.path.join(plots_dir, f"evaluation_mean100_{agent.trial_id}_seed_{seed}.png")
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"[{agent.trial_id} | seed {seed}] Per-seed evaluation plot saved to {plot_path}")
        except Exception as e:
            logging.warning(f'[{agent.trial_id} | seed {seed}] Per-seed plotting failed: {e}')

        # --- Append CSV summary row (only per-seed; no aggregation across seeds) ---
        summary_row = {
            'trial_id': agent.trial_id,
            'seed': int(seed),
            'stop_episode': int(stop_episode) if stop_episode is not None else '',
            'stop_cause': str(stop_cause) if stop_cause is not None else '',
            'stop_mean_100_eval': f"{stop_mean_100_eval:.6f}" if (stop_cause == 'goal_mean_100_reward' and stop_mean_100_eval is not None) else ''
        }
        summary_rows.append(summary_row)
        try:
            with open(plots_summary_csv, 'a', encoding='utf-8') as f:
                f.write(f"{summary_row['trial_id']},{summary_row['seed']},{summary_row['stop_episode']},{summary_row['stop_cause']},{summary_row['stop_mean_100_eval']}\n")
            logging.info(f"[{agent.trial_id} | seed {seed}] Summary appended to {plots_summary_csv}")
        except Exception as e:
            logging.warning(f"[{agent.trial_id} | seed {seed}] Failed writing plots summary CSV: {e}")

    return results_per_seed, final_scores, best_eval_score, summary_rows

# ----------------------------
# Main entry point
# ----------------------------

def main():
    """
    Main entry point for training and hyperparameter search on the Unity Banana environment.

    Command-line arguments:
        --search          Run a hyperparameter search over a predefined grid (default: False).
        --max-trials N    Maximum number of trials to run when --search is enabled (default: 10).

    Workflow:
        1) Create a timestamped run directory under results/navigation/.
        2) Initialize logging via _setup_logging(run_dir).
        3) Resolve the Unity executable path with _resolve_banana_exe().
        4) Load environment-level stop criteria from _default_env_settings().
        5) Branch by mode:
           a) Hyperparameter search (--search):
              - Build the grid with _build_search_space(), subsample up to --max-trials.
              - For each trial:
                • Train via run_single_training(...); one episode line is logged.
                • Append a line to hparam_search.csv with the trial’s final score and config.
                • Track the best trial by highest final score.
              - Save best_hparams.json summarizing the best trial.
           b) Default single run (no --search):
              - Use _default_hparams() and call run_single_training(...).

    Side effects (per run):
        results/navigation/run_YYYYMMDD_HHMMSS/
          - run.log                          Episode-by-episode logs.
          - plots_summary.csv                One row per seed per trial (stop episode/cause/metric).
          - plots/evaluation_mean100_*.png   Per-seed plot showing eval per-episode rewards and 100-ep MA.
          - trials/<trial_or_default>/seed_<SEED>/checkpoints/model.<ep_idx>.tar
          - hparam_search.csv                Only when --search is specified.
          - best_hparams.json                Only when --search is specified.

    Note:
        Unity env is now spawned in a separate worker process per seed/trial (no shared in-process env).
        Models run on CPU for compatibility with subprocess workers.
        Epsilon schedule supports linear/exp with warmup gating; decay rate matters and should be tuned to
        your total training steps (aim to hit the floor around 30–60% of the run).
    """
    # Parse CLI options
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', action='store_true', help='Enable hyperparameter search')
    parser.add_argument('--max-trials', type=int, default=10, help='Maximum number of trials to run (random subset of grid)')
    args = parser.parse_args()

    # Use the single DEFAULT_SEED
    seeds = (DEFAULT_SEED,)

    # Create run directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(RESULTS_DIR, 'navigation', f'run_{timestamp}')
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize logging
    _setup_logging(run_dir)
    logging.info(f'Results will be saved to: {run_dir}')
    logging.info(f'Seed: {seeds[0]}')

    # Resolve Unity executable
    banana_path = _resolve_banana_exe(None)
    logging.info(f'Unity exe: {banana_path}')

    # Load environment-level stop conditions and evaluation length
    env_settings = _default_env_settings()

    try:
        if args.search:
            # ------------- Hyperparameter search -------------
            all_cfgs = _build_search_space()

            # Reproducible random subset (or full grid if max-trials >= grid size)
            rng = np.random.default_rng(1234)
            cfgs = [all_cfgs[i] for i in (rng.choice(len(all_cfgs), size=args.max_trials, replace=False)
                    if args.max_trials < len(all_cfgs) else range(len(all_cfgs)))]

            logging.info(f'Hyperparameter search enabled. Trials to run: {len(cfgs)} (max-trials={args.max_trials})')

            # Prepare CSV to capture per-trial summary lines
            trials_csv = os.path.join(run_dir, 'hparam_search.csv')
            with open(trials_csv, 'w', encoding='utf-8') as f:
                f.write('trial_id,score,seed,hidden_dims,optimizer,lr,buffer_size,batch_size,init_epsilon,min_epsilon,schedule,decay_steps,tau_ratio,n_warmup_batches,target_update_every_steps\n')

            best = None

            for i, cfg in enumerate(cfgs, start=1):
                trial_id = f"trial_{i:03d}"
                logging.info(f'[{trial_id}] Starting trial with hparams: {_format_hparams(cfg)}')

                results_per_seed, final_scores, best_seed_score, summary_rows = run_single_training(
                    run_dir, seeds, env_settings, cfg, banana_path, trial_id=trial_id, trial_index=i
                )

                # With a single seed, final_scores has length 1
                score = float(final_scores[0]) if final_scores else float('-inf')
                logging.info(f'[{trial_id}] Final score: {score:.3f}')
                with open(trials_csv, 'a', encoding='utf-8') as f:
                    f.write(f"{trial_id},{score:.6f},{seeds[0]},\"{tuple(cfg['hidden_dims'])}\",{cfg['optimizer']},{cfg['lr']},{cfg['buffer_size']},{cfg['batch_size']},"  # noqa: E501
                            f"{cfg['init_epsilon']},{cfg['min_epsilon']},{cfg['schedule']},{cfg['decay_steps']},{cfg['tau_ratio']},{cfg['n_warmup_batches']},{cfg['target_update_every_steps']}\n")

                # Track best by highest final score
                if best is None or score > best['score']:
                    best = {'trial_id': trial_id, 'score': score, 'cfg': cfg}

            # Save best hyperparameters summary
            if best is not None:
                best_path = os.path.join(run_dir, 'best_hparams.json')
                _save_json(best_path, {
                    'trial_id': best['trial_id'],
                    'score': best['score'],
                    'seed': seeds[0],
                    'hyperparameters': best['cfg'],
                })
                logging.info(f"Best trial: {best['trial_id']} with score={best['score']:.3f}")
                logging.info(f"Best hyperparameters: {_format_hparams(best['cfg'])}")

            logging.info('Hyperparameter search finished.')
        else:
            # ------------- Default single-configuration run -------------
            logging.info('No hyperparameter search requested; running a single configuration.')
            hparams = _default_hparams()
            logging.info(f'Using hyperparameters: {_format_hparams(hparams)}')

            _ = run_single_training(run_dir, seeds, env_settings, hparams, banana_path, trial_id='default', trial_index=0)
            logging.info('Run finished.')
    finally:
        # Worker envs are owned and closed inside DQN.train()
        pass

if __name__ == "__main__":
    # Force PyTorch to CPU threads only if desired (optional, uncomment to cap threads)
    # torch.set_num_threads(1)
    main()
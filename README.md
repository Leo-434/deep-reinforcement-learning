# DRL Framework: A Modern, Unified Standard for Single & Multi-Agent Reinforcement Learning

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

[English](README.md) | [简体中文](README_zh.md)

**DRL Framework** is a highly modular, clean, and extensible reinforcement learning library built from the ground up to support both **Single-Agent** environments (`Gymnasium`) and **Multi-Agent** environments (`PettingZoo`). 

By abstracting away the boilerplate loops, buffer structures, and hardware checks, this framework provides a unified command-line entry point to train a massive suite of algorithms—from the classic DQN to various Multi-Agent cooperative architectures (like QMIX, MADDPG).

---

## 🎯 Key Features

- **Unified Training Pipeline**: A single `train.py` automates the instantiation of environments, buffers, networks, and algorithms dynamically based on a supplied YAML config.
- **Auto Hardware Detection**: Seamlessly detects and scales to the most optimal accelerator globally across Mac (`mps`), Nvidia (`cuda`), or fallback (`cpu`).
- **Standardized Multi-Agent Support (MARL)**: Implements true CTDE (Centralized Training, Decentralized Execution) concepts like global state mixing and double Q-learning action selection out-of-the-box using the `PettingZoo` parallel API.
- **High-Fidelity Tracking**: Automated `TensorBoard` logging, statistical episode reward smoothing, and end-of-run metric dumping (`.csv` and `.png`).

---

## 📦 Supported Algorithms

Our roadmap has meticulously implemented over 15 classic algorithms broken down into three logical phases of deep reinforcement learning architecture:

### 1. Value-Based Family (Discrete Control)
- **Base DQN**: Deep Q-Network
- **DDQN**: Double Q-Learning (solves overestimation)
- **Dueling DQN**: Dual-stream State-Value & Action-Advantage
- **PER DQN**: Prioritized Experience Replay
- **Noisy DQN**: Parametric exploration layers removing $\epsilon$-greedy
- **DRQN**: Recurrent Q-Networks for partially observable MDPs
- **Distributed RL**: QR-DQN (Quantile Regression) and C51 (Categorical)

### 2. Policy-Based Family (Continuous & Discrete Actor-Critic)
- **PG / NPG / PPG**: Standard and Natural/Phasic Policy Gradients
- **A2C**: Synchronous Advantage Actor-Critic
- **PPO**: Proximal Policy Optimization (Clip & KL penalty support)
- **DDPG**: Deep Deterministic Policy Gradient
- **TD3**: Twin Delayed DDPG (smooths target policies, clipping)
- **SAC**: Soft Actor-Critic (maximum entropy auto-tuning parameters)

### 3. MARL (Multi-Agent Reinforcement Learning)
- **IQL**: Independent Q-Learning (Decentralized baseline)
- **IPPO**: Independent PPO
- **QMIX**: Non-linear Value Decomposition Hypernetworks (CTDE)
- **MAPPO**: Multi-Agent PPO (Centralized Critic)
- **MADDPG**: Multi-Agent DDPG (Centralized Critic)

---

## 🚀 Installation

This project utilizes the Python packaging tool **`uv`** for fast virtual environment scaffolding and dependency resolution.

```bash
# 1. Install `uv` if you haven't already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone git@github.com:Leo-434/deep-reinforcement-learning.git
cd deep-reinforcement-learning

# 3. Quickstart & Sync Dependencies into a local virtual env
uv sync
```

---

## 💻 Quick Start & Usage

Forget writing loop boilerplate. Running an algorithm is as simple as aiming the `train.py` script at your desired YAML config file. 

The configuration folder `configs/` contains pre-tuned hyperparameter profiles.

**Example: Train PPO on CartPole-v1**
```bash
uv run python src/train.py --config configs/single_agent/policy_based/ppo_cartpole.yaml
```

**Example: Train the MARL QMIX Algorithm on MPE Simple Spread**
```bash
uv run python src/train.py --config configs/multi_agent/value_based/qmix_simple_spread.yaml
```

---

## 📁 Project Structure

```text
DRL/
├── configs/                # YAML hyperparameters
│   ├── single_agent/       # -> `value_based` / `policy_based`
│   └── multi_agent/        # -> `value_based` / `policy_based`
├── src/
│   ├── algorithms/         # DRL agent logic implementations
│   ├── envs/               # Environment wrappers (Gymnasium & PettingZoo)
│   ├── networks/           # Multi-layer perceptrons, Policies, and Mixers
│   ├── buffers/            # Replay (Vanilla, Prioritized, Rollout, MARL)
│   ├── utils/              # Hardware auto-detect, YAML parser, Logger
│   └── train.py            # 🚀 Unified Entry Point
├── logs/                   # Tensorboard logs & generated CSV/PNG
├── models/                 # Auto-saved checkpoints (.pth)
├── pyproject.toml          # `uv` definitions
└── README.md
```

---

## 📈 Monitoring Progress

Every training run automatically spawns a dedicated logging directory. 
To monitor your agents in real-time, spin up TensorBoard:

```bash
uv run tensorboard --logdir logs/
```

Navigate to `http://localhost:6006` in your browser to view the live reward scaling, loss functions, custom exploration decays (e.g. $\epsilon$, entropy).

## 🤝 Contributing

Contributions, issues and feature requests are welcome! 
If you find a bug, feel free to check the issues page.

## 📝 License

This project is [MIT](LICENSE) licensed.

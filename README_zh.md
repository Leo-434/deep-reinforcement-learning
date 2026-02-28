# DRL Framework: 现代化、统一的单/多智能体强化学习标准库

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

[English](README.md) | [简体中文](README_zh.md)

**DRL Framework** 是一个高度模块化、干净且可扩展的强化学习库，从零开始构建，原生支持**单智能体**环境（`Gymnasium`）和**多智能体**环境（`PettingZoo`）。

通过抽象掉繁杂的训练循环、经验回放池结构和硬件检测代码，本框架提供了一个统一的命令行入口。只需通过简单的 YAML 配置文件，即可训练海量的算法——从经典的 DQN 到各种多智能体协作架构（如 QMIX, MADDPG）。

---

## 📦 支持的算法矩阵

我们在深度强化学习架构的三个逻辑阶段中精心重构实现了超过 15 种经典算法：

### 1. 基于价值的家族 (离散控制)
- **Base DQN**: 深度 Q 网络
- **DDQN**: 双重 Q 学习 (解决过估计陷阱)
- **Dueling DQN**: 双路网络分离状态价值与动作优势
- **PER DQN**: 优先经验回放池机制
- **Noisy DQN**: 引入参数空间噪声，摒弃 $\epsilon$-贪婪
- **DRQN**: 递归 Q 网络，解决部分可观测环境 (POMDP)
- **分布式强化学习**: QR-DQN (分位数回归) 与 C51 (类别分布)

### 2. 基于策略的家族 (连续与离散 Actor-Critic)
- **PG / NPG / PPG**: 标准、自然与阶段策略梯度
- **A2C**: 同步优势演员-评论家
- **PPO**: 近端策略优化 (支持 Clip 裁剪与 KL 散度惩罚两版)
- **DDPG**: 深度确定性策略梯度
- **TD3**: 双延迟 DDPG (平滑目标策略噪音，双路 Critic 裁剪)
- **SAC**: 软演员-评论家 (最大熵，自动温度参数调整自适应)

### 3. MARL (多智能体强化学习)
- **IQL**: 独立 Q 学习 (去中心化分布式基线)
- **IPPO**: 独立 PPO
- **QMIX**: 非线性价值分解超网络 (CTDE 巅峰代表)
- **MAPPO**: 多智能体 PPO (带有全局视角辅助的集中式 Critic)
- **MADDPG**: 多智能体 DDPG (带有连续空间的集中式 Critic)

---

## 🚀 安装指南

本项目采用Python 包管理工具 **`uv`**，提供快速的虚拟环境构建与依赖解析。

```bash
# 1. 如果您还未安装 `uv`，请先通过 curl 一键安装：
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆本仓库到本地
git clone git@github.com:Leo-434/deep-reinforcement-learning.git
cd deep-reinforcement-learning

# 3. 快速启动并同步依赖到本地虚拟环境
uv sync
```

---

## 💻 快速上手与使用

忘记那些枯燥嵌套的循环代码吧！要运行任何一个算法，只需将 `train.py` 统帅脚本指向你希望的对应 YAML 配置文件。

配置文件夹 `configs/` 中已经包含了所有预先调优好的收敛超参数模板。

**示例 1：在 CartPole-v1 离散环境中训练 PPO**
```bash
uv run python src/train.py --config configs/single_agent/policy_based/ppo_cartpole.yaml
```

**示例 2：在 MPE Simple Spread 连续坐标场中训练多协同强策略 QMIX 混合超网络**
```bash
uv run python src/train.py --config configs/multi_agent/value_based/qmix_simple_spread.yaml
```

---

## 📁 核心架构及目录组织

```text
DRL/
├── configs/                # 声明式的 YAML 格式超参数库
│   ├── single_agent/       # -> `value_based` / `policy_based`
│   └── multi_agent/        # -> `value_based` / `policy_based`
├── src/
│   ├── algorithms/         # DRL 智能体核心逻辑推导实现 (基类高度抽象)
│   ├── envs/               # 环境封装容器 (底层代理对齐 Gymnasium 与 PettingZoo)
│   ├── networks/           # 多层感知机、Actor-Critic 策略分布网络与 MARL 混合中心网络
│   ├── buffers/            # 重播池基建 (基础、优先权重池、优势计算批处理缓冲区、MARL 全局视界并发池)
│   ├── utils/              # 硬件加速自动侦探、YAML 字典解析树、统一切割 TensorBoard 日志
│   └── train.py            # 🚀 全局统一无脑组装训练入口
├── logs/                   # 实时追踪日志走势与运行终止自动生存总结比对 (CSV/PNG)
├── models/                 # 设定保存频率自动按节点导出的的 PyTorch 模型权重文件 (.pth)
├── pyproject.toml          # `uv` 项目级工作区依赖定义流
└── README.md
```

---

## 📈 实战进程全时监控

每一轮调起的训练回合都会在 `logs/` 目录下自动生成一个按算法分发的包含最新数据走势的独立目录。 
你可以毫秒级启动 TensorBoard UI 层直接全景观测智能体的交互轨迹：

```bash
uv run tensorboard --logdir logs/
```

在你的浏览器中导航前往默认主页 `http://localhost:6006`，你就能以可视化仪表盘看到每个算法的实时学习奖励平滑爬坡曲线、Actor/Critic 损失函数走势，以及各种关键探索热度阈值的自由衰减（如 $\epsilon$ 或随机行动正态分布信息熵）。

## 🤝 参与开源贡献

非常欢迎任何方向的补丁拉取请求（PR）、Issue 漏洞修复以及前沿新功能探讨！
如果您敏锐地发现了基建 Bug ，欢迎随时点击 [Issues]() 提交建设性提案与工程规划。

## 📝 开源许可证使用说明

本项目采用 [MIT](LICENSE) 开源许可证。

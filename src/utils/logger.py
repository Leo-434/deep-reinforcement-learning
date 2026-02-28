import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    A simple wrapper around TensorBoard's SummaryWriter for structured logging.
    Now extended to save explicit CSV data and plotting graphs.
    """
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.ep_rewards = []
        
    def log_scalar(self, tag: str, value: float, step: int = None):
        """
        Log a scalar value directly to TensorBoard.
        """
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
        
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int = None):
        """
        Log multiple scalar values under a single main tag.
        """
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
    def log_episode_reward(self, reward: float, step: int = None):
        """
        Log episode mean reward, specifically tracking for final CSV and plotting.
        """
        if step is None:
            step = self.step
        self.ep_rewards.append({"step": step, "reward": reward})
        self.writer.add_scalar("rollout/ep_rew_mean", reward, step)

    def increment_step(self, amount: int = 1):
        self.step += amount
        
    def save_graphics(self):
        """
        Save the collected episode rewards to a CSV and generate a Matplotlib plot.
        """
        if not self.ep_rewards:
            return
            
        df = pd.DataFrame(self.ep_rewards)
        csv_path = os.path.join(self.log_dir, "training_rewards.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved raw reward tracking data to {csv_path}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(df["step"], df["reward"], alpha=0.3, label="Raw Reward")
        
        # Add smoothed line
        if len(self.ep_rewards) >= 10:
            smoothed = df["reward"].rolling(window=10).mean()
            plt.plot(df["step"], smoothed, color="orange", label="Smoothed (w=10)")
            
        plt.title("Training Episode Rewards")
        plt.xlabel("Global Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.log_dir, "training_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved reward plot to {plot_path}")
        
    def close(self):
        self.writer.close()

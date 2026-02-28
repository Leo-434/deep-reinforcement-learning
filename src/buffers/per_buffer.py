import numpy as np
import torch
from src.buffers.replay_buffer import ReplayBuffer

class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    Used for efficient O(log N) prioritized experience sampling.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        # The sum tree is stored in an array, size 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # Store exactly the data indices
        self.data_ptr = 0
        self.size = 0

    def add(self, p: float):
        tree_idx = self.data_ptr + self.capacity - 1
        self.update(tree_idx, p)
        
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, p: float):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float):
        parent_idx = 0
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            
            # If reach bottom, end search
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_idx]:
                    parent_idx = left_idx
                else:
                    v -= self.tree[left_idx]
                    parent_idx = right_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_p(self):
        return self.tree[0]


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) Buffer.
    Extends standard ReplayBuffer to include a SumTree for priorities.
    """
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device, alpha=0.6, beta=0.4, beta_increment=0.001):
        super().__init__(capacity, obs_dim, action_dim, device)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-5
        
        # We must align capacity to be a power of 2 for a perfect binary tree, 
        # or just simply use the given capacity if it's fine.
        # SumTree will handle given capacity but N=power_of_2 is most optimal.
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, obs, act, reward, next_obs, done):
        # Store in parent arrays
        super().add(obs, act, reward, next_obs, done)
        
        # New transitions get max priority to ensure they're seen at least once
        self.tree.add(self.max_priority)

    def sample(self, batch_size: int):
        idxs = np.zeros(batch_size, dtype=np.int32)
        tree_idxs = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        segment = self.tree.total_p / batch_size
        
        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            
            tree_idx, p, data_idx = self.tree.get_leaf(v)
            
            idxs[i] = data_idx
            tree_idxs[i] = tree_idx
            priorities[i] = p
            
        # Compute Importance Sampling (IS) Weights
        probs = priorities / self.tree.total_p
        # w_i = (N * P(i)) ^ -beta
        weights = np.power(self.size * probs, -self.beta)
        weights /= weights.max() # Normalize for stability
        
        batch = {
            "obs": torch.as_tensor(self.obs[idxs], dtype=torch.float32, device=self.device),
            "acts": torch.as_tensor(self.acts[idxs], dtype=torch.float32, device=self.device),
            "rews": torch.as_tensor(self.rews[idxs], dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(self.done[idxs], dtype=torch.float32, device=self.device),
            "weights": torch.as_tensor(weights, dtype=torch.float32, device=self.device),
            "tree_idxs": tree_idxs # Needed to update priorities later
        }
        return batch

    def update_priorities(self, tree_idxs, abs_td_errors):
        """
        After training step, update the priorities in the SumTree based on actual TD error.
        """
        abs_td_errors += self.epsilon
        clipped_errors = np.minimum(abs_td_errors, 1.0) # max priority = 1.0 roughly
        ps = np.power(clipped_errors, self.alpha)
        
        for idx, p in zip(tree_idxs, ps):
            self.max_priority = max(self.max_priority, p)
            self.tree.update(idx, p)

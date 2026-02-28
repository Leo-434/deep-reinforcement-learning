from abc import ABC, abstractmethod
import torch
import numpy as np

class BaseAlgorithm(ABC):
    """
    Base class for all Reinforcement Learning algorithms.
    """
    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.learning_steps = 0
        
    @abstractmethod
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Given an observation, returns an action.
        If evaluate=True, it should return a deterministic/greedy action.
        """
        pass
        
    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """
        Updates the algorithm's networks using data.
        Returns a dictionary of losses/metrics for logging.
        """
        pass
        
    @abstractmethod
    def save(self, filepath: str):
        """
        Saves the model weights to a file.
        """
        pass
        
    @abstractmethod
    def load(self, filepath: str):
        """
        Loads the model weights from a file.
        """
        pass

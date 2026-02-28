import gymnasium as gym

def make_env(env_id: str, seed: int = None, render_mode: str = None) -> gym.Env:
    """
    Creates and wraps a single-agent Gymnasium environment.
    Applies standard wrappers (e.g. RecordEpisodeStatistics) for reliable episode returns.
    """
    env = gym.make(env_id, render_mode=render_mode)
    # RecordEpisodeStatistics wrapper is useful to automatically store episodic returns and lengths in the `info` dict.
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
    return env

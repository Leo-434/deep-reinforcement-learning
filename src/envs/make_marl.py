from pettingzoo import ParallelEnv

def make_marl_env(env_name: str, **kwargs) -> ParallelEnv:
    """
    Creates a PettingZoo multi-agent environment.
    Automatically imports the correct sub-module based on the environment name (e.g., mpe.simple_spread_v3).
    Returns a PettingZoo ParallelEnv (which is more akin to Gym's vector environments syntax).
    """
    # Ex: if env_name == "mpe.simple_spread_v3"
    try:
        family, name = env_name.split('.')
        
        # Dynamically import the PettingZoo environment
        # e.g. import pettingzoo.mpe.simple_spread_v3 as env_module
        import importlib
        env_module = importlib.import_module(f"pettingzoo.{family}.{name}")
        
    except ValueError:
        raise ValueError("env_name must be in the format 'family.name', e.g., 'mpe.simple_spread_v3'")
    except ImportError as e:
        raise ImportError(f"Failed to import PettingZoo environment {env_name}. Error: {e}")

    # Initialize the parallel environment
    env = env_module.parallel_env(**kwargs)
    
    return env

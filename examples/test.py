from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import run_experiment
import random

env_path = '../env/AnimalAI'
worker_id = random.randint(1, 100)
arena_config_in = ArenaConfig('configs/1-Food_my.yaml')
base_dir = 'models/dopamine'
gin_files = ['configs/rainbow.gin']


def create_env_fn():
    env = AnimalAIEnv(environment_filename=env_path,
                      worker_id=worker_id,
                      n_arenas=1,
                      arenas_configurations=arena_config_in,
                      docker_training=False,
                      retro=False,
                      inference=True,
                      resolution=80)
    return env


env = create_env_fn()
print(env.action_space)
for i in range(100):
    action = [1,0]
    if i > 100:
        action = [0,1]

    obs,reward,done,info = env.step(action)
    print(obs[1])

env.close()

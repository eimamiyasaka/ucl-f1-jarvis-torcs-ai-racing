from torcs_rl_env import TorcsRLEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import traceback

def make_env():
    env = TorcsRLEnv(port=3001, max_steps=1000)
    env = Monitor(env)
    return env

try:
    print('Creating DummyVecEnv...')
    env = DummyVecEnv([make_env])
    print('Success!')
    obs = env.reset()
    print('Obs shape:', obs.shape)
    env.close()
    print('Done!')
except Exception as e:
    print('ERROR:', type(e).__name__)
    print(str(e))
    traceback.print_exc()

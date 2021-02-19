from gym_novel_gridworlds.novelty_wrappers import inject_novelty
import os
import time
import uuid 
import argparse

import gym
import gym_novel_gridworlds
from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap

import numpy as np
from stable_baselines.common import callbacks

from stable_baselines.common.env_checker import check_env

from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines import A2C
from stable_baselines.gail import ExpertDataset

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env

from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy


class RenderOnEachStep(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    """

    def __init__(self, env):
        super(RenderOnEachStep, self).__init__()
        self.env = env

    def _on_step(self):
        # print("observation: ", self.env.observation(None))
        # print("observation: ", self.env.get_observation())
        self.env.render()
        # time.sleep(0.5)


class SaveModelandInjectNovelty(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    """

    def __init__(self, env, check_freq, save_freq, log_dir, model_name, step_num, novelty_name, novelty_difficulty, novelty_arg1, novelty_arg2):
        super(SaveModelandInjectNovelty, self).__init__()
        
        self.step_num = step_num
        self.env = env
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, model_name)
        self.best_mean_reward = -np.inf
        
        self.novelty_name = novelty_name
        self.novelty_difficulty = novelty_difficulty
        self.novelty_arg1 = novelty_arg1
        self.novelty_arg2 = novelty_arg2

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
        if self.n_calls == self.step_num:
            self.env = inject_novelty(self.env, self.novelty_name, self.novelty_difficulty, self.novelty_arg1, self.novelty_arg2)
            print ("novel env.observation_space = {}".format(self.env.observation_space))
        
        # save best model every "save_freq" steps
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path + '_' + str(self.n_calls))
            #             # Retrieve training reward
            # x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            # if len(x) > 0:
            #     # Mean training reward over the last 100 episodes
            #     mean_reward = np.mean(y[-100:])

            #     # New best model, you could save the agent here
            #     if mean_reward > self.best_mean_reward:
            #         self.best_mean_reward = mean_reward
            #         print("Saving new best model to {}".format(self.save_path))
        # Save the first model
        if self.n_calls == 50:
            self.model.save(self.save_path + '_' + str(self.n_calls))


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("-R","--render", default = False, help="Want to render or not (False or True)", type = bool)
    ap.add_argument("-E", "--environment", default = 'NovelGridworld-Bow-v0', help = "Environment to train.")

    ap.add_argument("-N", "--novelty_name", default= '', help="Novelty to inject. Current version supports 'firewall', 'remapaction', 'logincrease'")
    ap.add_argument("-D", "--novelty_difficulty", default= 'hard', help="Type of Novelty to inject. Current version supports firewall: 'hard'; remapaction: 'hard'; breakincrease: 'stick' ")
    ap.add_argument("-N1", "--novelty_arg1", default= '', help="Type of Novelty to inject. Current version supports firewall: 'hard'; remapaction: 'hard'; breakincrease: 'stick' ")
    ap.add_argument("-N2", "--novelty_arg2", default= '', help="Type of Novelty to inject. Current version supports firewall: 'hard'; remapaction: 'hard'; breakincrease: 'stick' ")
   
    ap.add_argument("-I", "--inject", default=1200000, help="Number of trials (steps) to run before injecting novelty", type=int)
    ap.add_argument("-C", "--check_best", default=10000, help="Number of (steps) to run check and save best model", type=int)
    ap.add_argument("-M", "--num_models", default=20, help="Number of models to save for testing later", type=int)
    
    ap.add_argument("-print_output", default="", help="print stuff")
    args = vars(ap.parse_args())
    
    # generate unique ID for every run
    id = uuid.uuid1() 
    
    env_id = args['environment'] 
    timesteps = 2*args['inject']  # 200000
    model_name = str(env_id) + str(args['novelty_name']) + '_' + str(id.hex)
    exp_dir = 'results' + os.sep + str(env_id) + os.sep + str(args['novelty_name']) + '_'+ str(id.hex)
    os.makedirs(exp_dir, exist_ok=True)
    
    env = gym.make(env_id) # make the environment
    env = LimitActions(env, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'}) # limit actions for easy training
    env = LidarInFront(env) # generate the observation space using LIDAR sensors
    env = Monitor(env, exp_dir) # for monitoring and saving the results
    # callback = RenderOnEachStep(env)
    # callback for the saving best model and injectiing novelty
    save_freq = timesteps//args['num_models']
    callback = SaveModelandInjectNovelty(env, args['check_best'], save_freq, exp_dir, 'best_model', args['inject'], args['novelty_name'], args['novelty_difficulty'], args['novelty_arg1'], args['novelty_arg2'])
    check_env(env, warn=True)
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])
    tensorboard_name = "./ppo2_"+str(model_name)+'_tensorboard'+os.sep
    model = PPO2(MlpPolicy, env, n_steps=256, verbose=1, tensorboard_log=tensorboard_name)
    # model = A2C(MlpPolicy, env, n_steps=150, verbose=1, tensorboard_log="./a2c_bow_v_0_tensorboard/")

    # env = DummyVecEnv([lambda: env])
    # model = PPO2.load('NovelGridworld-Bow-v0_200000_8beams0filled11hypotenuserange3items_in_360degrees_best_model', env)

    # Pretrain the model from human recored dataset
    # specify `traj_limitation=-1` for using the whole dataset
    # if pretrain:
    #     dataset = ExpertDataset(expert_path='expert_NovelGridworld-Bow-v0_10demos.npz', traj_limitation=-1, batch_size=128)
    #     model.pretrain(dataset, n_epochs=2000)
    #     model.save(log_dir + os.sep + model_code)

    # model.learn(total_timesteps=timesteps)
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(exp_dir, 'last_model'))

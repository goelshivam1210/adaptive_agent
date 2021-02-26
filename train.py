################################
# Author: Shivam Goel
# email: shivam.goel@tufts.edu
################################

from gym_novel_gridworlds.novelty_wrappers import inject_novelty
import os
import time
import uuid 
import argparse
import csv
import copy

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

MAX_TIMESTEPS = 200

def make_env_with_constraints(env_id, novelty_family = None, inject = False):
    # print ()
    env2 = gym.make(env_id)
    env2.unbreakable_items.add('crafting_table') # Make crafting table unbreakable for easy solving of task.
    env2 = LimitActions(env2, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'}) # limit actions for easy training
    env2 = LidarInFront(env2) # generate the observation space using LIDAR sensors
    # print(env.unbreakable_items)
    env2.reward_done = 1000
    env2.reward_intermediate = 50
    if inject:
        env2 = inject_novelty(env2, novelty_family[0], novelty_family[1], novelty_family[2], novelty_family[3])
        if novelty_family[0] == 'breakincrease':
            print ("Break increase novelty injected::: self.env.itemtobreakmore = {}".format(env2.itemtobreakmore))
        if novelty_family[0] == 'remapaction':
            print("Action remap novelty check:: self.env.limited_actions_id = {}".format(env2.limited_actions_id))
    # if novelty_family[0]
    check_env(env2, warn=True) # check the environment    
    return env2
    
def run_tests(env_id, model, eval_eps, novelty_name, timestep, i, novelty_family, id):
    if i >= int(args['num_models'])//2: # check whether to load a novelty injected environment
        inject = True
    else:
        inject = False
    
    env2 = make_env_with_constraints(env_id, novelty_family, inject)# get the environment
    
    ctr = 0 # for counting the success
    total_reward = 0
    total_steps = 0
    for episode in range(eval_eps):
        rew_eps = 0
        count = 0
        obs = env2.reset()
        for step in range(MAX_TIMESTEPS):
            action, _states = model.predict(obs)
            obs, reward, done, info = env2.step(action)
            rew_eps+=reward
            count +=1
            if args['render']:
                env2.render()
            if done:
                if env2.inventory_items_quantity[env2.goal_item_to_craft] >= 1: # success measure(goal achieved)
                    ctr+=1
                # print ("The agent crafted a bow = {}".format(done))
                count = step
                break
        data = [timestep, rew_eps, count]
        total_reward += rew_eps
        total_steps += count
        # print ("After episode {} performance = {}".format(episode, data))
        save_results(data, novelty_name, tag = None, id = id)
    # compute and save average reward and average steps for the whole test
    avg_rew = float(total_reward/eval_eps)
    avg_steps = float(total_steps/eval_eps)
    D = [timestep, avg_rew, avg_steps, ctr]
    # print ("After all the episodes mean = {}".format(D))
    save_results(D, novelty_name, tag = 'average', id = id)

def save_results (data, novelty_name, tag, id):

    os.makedirs("results" + os.sep + args['environment'] + os.sep + "test_results", exist_ok=True)
    db_file_name = "results" + os.sep + args['environment'] + os.sep + "test_results" + os.sep + str(novelty_name) + "_" + str(id.hex)+ ".csv"
    if tag == 'average':
        db_file_name = "results" + os.sep + args['environment'] + os.sep + "test_results" + os.sep + str(novelty_name) + "_" + str(id.hex)+ "_mean.csv"
    with open(db_file_name, 'a') as f: # append to the file created
        writer = csv.writer(f)
        writer.writerow(data)


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


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("-R","--render", default = False, help="Want to render or not (False or True)", type = bool)
    ap.add_argument("-E", "--environment", default = 'NovelGridworld-Bow-v0', help = "Environment to train.")

    ap.add_argument("-N", "--novelty_name", default= '', help="Novelty to inject. Current version supports 'firewall', 'remapaction', 'logincrease'")
    ap.add_argument("-D", "--novelty_difficulty", default= 'hard', help="Type of Novelty to inject. Current version supports firewall: 'hard'; remapaction: 'hard'; breakincrease: 'stick' ")
    ap.add_argument("-N1", "--novelty_arg1", default= '', help="Type of Novelty to inject. Current version supports firewall: 'hard'; remapaction: 'hard'; breakincrease: 'stick' ")
    ap.add_argument("-N2", "--novelty_arg2", default= '', help="Type of Novelty to inject. Current version supports firewall: 'hard'; remapaction: 'hard'; breakincrease: 'stick' ")
   
    ap.add_argument("-I", "--inject", default=2000000, help="Number of trials (steps) to run before injecting novelty", type=int)
    ap.add_argument("-C", "--check_best", default=10000, help="Number of (steps) to run check and save best model", type=int)
    ap.add_argument("-M", "--num_models", default=20, help="Number of models to save for testing later", type=int)
    ap.add_argument("-T", "--num_tests", default=25, help="Number of tests to conduct", type=int)
    
    ap.add_argument("-print_output", default="", help="print stuff")
    args = vars(ap.parse_args())

    id = uuid.uuid1() # generate unique ID for every run
    novelty_family = [args['novelty_name'], args['novelty_difficulty'], args['novelty_arg1'], args['novelty_arg2']]
    timesteps = 2*args['inject']  # 200000
    model_name = args['environment'] + str(args['novelty_name']) + '_' + str(id.hex)
    exp_dir = 'results' + os.sep + args['environment'] + os.sep + str(args['novelty_name']) + '_'+ str(id.hex)
    os.makedirs(exp_dir, exist_ok=True)
    env = make_env_with_constraints(args['environment'])
    env = Monitor(env, exp_dir) # for monitoring and saving the results
    # callback = RenderOnEachStep(env)
    # callback for the saving best model and injectiing novelty
    save_freq = timesteps//args['num_models']
    # callback = SaveModelandInjectNovelty(env, args['check_best'], save_freq, exp_dir, 'model', args['inject'], args['novelty_name'], args['novelty_difficulty'], args['novelty_arg1'], args['novelty_arg2'])
    check_env(env, warn=True)
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])
    # tensorboard_name = "./ppo2_"+str(model_name)+'_tensorboard'+os.sep
    # model = PPO2(MlpPolicy, env, n_steps=256, verbose=1, tensorboard_log=tensorboard_name)
    model = PPO2(MlpPolicy, env, n_steps=256, verbose=1)
    
    for i in range (args['num_models']):
        if i == int(args['num_models'])//2:
            print ("Injecting Novelty now...")
            env = make_env_with_constraints(args['environment'], novelty_family, True)
        env.needs_reset = False # HACK: otherwise throws runtime error that environment needs reset (Monitor wrapper error!)
        model.learn(total_timesteps=timesteps)
        # model.learn(total_timesteps=20)
        file_name = model_name + "_" + str(i)
        # model.save(file_name)
        run_tests(args['environment'], model, args['num_tests'], novelty_family[0], timesteps*(i+1), i, novelty_family, id)
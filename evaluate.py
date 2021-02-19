from posix import listdir
from gym_novel_gridworlds.novelty_wrappers import inject_novelty
import os
import time
import uuid 
import argparse
import csv

import gym
import gym_novel_gridworlds
from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap

# import numpy as np
# from stable_baselines.common import callbacks

# from stable_baselines.common.env_checker import check_env

from stable_baselines import PPO2
# from stable_baselines import DQN
# from stable_baselines import A2C
# from stable_baselines.gail import ExpertDataset

# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.common import make_vec_env

# from stable_baselines.bench import Monitor
# from stable_baselines.common.callbacks import BaseCallback
# from stable_baselines.results_plotter import load_results, ts2xy

MAX_STEPS = 5000

def save_results (data, novelty_name, tag):

    os.makedirs("results" + os.sep + args['environment'] + os.sep + "test_results", exist_ok=True)
    db_file_name = "results" + os.sep + args['environment'] + os.sep + "test_results" + os.sep + str(novelty_name) + ".csv"
    if tag == 'average':
        db_file_name = "results" + os.sep + args['environment'] + os.sep + "test_results" + os.sep + str(novelty_name) + "_mean.csv"
    with open(db_file_name, 'a') as f: # append to the file created
        writer = csv.writer(f)
        writer.writerow(data)
    
def run(novelty_family, env_id, eval_eps):

    dir_name = 'results/'+str(env_id)
    print ("Environment = {}".format(env_id))
    env = gym.make(env_id) # make the environment
    env = LidarInFront(env) # wrap the observation space in the environment
    # env = DummyVecEnv([lambda: env])
    if env_id == 'NovelGridworld-Bow-v0':
        env = LimitActions(env, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'}) # this is hardcoded for now
    # print (os.listdir(dir_name))
    for name in os.listdir(dir_name):
        # name = 'remapaction_4gbdsu34ddhjakndj'
        # print ("each name = {}".format(name))
        novelty_name = name.split('_')[0] # get novelty name "remapaction_fjbwuyguyqw12324213"
        print ("novelty_name = {}".format(novelty_name))
        if novelty_name == novelty_family[0]: # we only want all the experiments of the same novelty
            sub_dir_name = dir_name+os.sep+str(name)
            print ("sub_dir_name = {}".format(sub_dir_name))
            for filename in listdir(sub_dir_name):
                model_file_name = filename.split('.')[0] # filename = 'best_model_1200.zip'
                # filter out the unwanted files
                filtered_model_files = model_file_name.split('_')
                if len(filtered_model_files) >=3:
                    model_path = sub_dir_name+os.sep+model_file_name
                    print ("model_path = {}".format(model_path))
                    # file_name = dir_name+os.sep+str(name)+os.sep+"*.zip"
                    # print (file_name)
                    # now we have the file name and quite possibly the path
                    # inject novelty for corresponding models
                    file_name_split = model_file_name.split('_') # model_file_name = 'best_model_1200' 
                    if int(file_name_split[-1]) >= args['inject']:
                            env = gym.make(env_id) # make the environment
                            env = LidarInFront(env) # wrap the observation space in the environment
                            env = inject_novelty(env, novelty_family[0], novelty_family[1], novelty_family[2], novelty_family[3])

                    # time.sleep(2)
                    model = PPO2.load(model_path) # load the model
                    timestep = file_name_split[-1]
                    run_tests(env, model, eval_eps, name, timestep) # runs the tests and writes the result

def run_tests(env, model, eval_eps, novelty_name, timestep):
    ctr = 0
    total_reward = 0
    total_steps = 0
    for episode in range(eval_eps):
        rew_eps = 0
        count = 0
        obs = env.reset()
        for step in range(MAX_STEPS):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            rew_eps+=reward
            if args['render']:
                env.render()
            if done:
                ctr+=1
                count = step
                break
        data = [timestep, rew_eps, count]
        total_reward += rew_eps
        total_steps += count
        save_results(data, novelty_name, tag = None)
    # compute and save average reward and average steps for the whole test
    avg_rew = float(total_reward/eval_eps)
    avg_steps = float(total_steps/eval_eps)
    D = [timestep, avg_rew, avg_steps]
    save_results(D, novelty_name, tag = 'average')
    
# Things to take care of



# 4. get all the results file and plot  


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-T", "--num_tests", default=25, help="Number of tests to conduct", type=int)
    ap.add_argument("-R","--render", default = False, help="Want to render or not (False or True)", type = bool)
    ap.add_argument("-E", "--environment", default = 'NovelGridworld-Bow-v0', help = "Environment to train.")

    ap.add_argument("-N", "--novelty_name", default= '', help="Novelty to inject. Current version supports 'firewall', 'remapaction', 'logincrease'")
    ap.add_argument("-D", "--novelty_difficulty", default= 'hard', help="Type of Novelty to inject. Current version supports firewall: 'hard'; remapaction: 'hard'; breakincrease: 'stick' ")
    ap.add_argument("-N1", "--novelty_arg1", default= '', help="Type of Novelty to inject. Current version supports firewall: 'hard'; remapaction: 'hard'; breakincrease: 'stick' ")
    ap.add_argument("-N2", "--novelty_arg2", default= '', help="Type of Novelty to inject. Current version supports firewall: 'hard'; remapaction: 'hard'; breakincrease: 'stick' ")
   
    ap.add_argument("-I", "--inject", default=1100000, help="Number of trials (steps) to run before injecting novelty", type=int)
    ap.add_argument("-print_output", default="", help="print stuff")
    args = vars(ap.parse_args())

    # generate the folders to search
    novelty_family = [args['novelty_name'], args['novelty_difficulty'], args['novelty_arg1'], args['novelty_arg2']]
    # run the tests and save results
    run(novelty_family, args['environment'], args['num_tests'])
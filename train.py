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
    if i >= int(args['num_models'])//2:
        inject = True
    else:
        inject = False
    env2 = make_env_with_constraints(env_id, novelty_family, inject)
    ctr = 0
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
                if env2.inventory_items_quantity[env2.goal_item_to_craft] >= 1:
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
            check_env(self.env, warn=True)
            # if self.novelty_name == 'breakincrease':
            #     print ("Break increase novelty injected::: self.env.itemtobreakmore = {}".format(self.env.itemtobreakmore))
            # if self.novelty_name == 'remapaction':
            #     print("Action remap novelty check:: self.env.limited_actions_id = {}".format(self.env.limited_actions_id))
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
        # print ("self.env observation space = {} ".format(self.env.observation_space))


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
    ap.add_argument("-T", "--num_tests", default=25, help="Number of tests to conduct", type=int)
    
    ap.add_argument("-print_output", default="", help="print stuff")
    args = vars(ap.parse_args())
    
    # generate unique ID for every run
    id = uuid.uuid1() 
    
    novelty_family = [args['novelty_name'], args['novelty_difficulty'], args['novelty_arg1'], args['novelty_arg2']]
    # env_id = args['environment'] 
    timesteps = 2*args['inject']  # 200000
    model_name = args['environment'] + str(args['novelty_name']) + '_' + str(id.hex)
    exp_dir = 'results' + os.sep + args['environment'] + os.sep + str(args['novelty_name']) + '_'+ str(id.hex)
    os.makedirs(exp_dir, exist_ok=True)
    env = make_env_with_constraints(args['environment'])
    # env = gym.make(env_id) # make the environment
    # env.unbreakable_items.add('crafting_table') # Make crafting table unbreakable for easy solving of task.
    # env = LimitActions(env, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'}) # limit actions for easy training
    # env = LidarInFront(env) # generate the observation space using LIDAR sensors
    # print(env.unbreakable_items)
    # env.reward_done = 1000
    # env.reward_intermediate = 50
    env = Monitor(env, exp_dir) # for monitoring and saving the results
    # callback = RenderOnEachStep(env)
    # callback for the saving best model and injectiing novelty
    save_freq = timesteps//args['num_models']
    # callback = SaveModelandInjectNovelty(env, args['check_best'], save_freq, exp_dir, 'model', args['inject'], args['novelty_name'], args['novelty_difficulty'], args['novelty_arg1'], args['novelty_arg2'])
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
    # model.learn(total_timesteps=timesteps, callback=callback)
    # model.load("base_model_9")
    # run_tests(env.env_id, model, 20, 'default',14)
    # import sys
    # sys.exit(1)

    
    for i in range (args['num_models']):
        if i == int(args['num_models'])//2:
            print ("Injecting Novelty now...")
            env = make_env_with_constraints(args['environment'], novelty_family, True)
        env.needs_reset = False # HACK: otherwise throws runtime error that environment needs reset (Monitor wrapper error!)
        model.learn(total_timesteps=200000)
        # model.learn(total_timesteps=20)
        file_name = model_name + "_" + str(i)
        # model.save(file_name)
        run_tests(args['environment'], model, args['num_tests'], novelty_family[0], 200000*(i+1), i, novelty_family, id)
        # run_tests(args['environment'], model, args['num_tests'], novelty_family[0], 200*(i+1))

    # model.learn(total_timesteps=1000000)

    # model.save(os.path.join(exp_dir, 'last_model'))

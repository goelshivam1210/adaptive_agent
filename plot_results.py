#!/usr/bin/env python
# coding: utf-8

import os, csv

import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')


results_path = r"C:\Users\GyanT\Documents\GitHub\adaptive_agent_data\data_from_server_2\test_results"
novelty_injection_point = 2000000
novelty_used_for_pre_novelty_plot = 'breakincrease'


novelty_files = {}

for root, dirs, filenames in os.walk(results_path):
    for a_file in filenames:
        if a_file.endswith("mean.csv"):
            print(a_file)
            
            novelty = a_file.split('_')[0]
            
            novelty_files.setdefault(novelty, [])
            novelty_files[novelty].append(a_file)


pre_novelty_plot = True
for novelty in novelty_files:
#     print("novelty: ", novelty)
    
    timestep_rewards = {}
    
    for result_file in novelty_files[novelty]:
        
        with open(results_path+os.sep+result_file, mode='r') as infile:
            reader = csv.reader(infile)
            for line in reader:
#                 print(line)
                
                timestep_rewards.setdefault(int(line[0]), [])
                timestep_rewards[int(line[0])].append(float(line[1])) # use 2 for steps instead of 1
    
#     print("timestep_rewards: ", timestep_rewards)
    
    for timestep in timestep_rewards:
        
        timestep_rewards[timestep] = {'mean': np.mean(timestep_rewards[timestep]),
                                     'std': np.std(timestep_rewards[timestep])}
    
#     print("timestep_rewards: ", timestep_rewards)
    
    if pre_novelty_plot:
        pre_timesteps = []
        pre_rewards = []
        pre_rewards_std = []
    
    post_timesteps = []
    post_rewards = []
    post_rewards_std = []
    for timestep in sorted(timestep_rewards):
        if pre_novelty_plot and novelty == novelty_used_for_pre_novelty_plot:
            if timestep <= novelty_injection_point:
                pre_timesteps.append(timestep)
                pre_rewards.append(timestep_rewards[timestep]['mean'])
                pre_rewards_std.append(timestep_rewards[timestep]['std'])
        
        if timestep >= novelty_injection_point:
            post_timesteps.append(timestep)
            post_rewards.append(timestep_rewards[timestep]['mean'])
            post_rewards_std.append(timestep_rewards[timestep]['std'])
    
    if pre_novelty_plot and novelty == novelty_used_for_pre_novelty_plot:
        pre_novelty_plot = False
        
        
#     print("timesteps: ", timesteps)
#     print("rewards: ", rewards)
#     print("rewards_std: ", rewards_std)
    
#     plt.figure(figsize=(5, 5))
    plt.errorbar(post_timesteps, post_rewards, yerr=post_rewards_std, fmt='-o', label=novelty)
    
#     std_pos = np.array(post_rewards) + np.array(post_rewards_std)
#     std_neg = np.array(post_rewards) - np.array(post_rewards_std)
#     plt.plot(post_rewards, label = 'V.Q.L.', color = 'forestgreen')
#     plt.fill(std_pos, color = 'honeydew')
#     plt.fill(std_neg, color = 'honeydew')
    
#     plt.xticks(range(1, 5))
#     plt.ylim(0, 1)

plt.axvline(x=novelty_injection_point, linewidth=3.0, color='pink')
    
plot_lines = plt.errorbar(pre_timesteps, pre_rewards, yerr=pre_rewards_std, fmt='-o')

plt.legend(title='post-novelty', title_fontsize=12, loc='upper right')

props = dict(boxstyle='round', facecolor='w', alpha=0.2)
plt.text(600000, 215, 'pre-novelty', fontsize = 12, bbox=props)
plt.grid(True)
plt.ylabel("Average cumulative reward")
plt.xlabel("Timesteps")
plt.show()
plt.savefig("results.png", bbox_inches='tight', dpi=100)

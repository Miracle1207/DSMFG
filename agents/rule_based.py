import copy
import numpy as np
import torch
import os,sys
import wandb
import time
sys.path.append(os.path.abspath('../..'))
#
# from agents.log_path import make_logpath
# from datetime import datetime
# from tensorboardX import SummaryWriter
# from env.evaluation import save_parameters
#
# import pygame
torch.autograd.set_detect_anomaly(True)

def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class rule_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args

        # get the action max
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0]
        self.on_policy = True

    def get_action(self, global_obs_tensor, private_obs_tensor, gov_action=None, agent_name="government"):
        if agent_name == "government":
            # gov_action = np.array([0.23, 0.01, 0.5, 0.01, 0.4494 / 0.3])
            gov_action = np.array([0., 0.0, 0., 0., 0.])
            return gov_action
        elif agent_name == "household":
            house_action = np.random.random((self.args.n_households, self.envs.households.action_space.shape[1]))
            return house_action
        
    def train(self, transition):
        return 0,0

    
    def save(self, dir_path, step=0):
        pass
import os

import copy
import torch
import numpy as np
from agents.models import mlp_net
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

class ppo_agent:
    def __init__(self, envs, args, agent_name="household"):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name
        if agent_name == "household":
            self.obs_dim = self.envs.households.observation_space.shape[0]
            self.action_dim = self.envs.households.action_space.shape[1]
        elif agent_name == "government":
            self.obs_dim = self.envs.government.observation_space.shape[0]
            self.action_dim = self.envs.government.action_space.shape[0]
        else:
            print("AgentError: Please choose the correct agent name!")

        # if use the cuda...
        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.net = mlp_net(state_dim=self.obs_dim, num_actions=self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.p_lr,eps=1e-5)
        lambda_function = lambda epoch: 0.97 ** (epoch // 10)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_function)
        self.on_policy = True

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage_output = torch.tensor(np.array(advantage_list), dtype=torch.float)
        # batch adv norm
        norm_advantage = (advantage_output - torch.mean(advantage_output))/torch.std(advantage_output)
        return norm_advantage
    
    def train(self, transition_dict):
        sum_loss = torch.tensor([0.,0.], dtype=torch.float32).to(self.device)
        global_obses = torch.tensor(np.array(transition_dict['global_obs']), dtype=torch.float32).to(self.device)
        private_obses = torch.tensor(np.array(transition_dict['private_obs']), dtype=torch.float32).to(self.device)
        gov_actions = torch.tensor(np.array(transition_dict['gov_action']), dtype=torch.float32).to(self.device)
        house_actions = torch.tensor(np.array(transition_dict['house_action']), dtype=torch.float32).to(self.device)
        gov_rewards = torch.tensor(np.array(transition_dict['gov_reward']), dtype=torch.float32).to(self.device).unsqueeze(-1)
        house_rewards = torch.tensor(np.array(transition_dict['house_reward']), dtype=torch.float32).to(self.device)
        next_global_obses = torch.tensor(np.array(transition_dict['next_global_obs']), dtype=torch.float32).to(self.device)
        next_private_obses = torch.tensor(np.array(transition_dict['next_private_obs']), dtype=torch.float32).to(self.device)
        inverse_dones = torch.tensor([x - 1 for x in np.array(transition_dict['done'])], dtype=torch.float32).to(self.device).unsqueeze(-1)
        
        if self.agent_name == "government":
            obs_tensor = global_obses
            action_tensor = gov_actions
            reward_tensor = gov_rewards
            next_obs_tensor = next_global_obses
        elif self.agent_name == "household":
            obs_tensor = private_obses
            action_tensor = house_actions
            reward_tensor = house_rewards
            next_obs_tensor = next_private_obses
            inverse_dones = inverse_dones.unsqueeze(-1).repeat(1, self.args.n_households,1)
        else:
            obs_tensor, action_tensor, reward_tensor, next_obs_tensor = None, None, None, None

        next_value, next_pi = self.net(next_obs_tensor)
        td_target = reward_tensor + self.args.gamma * next_value * inverse_dones
        value, pi = self.net(obs_tensor)
        td_delta = td_target - value
        advantage = self.compute_advantage(self.args.gamma, self.args.tau, td_delta.cpu()).to(self.device)
        mu, std = pi
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(action_tensor)
        
        for i in range(self.args.update_each_epoch):
            value, pi = self.net(obs_tensor)
            mu, std = pi
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(action_tensor)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.args.clip, 1 + self.args.clip) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(value, td_target.detach()))
            # # policy entropy
            # entropy_loss = self.args.entropy_coef * action_dists.entropy().sum()
            total_loss = actor_loss + self.args.vloss_coef * critic_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()
            sum_loss[0] += actor_loss
            sum_loss[1] += critic_loss

        self.scheduler.step()
        return sum_loss[0], sum_loss[1]

    def get_action(self, global_obs_tensor, private_obs_tensor, gov_action=None, agent_name="household"):
        if self.agent_name == "government":
            obs_tensor = global_obs_tensor.reshape(-1, self.obs_dim)
        elif self.agent_name == "household":
            obs_tensor = private_obs_tensor
        else:
            obs_tensor = None
        _, pi = self.net(obs_tensor)
        mu, sigma = pi
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        action = (action+1)/2
        if self.agent_name == "government":
            return action.cpu().numpy().flatten()
        elif self.agent_name == "household":
            action[:, 0] = action[:, 0] * 0.3 + 0.7
            return action.cpu().numpy()
        

    def save(self, dir_path):
        torch.save(self.net.state_dict(), str(dir_path) + '/ppo_net.pt')

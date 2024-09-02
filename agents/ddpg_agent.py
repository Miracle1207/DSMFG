
import copy
import numpy as np
import torch
from agents.models import PolicyNet, QValueNet
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


class ddpg_agent:
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
        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.actor = PolicyNet(state_dim=self.obs_dim, hidden_dim=128, action_dim=self.action_dim).to(self.device)
        self.critic = QValueNet(state_dim=self.obs_dim, hidden_dim=128, action_dim=self.action_dim).to(self.device)
        
        if agent_name == "household":
            if self.args.bc == True:
                self.actor.load_state_dict(torch.load("agents/real_data/2024_01_04_21_21_maddpg_trained_model.pth", weights_only=True))
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-6)
            else:
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.p_lr)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.p_lr)
        self.target_actor = copy.copy(self.actor)
        self.target_critic = copy.copy(self.critic)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.p_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.q_lr)
        lambda_function = lambda epoch: 0.95 ** (epoch // (35*self.args.update_cycles))

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=lambda_function)
        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=lambda_function)
        self.on_policy = False
    
    def train(self, transitions, other_agent=None):
        global_obses, private_obses, gov_actions, house_actions, gov_rewards,\
        house_rewards, next_global_obses, next_private_obses, dones, past_mean_house_actions = transitions
        private_obses = torch.tensor(private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        global_obses = torch.tensor(global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_actions = torch.tensor(gov_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        house_actions = torch.tensor(house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        # mean_house_actions = torch.tensor(past_mean_house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_rewards = torch.tensor(gov_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        house_rewards = torch.tensor(house_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_global_obses = torch.tensor(next_global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_private_obses = torch.tensor(next_private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        
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

        next_q_values = self.target_critic(next_obs_tensor, self.target_actor(next_obs_tensor))
        q_targets = reward_tensor + self.args.gamma * next_q_values * inverse_dones
        critic_loss = torch.mean(F.mse_loss(self.critic(obs_tensor, action_tensor), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(obs_tensor, self.actor(obs_tensor)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update_target_network(self.target_actor, self.actor)
        self._soft_update_target_network(self.target_critic, self.critic)
        
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        return actor_loss, critic_loss

    # soft update the target network...
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.tau) * param.data + self.args.tau * target_param.data)
            
    def get_action(self, global_obs_tensor, private_obs_tensor, gov_action=None, agent_name="household"):
        if self.agent_name == "government":
            obs_tensor = global_obs_tensor.reshape(-1, self.obs_dim)
        elif self.agent_name == "household":
            obs_tensor = private_obs_tensor
        else:
            obs_tensor = None
        action = self.actor(obs_tensor).detach().cpu().numpy()

        sigma = 0.01
        action = action + sigma * np.random.randn(self.action_dim)
        if self.agent_name == "government":
            return action.flatten()
        elif self.agent_name == "household":
            return action
        
    
    def save(self, dir_path):
        torch.save(self.actor.state_dict(), str(dir_path) +'/'+ self.agent_name + '_ddpg_net.pt')
    def load(self, dir_path):
        self.actor.load_state_dict(torch.load(dir_path, map_location=torch.device(self.device), weights_only=True))


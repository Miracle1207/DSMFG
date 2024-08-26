import copy
import numpy as np
import torch
from agents.models import PolicyNet, QValueNet
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


class maddpg_agent:
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
        self.critic = QValueNet(state_dim=self.envs.households.observation_space.shape[0]*self.args.n_households + self.envs.government.observation_space.shape[0],
                                hidden_dim=128, action_dim=self.envs.households.action_space.shape[1]*self.args.n_households + self.envs.government.action_space.shape[0]).to(self.device)
        self.target_actor = copy.copy(self.actor)
        self.target_critic = copy.copy(self.critic)
        if agent_name == "household":
            if self.args.bc == True:
                self.actor.load_state_dict(torch.load("agents/real_data/2024_01_04_21_21_maddpg_trained_model.pth"))
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-6)
            else:
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.p_lr)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.p_lr)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.p_lr)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.q_lr)
        lambda_function = lambda epoch: 0.95 ** (epoch // (self.args.update_cycles / 2))
        
        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=lambda_function)
        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=lambda_function)
        self.on_policy = False
        self.current_step = 0
        
    
    def train(self, transitions, other_agent):
        global_obses, private_obses, gov_actions, house_actions, gov_rewards, \
        house_rewards, next_global_obses, next_private_obses, dones, past_mean_house_actions = transitions
        private_obses = torch.tensor(private_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        global_obses = torch.tensor(global_obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_actions = torch.tensor(gov_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        house_actions = torch.tensor(house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        # mean_house_actions = torch.tensor(past_mean_house_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        gov_rewards = torch.tensor(gov_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        house_rewards = torch.tensor(house_rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        next_global_obses = torch.tensor(next_global_obses, dtype=torch.float32,
                                         device='cuda' if self.args.cuda else 'cpu')
        next_private_obses = torch.tensor(next_private_obses, dtype=torch.float32,
                                          device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        
        if self.agent_name == "government":
            obs_tensor = global_obses
            action_tensor = gov_actions
            reward_tensor = gov_rewards
            next_obs_tensor = next_global_obses
            other_action_tensor = house_actions
            other_obs_tensor = private_obses
            next_other_obs_tensor = next_private_obses
        elif self.agent_name == "household":
            obs_tensor = private_obses
            action_tensor = house_actions
            reward_tensor = house_rewards
            next_obs_tensor = next_private_obses
            other_action_tensor = gov_actions
            other_obs_tensor = global_obses
            next_other_obs_tensor = next_global_obses
        else:
            obs_tensor, action_tensor, reward_tensor, next_obs_tensor,\
            other_action_tensor,next_other_obs_tensor,other_obs_tensor = None, None, None, None,None,None,None
            
        '''Q(vector{s}, vector{a},ag)'''
        action_tensor_ = self.actor(obs_tensor)
        # get next action
        next_action = self.target_actor(next_obs_tensor)
        next_otheragent_action = other_agent.get_action(global_obs_tensor=next_global_obses,
                                                        private_obs_tensor=next_private_obses, explore=False)
        next_otheragent_action = torch.tensor(next_otheragent_action, dtype=torch.float32,
                                              device='cuda' if self.args.cuda else 'cpu')
        if self.agent_name == "household":
            next_action = next_action.flatten(1)
            action_tensor = action_tensor.flatten(1)
            action_tensor_ = action_tensor_.flatten(1)
            next_obs_tensor = next_obs_tensor.flatten(1)
            obs_tensor = obs_tensor.flatten(1)
            reward_tensor = torch.sum(house_rewards,dim=1)
        else:
            next_otheragent_action = next_otheragent_action.flatten(1)
            other_action_tensor = other_action_tensor.flatten(1)
            next_other_obs_tensor = next_other_obs_tensor.flatten(1)
            other_obs_tensor = other_obs_tensor.flatten(1)
        
        next_action_tensor = torch.cat([next_action, next_otheragent_action], dim=-1)
        next_obs_tensor = torch.cat([next_obs_tensor, next_other_obs_tensor],dim=-1)
        next_q_values = self.target_critic(next_obs_tensor, next_action_tensor)
        q_targets = reward_tensor + self.args.gamma * next_q_values * inverse_dones

        action_tensor = torch.cat([action_tensor, other_action_tensor], dim=-1)
        obs_tensor = torch.cat([obs_tensor, other_obs_tensor], dim=-1)
        critic_loss = torch.mean(F.mse_loss(self.critic(obs_tensor, action_tensor), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_tensor_ = torch.cat([action_tensor_, other_action_tensor], dim=-1)
        actor_loss = -torch.mean(self.critic(obs_tensor, action_tensor_))
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
            target_param.data.copy_((1 - 0.95) * param.data + 0.95 * target_param.data)
    
    def get_action(self, global_obs_tensor, private_obs_tensor, gov_action=None, agent_name="household", explore=True):
        epsilon = self.get_epsilon()
        if self.agent_name == "government":
            if explore and np.random.uniform() < epsilon:
                return np.random.uniform(-1, 1, self.action_dim)
            else:
                obs_tensor = global_obs_tensor.reshape(-1, self.obs_dim)
                action = self.actor(obs_tensor).detach().cpu().numpy()
                action = action + self.args.noise_rate * np.random.randn(*action.shape)
                if explore:
                    return action.flatten()
                else:
                    return action
        elif self.agent_name == "household":
            if explore and np.random.uniform() < epsilon:
                return np.random.uniform(-1, 1, (self.args.n_households,self.action_dim))
            else:
                obs_tensor = private_obs_tensor
                action = self.actor(obs_tensor).detach().cpu().numpy()
                action = action + self.args.noise_rate * np.random.randn(*action.shape)
                # action[:,0] = 1- action[:,0]
                return action
        
    
    def save(self, dir_path):
        torch.save(self.actor.state_dict(), str(dir_path) + '/'+self.agent_name+ '_ddpg_net.pt')
        
    def load(self, dir_path):
        self.actor.load_state_dict(torch.load(dir_path))

    def get_epsilon(self, start=0.1, end=0.05, decay=1e5):
        self.current_step += 1
        return end + (start - end) * np.exp(-1. * self.current_step / decay)

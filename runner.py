import copy
import numpy as np
import torch
import os, sys
import wandb
import json
sys.path.append(os.path.abspath('../..'))
from agents.log_path import make_logpath
from utils.experience_replay import replay_buffer
from datetime import datetime

torch.autograd.set_detect_anomaly(True)

class Runner:
    def __init__(self, envs, args, house_agent, government_agent, heter_house=None):
        self.envs = envs
        self.args = args
        self.eval_env = copy.copy(envs)
        
        self.house_agent = house_agent
        self.government_agent = government_agent
        if self.args.heterogeneous_house_agent == True:
            self.heter_house = heter_house
        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)
        # state normalization
        self.global_state_norm = Normalization(shape=self.envs.government.observation_space.shape[0])
        self.private_state_norm = Normalization(shape=self.envs.households.observation_space.shape[0])
        self.model_path, _ = make_logpath(algo=self.args.house_alg +"_"+ self.args.gov_alg, n=self.args.n_households, task=self.envs.gov_task)
        save_args(path=self.model_path, args=self.args)
        self.households_welfare = 0
        self.find_best_response = self.args.find_best_response
        self.eva_year_indicator = 0
        self.wandb = self.args.wandb
        if self.wandb:
            wandb.init(
                config=self.args,
                project="MACRO",
                entity="ai_tax",
                name=self.model_path.parent.parent.parent.name + "_" + self.model_path.name + '_' + str(self.args.n_households)+"_"+self.envs.gov_task+"_seed="+str(self.args.seed),
                dir=str(self.model_path),
                job_type="training",
                reinit=True
            )
    
    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        return obs_tensor
        
    def run(self):
        agents = [self.government_agent, self.house_agent]
        gov_rew, house_rew, epochs = [], [], []
        global_obs, private_obs = self.envs.reset()
        global_obs = self.global_state_norm(global_obs)
        private_obs = self.private_state_norm(private_obs)
        train_freq = []
        for epoch in range(self.args.n_epochs):
            transition_dict = {'global_obs': [], 'private_obs': [], 'gov_action': [], 'house_action': [],'gov_reward': [],
                               'house_reward': [], 'next_global_obs': [], 'next_private_obs': [], 'done': [], "mean_house_actions": []}
            sum_loss = np.zeros((len(agents), 2))
            for t in range(self.args.epoch_length):
                global_obs_tensor = self._get_tensor_inputs(global_obs)
                private_obs_tensor = self._get_tensor_inputs(private_obs)
                gov_action = self.government_agent.get_action(global_obs_tensor=global_obs_tensor,
                                                              private_obs_tensor=private_obs_tensor,
                                                              agent_name="government")
                house_action = self.house_agent.get_action(global_obs_tensor=global_obs_tensor,
                                                           private_obs_tensor=private_obs_tensor,
                                                           gov_action=gov_action, agent_name="household")
                if "mf" in self.args.house_alg:
                    house_action, mean_house_action = house_action
                else:
                    mean_house_action = None
                action = {self.envs.government.name: gov_action,
                          self.envs.households.name: house_action}
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)
                next_global_obs = self.global_state_norm(next_global_obs)
                next_private_obs = self.private_state_norm(next_private_obs)
                
                if agents[0].on_policy or agents[1].on_policy:
                    # on policy
                    transition_dict['global_obs'].append(global_obs)
                    transition_dict['private_obs'].append(private_obs)
                    transition_dict['gov_action'].append(gov_action)
                    transition_dict['house_action'].append(house_action)
                    transition_dict['gov_reward'].append(gov_reward)
                    transition_dict['house_reward'].append(house_reward)
                    transition_dict['next_global_obs'].append(next_global_obs)
                    transition_dict['next_private_obs'].append(next_private_obs)
                    transition_dict['done'].append(float(done))
                    transition_dict['mean_house_actions'].append(mean_house_action)
                if (not agents[0].on_policy) or (not agents[1].on_policy):
                    # off policy: replay buffer
                    self.buffer.add(global_obs, private_obs, gov_action, house_action, gov_reward, house_reward,
                                    next_global_obs, next_private_obs, float(done), mean_action=mean_house_action)
            
                global_obs = next_global_obs
                private_obs = next_private_obs
                if done:
                    global_obs, private_obs = self.envs.reset()
                    global_obs = self.global_state_norm(global_obs)
                    private_obs = self.private_state_norm(private_obs)

            # for leader, if follower is not BR, break
            for i in range(len(agents)):
                # if epoch < 10 or epoch % 5 == 0 or i == 1:
                if agents[i].on_policy == True:
                    actor_loss, critic_loss = agents[i].train(transition_dict)
                    sum_loss[i, 0] = actor_loss
                    sum_loss[i, 1] = critic_loss
                else:
                    for _ in range(self.args.update_cycles):
                        transitions = self.buffer.sample(self.args.batch_size)
                        actor_loss, critic_loss = agents[i].train(transitions, other_agent=agents[1-i])  # MARL has other agents
                        sum_loss[i, 0] += actor_loss
                        sum_loss[i, 1] += critic_loss
                    
            # print the log information
            if epoch % self.args.display_interval == 0:
                if epoch == 20:
                    write_flag = False
                else:
                    write_flag = False
                economic_idicators_dict = self._evaluate_agent(write_evaluate_data=write_flag)
                now_step = (epoch + 1) * self.args.epoch_length
                gov_rew.append(economic_idicators_dict["gov_reward"])
                house_rew.append(economic_idicators_dict["house_reward"])
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)
                loss_dict = {
                    "house_actor_loss": sum_loss[1, 0],
                    "house_critic_loss": sum_loss[1, 1],
                    "gov_actor_loss": sum_loss[0, 0],
                    "gov_critic_loss": sum_loss[0, 1]
                }
                if self.find_best_response == True:
                    exploitability_rate = self.judge_best_response()
                else:
                    exploitability_rate = 100
                exploitability_dict = {
                    "exploitability": exploitability_rate
                }
                if self.wandb:
                    wandb.log(economic_idicators_dict)
                    wandb.log(loss_dict)
                    wandb.log(exploitability_dict)
                
                print(
                    '[{}] Epoch: {} / {}, Frames: {}, Gov_Rewards: {:.3f}, House_Rewards: {:.3f}, years:{:.3f}, actor_loss: {:.3f}, critic_loss: {:.3f}, exploitability_rate: {:.6f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length,
                        economic_idicators_dict["gov_reward"], economic_idicators_dict["house_reward"],
                        economic_idicators_dict["years"], np.sum(sum_loss[:,0]), np.sum(sum_loss[:,1]), exploitability_rate))

            if epoch % self.args.save_interval == 0:
                self.house_agent.save(dir_path=self.model_path)
                self.government_agent.save(dir_path=self.model_path)
    
        if self.wandb:
            wandb.finish()

    def test(self,house_model_path, government_model_path):
        self.house_agent.load(dir_path=house_model_path)
        self.government_agent.load(dir_path=government_model_path)
        economic_idicators_dict = self._evaluate_agent(write_evaluate_data=True)

    def init_economic_dict(self, gov_reward, households_reward):
    
        self.econ_dict = {
            "gov_reward": gov_reward,  # sum
            "social_welfare": np.sum(households_reward),  # sum
            "house_reward": households_reward,  # sum
            "years": self.eval_env.step_cnt,  # max
            "house_income": self.eval_env.post_income,  # mean
            "house_total_tax": self.eval_env.tax_array,
            "house_income_tax": self.eval_env.income_tax,
            "house_wealth": self.eval_env.households.at_next,
            "house_wealth_tax": self.eval_env.asset_tax,
            "per_gdp": self.eval_env.per_household_gdp,
            "GDP": self.eval_env.GDP,  # sum
            "income_gini": self.eval_env.income_gini,
            "wealth_gini": self.eval_env.wealth_gini,
            "WageRate": self.eval_env.WageRate,
            "total_labor": self.eval_env.Lt,
            "house_consumption": self.eval_env.consumption,
            "house_work_hours": self.eval_env.ht,
            "gov_spending": self.eval_env.Gt_prob * self.eval_env.GDP}
        
    def judge_best_response(self, transition_dict=None):
        # fix the leader's policy, update follower's policy until follower's value don't change.
        current_indicators = self._evaluate_agent()
        current_households_welfare = current_indicators["social_welfare"]
        current_government_payoff = current_indicators['gov_reward']
        house_agent_update = copy.copy(self.house_agent)
        government_agent_update = copy.copy(self.government_agent)
        
        if self.house_agent.on_policy == True:
            actor_loss, critic_loss = house_agent_update.train(transition_dict)
            actor_loss, critic_loss = government_agent_update.train(transition_dict)
        else:
            for _ in range(self.args.update_cycles):
                transitions = self.buffer.sample(self.args.batch_size)
                house_agent_update.train(transitions, other_agent=self.government_agent)  # MARL has other agents
                government_agent_update.train(transitions, other_agent=self.house_agent)  # MARL has other agents
        new_households_welfare = self._evaluate_agent(single_update_household=house_agent_update, judge_exploitability=True)["social_welfare"]
        new_government_payoff = self._evaluate_agent(single_update_government=government_agent_update, judge_exploitability=True)["gov_reward"]
        
        exploitability_rate = abs((new_households_welfare - current_households_welfare)/current_households_welfare) + abs((new_government_payoff - current_government_payoff)/current_government_payoff)
        return exploitability_rate


    def _evaluate_agent(self, single_update_household=None, single_update_government=None, judge_exploitability=False, write_evaluate_data=False):
        # eval_econ = ["gov_reward", "house_reward", "social_welfare", "per_gdp","income_gini","wealth_gini","years","GDP"]
        eval_econ = ["gov_reward", "house_reward", "social_welfare", "per_gdp", "income_gini",
                     "wealth_gini", "years", "GDP", "gov_spending", "house_total_tax", "house_income_tax",
                     "house_wealth_tax", "house_wealth", "house_income", "house_consumption", "house_work_hours",
                     "total_labor", "WageRate"]

        episode_econ_dict = dict(zip(eval_econ, [[] for i in range(len(eval_econ))]))
        final_econ_dict = dict(zip(eval_econ, [None for i in range(len(eval_econ))]))
        for epoch_i in range(self.args.eval_episodes):
            global_obs, private_obs = self.eval_env.reset()
            global_obs = self.global_state_norm(global_obs)
            private_obs = self.private_state_norm(private_obs)
            eval_econ_dict = dict(zip(eval_econ, [[] for i in range(len(eval_econ))]))
            
            while True:
                with torch.no_grad():
                    global_obs_tensor = self._get_tensor_inputs(global_obs)
                    private_obs_tensor = self._get_tensor_inputs(private_obs)
                    if judge_exploitability == True and single_update_government != None:
                        gov_action = single_update_government.get_action(global_obs_tensor=global_obs_tensor,
                                                                      private_obs_tensor=private_obs_tensor,
                                                                      agent_name="government")
                    else:
                        gov_action = self.government_agent.get_action(global_obs_tensor=global_obs_tensor,
                                                                  private_obs_tensor=private_obs_tensor,
                                                                  agent_name="government")
                    house_action = self.house_agent.get_action(global_obs_tensor=global_obs_tensor,
                                                               private_obs_tensor=private_obs_tensor,
                                                               gov_action=gov_action, agent_name="household")
                    
                    if judge_exploitability == True and single_update_household != None:
                        
                        update_house_action = single_update_household.get_action(global_obs_tensor=global_obs_tensor,
                                                                                 private_obs_tensor=private_obs_tensor,
                                                                                 gov_action=gov_action,
                                                                                 agent_name="household")
                    
                    if "mf" in self.args.house_alg:
                        house_action, mean_house_action = house_action
                    else:
                        mean_house_action = None
                    if judge_exploitability == True and single_update_household != None:
                        if "mf" in self.args.house_alg:
                            update_house_action, _ = update_house_action
                        random_house_index = np.random.randint(0, self.args.n_households)
                        house_action[random_house_index] = update_house_action[random_house_index]
                        if "mf" in self.args.house_alg:
                            mean_house_action = np.mean(house_action, axis=-2)
                    action = {self.envs.government.name: gov_action,
                              self.envs.households.name: house_action}
                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    next_global_obs = self.global_state_norm(next_global_obs)
                    next_private_obs = self.private_state_norm(next_private_obs)

                    if (not self.government_agent.on_policy) or (not self.house_agent.on_policy):
                        # off policy: replay buffer
                        self.buffer.add(global_obs, private_obs, gov_action, house_action, gov_reward, house_reward,
                                        next_global_obs, next_private_obs, float(done), mean_action=mean_house_action)

                self.init_economic_dict(gov_reward, house_reward)
                for each in eval_econ:
                    if "house_" in each:
                        eval_econ_dict[each].append(self.econ_dict[each].tolist())
                    else:
                        eval_econ_dict[each].append(self.econ_dict[each])
                if done:
                    print(self.eval_env.step_cnt)
                    break
                
                global_obs = next_global_obs
                private_obs = next_private_obs
            
            for key, value in eval_econ_dict.items():
                if key == "gov_reward" or key == "house_reward" or key == "GDP":
                    episode_econ_dict[key].append(np.sum(value))
                elif key == "years":
                    episode_econ_dict[key].append(np.max(value))
                else:
                    episode_econ_dict[key].append(np.mean(value))
        
        for key, value in episode_econ_dict.items():
            final_econ_dict[key] = np.mean(value)
        # 将字典直接写入文件
        if self.econ_dict['years'] >= self.eva_year_indicator:
            write_evaluate_data = write_evaluate_data and True
            self.eva_year_indicator = self.econ_dict['years']
       
        if write_evaluate_data == True:
            print("============= Finish ================")
            print("============= Finish ================")
            store_path = "agents/data/"+self.args.economic_shock+"/N="+str(self.args.n_households)+"/"
            if not os.path.exists(os.path.dirname(store_path)):
                os.makedirs(os.path.dirname(store_path))
            with open(store_path + self.args.house_alg + "_" + self.args.gov_alg +"_"+str(self.args.n_households)+ "_data.json", "w") as file:
                json.dump(eval_econ_dict, file)
        return final_econ_dict


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)
    
    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x
    
    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
    
    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)
    
    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x


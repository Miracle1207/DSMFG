import numpy as np
from env.env_core import economic_society
from agents.rule_based import rule_agent
from agents.ddpg_agent import ddpg_agent
from agents.maddpg_agent import maddpg_agent
from agents.real_data.real_data import real_agent
from agents.mfrl import mfrl_agent
from agents.bi_mfrl import bi_mfrl_agent
from agents.bi_ddpg_agent import bi_ddpg_agent
from agents.aie_agent import aie_agent
from utils.seeds import set_seeds
import os
import argparse
from omegaconf import OmegaConf
from runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='default')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--br", action='store_true')
    parser.add_argument("--house_alg", type=str, default='mfac', help="rule_based, ippo, mfac, real")
    parser.add_argument("--gov_alg", type=str, default='ac', help="ac, rule_based, independent")
    parser.add_argument("--task", type=str, default='gdp', help="gini, social_welfare, gdp_gini")
    parser.add_argument('--device-num', type=int, default=1, help='the number of cuda service num')
    parser.add_argument('--n_households', type=int, default=100, help='the number of total households')
    parser.add_argument('--seed', type=int, default=1, help='the random seed')
    parser.add_argument('--hidden_size', type=int, default=128, help='[64, 128, 256]')
    parser.add_argument('--q_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--p_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--batch_size', type=int, default=64, help='[32, 64, 128, 256]')
    parser.add_argument('--update_cycles', type=int, default=100, help='[10，100，1000]')
    parser.add_argument('--update_freq', type=int, default=10, help='[10，20，30]')
    parser.add_argument('--initial_train', type=int, default=10, help='[10，100，200]')
    args = parser.parse_args()
    return args


def select_agent(alg, agent_name):
    if agent_name == "households":
        if alg == "real":
            house_agent = real_agent(env, yaml_cfg.Trainer)
        elif alg == "mfrl":
            house_agent = mfrl_agent(env, yaml_cfg.Trainer)
        elif alg == "bi_mfrl":
            house_agent = bi_mfrl_agent(env, yaml_cfg.Trainer)
        elif alg == "ppo":
            house_agent = ppo_agent(env, yaml_cfg.Trainer, agent_name="household")
        elif alg == "ddpg":
            house_agent = ddpg_agent(env, yaml_cfg.Trainer, agent_name="household")
        elif alg == "maddpg":
            house_agent = maddpg_agent(env, yaml_cfg.Trainer, agent_name="household")
        elif alg == "rule_based":
            house_agent = rule_agent(env, yaml_cfg.Trainer)
        elif alg == "aie":
            house_agent = aie_agent(env, yaml_cfg.Trainer)
        else:
            print("Wrong Choice!")
        return house_agent
    else:
        if alg == "rule_based" or alg == "us_federal" or alg == "saez":
            gov_agent = rule_agent(env, yaml_cfg.Trainer)
        # elif alg == "ppo":
        #     gov_agent = ppo_agent(env, yaml_cfg.Trainer, agent_name="government")
        elif alg == "ddpg":
            gov_agent = ddpg_agent(env, yaml_cfg.Trainer, agent_name="government")
        elif alg == "maddpg":
            gov_agent = maddpg_agent(env, yaml_cfg.Trainer, agent_name="government")
        elif alg == "bi_ddpg":
            gov_agent = bi_ddpg_agent(env, yaml_cfg.Trainer, agent_name="government")
        elif alg == "aie":
            gov_agent = aie_agent(env, yaml_cfg.Trainer, agent_name="government")
        else:
            print("Wrong Choice!")
        return gov_agent

def choose_model_path(house_alg, gov_alg, households_num):
    if house_alg == "bi_mfrl" and gov_alg == "bi_ddpg":
        # bi_mfrl_bi_ddpg
        if households_num == 100:
            house_model_path="agents/models/bi_mfrl_bi_ddpg/100/gdp/run39/bimf_house_actor.pt"
            government_model_path="agents/models/bi_mfrl_bi_ddpg/100/gdp/run39/bi_ddpg_net.pt"
        elif households_num == 1000:
            house_model_path="agents/models/bi_mfrl_bi_ddpg/1000/gdp/run15/bimf_house_actor.pt" # seed 10
            government_model_path="agents/models/bi_mfrl_bi_ddpg/1000/gdp/run15/bi_ddpg_net.pt"
    if house_alg == "aie" and gov_alg == "aie":
        if households_num == 100:
            house_model_path="agents/models/aie_aie/100/gdp/run64/household_aie_net.pt"
            government_model_path="agents/models/aie_aie/100/gdp/run64/government_aie_net.pt"
        elif households_num == 1000:
            house_model_path="agents/models/bi_mfrl_bi_ddpg/1000/gdp/run15/bimf_house_actor.pt"
            government_model_path="agents/models/bi_mfrl_bi_ddpg/1000/gdp/run15/bi_ddpg_net.pt"
    if house_alg == "maddpg" and gov_alg == "maddpg":
        if households_num == 100:
            # maddpg
            house_model_path="agents/models/maddpg_maddpg/100/gdp/run28/household_ddpg_net.pt"
            government_model_path="agents/models/maddpg_maddpg/100/gdp/run28/government_ddpg_net.pt"
        elif households_num == 1000:
            # maddpg
            house_model_path="agents/models/maddpg_maddpg/1000/gdp/run7/household_ddpg_net.pt"  # run10 - seed 11
            government_model_path="agents/models/maddpg_maddpg/1000/gdp/run7/government_ddpg_net.pt"  # run7 -seed 2
    # mfrl+ddpg
    if house_alg == "mfrl" and gov_alg == "ddpg":
        if households_num == 100:
            house_model_path="agents/models/mfrl_ddpg/100/gdp/run15/house_actor.pt"
            government_model_path="agents/models/mfrl_ddpg/100/gdp/run15/ddpg_net.pt"
        elif households_num == 1000:
            house_model_path="agents/models/mfrl_ddpg/1000/gdp/run4/house_actor.pt"  # run2 -seed2
            government_model_path="agents/models/mfrl_ddpg/1000/gdp/run4/government_ddpg_net.pt"  # run4 - seed1
    # iddpg
    if house_alg == "ddpg" and gov_alg == "ddpg":
        if households_num == 100:
            house_model_path="agents/models/ddpg_ddpg/100/gdp/run15/household_ddpg_net.pt"
            government_model_path="agents/models/ddpg_ddpg/100/gdp/run15/government_ddpg_net.pt"
        elif households_num == 1000:
            house_model_path="agents/models/ddpg_ddpg/1000/gdp/run7/household_ddpg_net.pt"
            government_model_path="agents/models/ddpg_ddpg/1000/gdp/run7/government_ddpg_net.pt"
    return house_model_path, government_model_path

if __name__ == '__main__':
    # set signle thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    args = parse_args()
    path = args.config
    yaml_cfg = OmegaConf.load(f'./cfg/{path}.yaml')
    yaml_cfg.Trainer["n_households"] = args.n_households
    yaml_cfg.Environment.Entities[1]["entity_args"].n = args.n_households
    yaml_cfg.Environment.env_core["env_args"].gov_task = args.task
    yaml_cfg.Trainer["seed"] = args.seed
    yaml_cfg.Trainer["wandb"] = args.wandb
    yaml_cfg.Trainer["find_best_response"] = args.br
    
    '''tuning'''
    # tuning(yaml_cfg)
    yaml_cfg.Trainer["hidden_size"] = args.hidden_size
    yaml_cfg.Trainer["q_lr"] = args.q_lr
    yaml_cfg.Trainer["p_lr"] = args.p_lr
    yaml_cfg.Trainer["batch_size"] = args.batch_size
    yaml_cfg.Trainer["house_alg"] = args.house_alg
    yaml_cfg.Trainer["gov_alg"] = args.gov_alg
    
    if args.gov_alg == "saez" or args.gov_alg == "us_federal":
        yaml_cfg.Environment['env_core']['env_args']['tax_moudle'] = args.gov_alg
    set_seeds(args.seed, cuda=yaml_cfg.Trainer["cuda"])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    env = economic_society(yaml_cfg.Environment)
    
    house_agent = select_agent(args.house_alg, agent_name="households")
    gov_agent = select_agent(args.gov_alg, agent_name="government")
    
    print("n_households: ", yaml_cfg.Trainer["n_households"])
    
    TEST = args.test
    if TEST == True:
        heter_house_agent = select_agent(yaml_cfg.Trainer["heter_house_alg"], agent_name="households")
        runner = Runner(env, yaml_cfg.Trainer, house_agent=house_agent, government_agent=gov_agent,
                        heter_house=heter_house_agent)
        house_model_path, government_model_path = choose_model_path(house_alg=args.house_alg, gov_alg=args.gov_alg, households_num=args.n_households)
        runner.test(house_model_path, government_model_path)
    else:
        runner = Runner(env, yaml_cfg.Trainer, house_agent=house_agent, government_agent=gov_agent)
        runner.run()





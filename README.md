# TaxAI: A Dynamic Economic Simulator and Benchmark for Multi-Agent Reinforcement Learning

<div style="text-align:center">
  <img src="./img/new_model_dynamics.png" alt="示例图片" >
  <figcaption style="text-align:center;"></figcaption>
</div>



The optimization of fiscal policies by governments to stimulate economic growth, ensure social equity and stability, and maximize social welfare has been a subject of significant interest. Simultaneously, individuals keenly observe government policies to optimize their own production and saving strategies. 

To simulate this problem, we propose a multi-agent reinforcement learning simulator based on the Bewley-Aiyagari model. Our simulator encompasses various economic activities of governments, households, technology, and financial intermediaries. By integrating reinforcement learning algorithms, it enables the derivation of optimal strategies for governments and individuals while facilitating the study of the relationship between government policies, micro-level household behaviors, and macroeconomic phenomena.

### A comparison of MARL simulators for optimal taxation problems

| Simulator             | AI Economist | RBC Model | **TaxAI** (ours)              |
| --------------------- | ------------ | --------- | ----------------------------- |
| Households' Number    | 10           | 100       | 10000                         |
| Tax Schedule          | Non-linear   | Linear    | Non-linear                    |
| Tax Type              | Income       | Income    | Income & Wealth & Consumption |
| Social Roles' Types   | 2            | 3         | 4                             |
| Saving Strategy       | &#x2716;     | &#x2714;  | &#x2714;                      |
| Heterogenous Agent    | &#x2714;     | &#x2714;  | &#x2714;                      |
| Real-data Calibration | &#x2716;     | &#x2716;  | &#x2714;                      |
| Open source           | &#x2714;     | &#x2716;  | &#x2714;                      |
| MARL Benchmark        | &#x2716;     | &#x2716;  | &#x2714;                      |



## Install

You can use any tool to manage your python environment. Here, we use conda as an example.

1. Install conda/miniconda.

2. Build a Python virtual environment.

```bash
conda create -n smfg python=3.6
```

3. Activate the virtual environment

```bash
conda activate smfg
```

4. Clone the repository and install the required dependencies

```bash 
cd SMFG
pip install -r requirements.txt
```

## ALgorithms for solving SMFGs

The details of the algorithms used by the leader and follower agents in the baselines for solving SMFG.

| Baselines  | Follower's Algorithm    | Leader's Algorithm     |
| ---------- | ----------------------- | ---------------------- |
| Rule-based | Random/Behavior cloning | Rule-based/Free market |
| DDPG       | Random/Behavior cloning | DDPG                   |
| MADDPG     | MADDPG                  | MADDPG                 |
| IDDPG      | IDDPG                   | IDDPG                  |
| MF-MARL    | Mean Field MARL         | DDPG                   |
| SMFRL      | SMFRL                   | SMFRL                  |

### Train agents

1. Rule-based

when households' policy is behavior cloning on real data:

```bash
python main.py --n_households 100 --house_alg "real" --gov_alg "rule_based" --task "gdp" --seed 8 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```

when households' policy is random policy:

```bash
python main.py --n_households 100 --house_alg "rule_based" --gov_alg "rule_based" --task "gdp" --seed 8 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```

2. DDPG

when households' policy is behavior cloning on real data:

```bash
python main.py --n_households 100 --house_alg "real" --gov_alg "ddpg" --task "gdp" --seed 8 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```

when households' policy is random policy:

```bash
python main.py --n_households 100 --house_alg "rule_based" --gov_alg "ddpg" --task "gdp" --seed 8 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```

3. MADDPG

```bash
python main.py --n_households 100 --house_alg "maddpg" --gov_alg "maddpg" --task "gdp" --seed 112 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```

4. IDDPG

```bash
python main.py --n_households 100 --house_alg "ddpg" --gov_alg "ddpg" --task "gdp" --seed 8 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```

5. MF-MARL

```bash
python main.py --n_households 100 --house_alg "mfrl" --gov_alg "ddpg" --task "gdp" --seed 8 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```

6. SMFRL (our method)

```bash
python main.py --n_households 100 --house_alg "bi_mfrl" --gov_alg "bi_ddpg" --task "gdp" --seed 8 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```



## Economic policies

(1) Free Market: A market without policy intervention.

```bash
python main.py --n_households 100 --house_alg "real" --gov_alg "rule_based" --task "gdp" --seed 112 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```

(2) Saez Tax: The Saex tax policy is often considered a suggestion for specific tax reforms in the real world.

```bash
python main.py --n_households 100 --house_alg "real" --gov_alg "saez" --task "gdp" --seed 112 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```

(3) U.S. Federal Tax: Real data from OECD in 2022 for this policy.

```bash
python main.py --n_households 100 --house_alg "real" --gov_alg "us_federal" --task "gdp" --seed 112 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128
```

(4) AI Economist: This is a two-level MARL method based on Proximal Policy Optimization (PPO). In the first phase, households' policies are trained from scratch in a free-market (no-tax) environment. In the second phase, households continue to learn under an RL social planner.

```bash
python main.py --n_households 100 --house_alg "aie" --gov_alg "aie" --task "gdp" --seed 112 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```

(5) AI Economist-BC: For fairness in comparison, we evaluated the AI Economist method with behavior cloning as pre-training to determine its effectiveness.

```bash
python main.py --n_households 100 --house_alg "aie" --gov_alg "aie" --task "gdp" --seed 112 --hidden_size 128 --q_lr 3e-4 --p_lr 3e-4 --batch_size 128 
```




## Acknowledgement

[TaxAI](https://github.com/jidiai/TaxAI)


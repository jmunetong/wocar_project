#!/bin/bash
python run.py --config-path /Users/jmuneton/Documents/stanford/s_2024/cs234/project/project/config_experiments/config_hopper_vanilla_ppo.json
python runt.py --config-path /Users/jmuneton/Documents/stanford/s_2024/cs234/project/project/config_experiments/config_hopper_robust_q_ppo_sgld.json
python run.py --config-path /Users/jmuneton/Documents/stanford/s_2024/cs234/project/project/config_experiments/config_hopper_robust_ppo_sgld.json


# python run_draft.py --config-path /Users/jmuneton/Documents/stanford/s_2024/cs234/project/project/config_experiments/config_halfcheetah_vanilla_ppo.json
# python run_draft.py --config-path /Users/jmuneton/Documents/stanford/s_2024/cs234/project/project/config_experiments/config_halfcheetah_robust_ppo_sgld.json
# python run_draft.py --config-path  /Users/jmuneton/Documents/stanford/s_2024/cs234/project/project/config_experiments/config_halfcheetah_robust_q_ppo_sgld.json

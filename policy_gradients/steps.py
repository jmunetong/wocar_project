
from .ppo import *
from .robust_ppo import *
from .robust_q_ppo import *

'''
File for taking steps in both policy and value network space.
Layout of this file:
    - Surrogate reward function
    - Logging functions for TRPO approximations
        - kl_approximation_logging
        - kl_vs_second_order_approx
    - Possible value loss functions
        - consistency loss [+ clipped version for matching OpenAI]
        - time-dependent baseline
    - Actual optimization functions
        - value_step
        - ppo_step
        - trpo_step
'''


def step_with_mode(mode, adversary=False):
    STEPS = {
        'ppo': ppo_step,
        'robust_ppo': robust_ppo_step,
        'adv_ppo': ppo_step,
        'adv_sa_ppo': robust_ppo_step,
        'robust_q_ppo': robust_q_ppo_step,
    }
    ADV_STEPS = {
        'trpo': None,
        'ppo': None,
        'robust_ppo': None,
        'adv_ppo': ppo_step,
        'adv_sa_ppo': ppo_step,
        'robust_q_ppo': None,
    }
    if adversary:
        return ADV_STEPS[mode]
    else:
        return STEPS[mode]


last_norm = None



from tqdm import tqdm 
import numpy as np 

import torch
import torch.nn.functional as F

from .step_utils import shape_equal_cmp, soft_update
from .pgd_act import worst_action_pgd

def worst_q_step(all_states, actions, next_states, not_dones, rewards, q_net, target_q_net, policy_net, q_opt,     
                params, eps_scheduler, should_tqdm=False, should_cuda=False):
    '''
    Take an optimizer step training the worst-q function
    parameterized by a neural network
    Inputs:
    - all_states, the states at each timestep
    - actions, the actions taking at each timestep
    - next_states, the next states after taking actions
    - not dones, N * T array with 0s at final steps and 1s everywhere else
    - rewards, the rewards gained at each timestep
    - q_net, worst-case q neural network
    - q_opt, the optimizer for q_net
    - target_q_net, the target q_net
    - params, dictionary of parameters
    Returns:
    - Loss of the q_net regression problem
    '''
    current_eps = eps_scheduler.get_eps()

    r = range(params.Q_EPOCHS) if not should_tqdm else \
                            tqdm(range(params.Q_EPOCHS))
    for i in r:
        # Create minibatches with shuffuling
        state_indices = np.arange(rewards.nelement())
        np.random.shuffle(state_indices)
        splits = np.array_split(state_indices, params.NUM_MINIBATCHES)

        assert shape_equal_cmp(rewards, not_dones)

        # Minibatch SGD
        for selected in splits:
            q_opt.zero_grad()

            def sel(*args):
                return [v[selected] for v in args]

            def to_cuda(*args):
                return [v.cuda() for v in args]

            # Get a minibatch (64).
            tup = sel(actions, rewards, not_dones, next_states, all_states)
            mask = torch.tensor(True)

            if should_cuda: tup = to_cuda(*tup)
            sel_acts, sel_rews, sel_not_dones, sel_next_states, sel_states = tup

            # Worst q prediction of current network given the states.
            curr_q = q_net(torch.cat((sel_states, sel_acts), dim=1)).squeeze(-1)
            worst_actions = worst_action_pgd(q_net, policy_net, sel_next_states, eps=0.01, maxiter=50)
            expected_q = sel_rews + params.GAMMA * sel_not_dones * target_q_net(torch.cat((sel_next_states, worst_actions), dim=1)).squeeze(-1)
            '''
            print('curr_q', curr_q.mean())
            print('expected_q', expected_q.mean())
            '''
            q_loss = F.mse_loss(curr_q, expected_q)
            q_loss.backward()
            q_opt.step()
            soft_update(target_q_net, q_net, params.TAU)

        print(f'q_loss={q_loss.item():8.5f}')

    return q_loss
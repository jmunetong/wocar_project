
import torch 

import numpy as np
from tqdm import tqdm
from .step_utils import pack_history, shape_equal_cmp


def value_step(all_states, returns, advantages, not_dones, net,
               val_opt, params, store, old_vs=None, opt_step=None,
               should_tqdm=False, should_cuda=False, test_saps=None):
    '''
    Take an optimizer step fitting the value function
    parameterized by a neural network
    Inputs:
    - all_states, the states at each timestep
    - rewards, the rewards gained at each timestep
    - returns, discounted rewards (ret_t = r_t + gamma*ret_{t+1})
    - advantaages, estimated by GAE
    - not_dones, N * T array with 0s at final steps and 1s everywhere else
    - net, the neural network representing the value function 
    - val_opt, the optimizer for net
    - params, dictionary of parameters
    Returns:
    - Loss of the value regression problem
    '''

    # (sharing weights) XOR (old_vs is None)
    # assert params.SHARE_WEIGHTS ^ (old_vs is None)

    # Options for value function
    VALUE_FUNCS = {
        "gae": value_loss_gae,
        "time": value_loss_returns
    }
         
    # If we are not sharing weights, then we need to keep track of what the 
    # last value was here. If we are sharing weights, this is handled in policy_step
    with torch.no_grad():
        if old_vs is None:
            state_indices = np.arange(returns.nelement())
            # No shuffling, just split an sequential list of indices.
            splits = np.array_split(state_indices, params.NUM_MINIBATCHES)
            orig_vs = []
            # Minibattorch.
            for selected in splits:
                # Values of current network prediction.
                orig_vs.append(net(all_states[selected]).squeeze(-1))
            orig_vs = torch.cat(orig_vs)
            old_vs = orig_vs.detach()
        if test_saps is not None:
            old_test_vs = net(test_saps.states).squeeze(-1)


    """
    print('all_states', all_states.size())
    print('returns', returns.size())
    print('advantages', advantages.size())
    print('not_dones', not_dones.size())
    print('old_vs', old_vs.size())
    """


    r = range(params.VAL_EPOCHS) if not should_tqdm else \
                            tqdm(range(params.VAL_EPOCHS))

    if params.HISTORY_LENGTH > 0 and params.USE_LSTM_VAL:
        # LSTM policy. Need to go over all episodes instead of states.
        batches, alive_masks, time_masks, lengths = pack_history([all_states, returns, not_dones, advantages, old_vs], not_dones, max_length=params.HISTORY_LENGTH)
        assert not params.SHARE_WEIGHTS

    for i in r:
        if params.HISTORY_LENGTH > 0 and params.USE_LSTM_VAL:
            # LSTM policy. Need to go over all episodes instead of states.
            hidden = None
            val_opt.zero_grad()
            val_loss = 0.0
            for i, batch in enumerate(batches):
                # Now we get chunks of time sequences, each of them with a maximum length of params.HISTORY_LENGTH.
                # select log probabilities, advantages of this minibattorch.
                batch_states, batch_returns, batch_not_dones, batch_advs, batch_old_vs = batch
                mask = time_masks[i]
                # keep only the alive hidden states.
                if hidden is not None:
                    # print('hidden[0]', hidden[0].size())
                    hidden = [h[:, alive_masks[i], :].detach() for h in hidden]
                    # print('hidden[0]', hidden[0].size())
                vs, hidden = net.multi_forward(batch_states, hidden=hidden)
                vs = vs.squeeze(-1)

                vf = VALUE_FUNCS[params.VALUE_CALC]
                batch_val_loss = vf(vs, batch_returns, batch_advs, batch_not_dones, params,
                              batch_old_vs, mask=mask, store=store, reduction='sum')
                val_loss += batch_val_loss

            val_loss = val_loss / all_states.size(0)
            val_loss.backward()
            val_opt.step()
        else:
            # Create minibatches with shuffuling
            state_indices = np.arange(returns.nelement())
            np.random.shuffle(state_indices)
            splits = np.array_split(state_indices, params.NUM_MINIBATCHES)

            assert shape_equal_cmp(returns, advantages, not_dones, old_vs)

            # Minibatch SGD
            for selected in splits:
                val_opt.zero_grad()

                def sel(*args):
                    return [v[selected] for v in args]

                def to_cuda(*args):
                    return [v.cuda() for v in args]

                # Get a minibatch (64) of returns, advantages, etc.
                tup = sel(returns, advantages, not_dones, old_vs, all_states)
                mask = torch.tensor(True)

                if should_cuda: tup = to_cuda(*tup)
                sel_rets, sel_advs, sel_not_dones, sel_ovs, sel_states = tup

                # Value prediction of current network given the states.
                vs = net(sel_states).squeeze(-1)

                vf = VALUE_FUNCS[params.VALUE_CALC]
                val_loss = vf(vs, sel_rets, sel_advs, sel_not_dones, params,
                              sel_ovs, mask=mask, store=store)
                # If we are sharing weights, then value_step gets called 
                # once per policy optimizer step anyways, so we only do one batch
                if params.SHARE_WEIGHTS:
                    return val_loss

                # From now on, params.SHARE_WEIGHTS must be False
                
                val_loss.backward()
                val_opt.step()
        if should_tqdm:
            if test_saps is not None: 
                vs = net(test_saps.states).squeeze(-1)
                test_loss = vf(vs, test_saps.returns, test_saps.advantages,
                    test_saps.not_dones, params, old_test_vs, None)
            r.set_description(f'vf_train: {val_loss.mean().item():.2f}'
                              f'vf_test: {test_loss.mean().item():.2f}')
        print(f'val_loss={val_loss.item():8.5f}')

    return val_loss


def value_loss_gae(vs, _, advantages, not_dones, params, old_vs, mask=None, store=None, re=False, reduction='mean'):
    '''
    GAE-based loss for the value function:
        L_t = ((v_t + A_t).detach() - v_{t})
    Optionally, we clip the value function around the original value of v_t

    Inputs: rewards, returns, not_dones, params (from value_step)
    Outputs: value function loss
    '''
    # Desired values are old values plus advantage of the action taken. They do not change during the optimization process.
    # We want the current values are close to them.
    val_targ = (old_vs + advantages).detach()
    assert shape_equal_cmp(val_targ, vs, not_dones, old_vs, advantages)
    assert len(vs.shape) == 1 or len(vs.shape) == 2

    try:
        vs_clipped = old_vs + torch.clamp(vs - old_vs, -params.CLIP_VAL_EPS, params.CLIP_VAL_EPS)
    except AttributeError as e:
        vs_clipped = old_vs + torch.clamp(vs - old_vs, -params.CLIP_EPS, params.CLIP_EPS)
        
    # Don't incur loss from last timesteps (since there's no return to use)
    sel = torch.logical_and(not_dones.bool(), mask)
    # print('selected', sel.sum().item())
    assert shape_equal_cmp(vs, sel)
    val_loss_mat_unclipped = (vs - val_targ)[sel].pow(2)
    val_loss_mat_clipped = (vs_clipped - val_targ)[sel].pow(2)

    # In OpenAI's PPO implementation, we clip the value function around the previous value estimate
    # and use the worse of the clipped and unclipped versions to train the value function

    # Presumably the inspiration for this is similar to PPO
    if params.VALUE_CLIPPING:
        val_loss_mat = torch.max(val_loss_mat_unclipped, val_loss_mat_clipped)
    else:
        val_loss_mat = val_loss_mat_unclipped

    # assert shape_equal_cmp(val_loss_mat, vs)
    # Mean squared loss
    if reduction == 'mean':
        mse = val_loss_mat.mean()
    elif reduction == 'sum':
        mse = val_loss_mat.sum()
    else:
        raise ValueError('Unknown reduction ' + reduction)

    if re:
        # Relative error.
        se = not_dones.bool()
        relerr = val_loss_mat/val_targ[se].abs()
        mre = relerr.abs().mean()
        msre = relerr.pow(2).mean()
        return mse, mre, msre

    return mse

def value_loss_returns(vs, returns, advantages, not_dones, params, old_vs,
                       mask=None, store=None, re=False):
    '''
    Returns (with time input) loss for the value function:
        L_t = (R_t - v(s, t))
    Inputs: rewards, returns, not_dones, params (from value_step)
    Outputs: value function loss
    '''
    assert shape_equal_cmp(vs, returns)
    sel = not_dones.bool()
    val_loss_mat = (vs - returns)[sel]
    mse = val_loss_mat.pow(2).mean()
    val_targ = returns

    if re:
        relerr = val_loss_mat/val_targ[sel].abs()
        mre = relerr.abs().mean()
        msre = relerr.pow(2).mean()
        return mse, mre, msre

    return mse

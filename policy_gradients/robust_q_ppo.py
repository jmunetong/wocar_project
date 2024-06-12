import numpy as np
import torch
from torch.nn.utils import parameters_to_vector as flatten
from torch.nn.utils import vector_to_parameters as assign
from .step_utils import adv_normalize, pack_history, surrogate_adv_q, shape_equal_cmp, get_state_kl_bound_sgld
from .value import value_step
from .convex_relaxation import get_kl_bound as get_state_kl_bound
import math

def robust_q_ppo_step(all_states, actions, old_log_ps, rewards, returns, not_dones, 
                advs, net, params, store, opt_step, relaxed_net, eps_scheduler, beta_scheduler, target_q_net,  value_net, n_steps):
    '''
    Proximal Policy Optimization with robustness regularizer
    Runs K epochs of PPO as in https://arxiv.org/abs/1707.06347
    Inputs:
    - all_states, the historical value of all the states
    - actions, the actions that the policy sampled
    - old_log_ps, the log probability of the actions that the policy sampled
    - advs, advantages as estimated by GAE
    - net, policy network to train [WILL BE MUTATED]
    - params, additional placeholder for parameters like EPS
    Returns:
    - The PPO loss; main job is to mutate the net
    '''
    # Storing batches of stuff
    # if store is not None:
    #     orig_dists = net(all_states)

    ### ACTUAL PPO OPTIMIZATION START
    if params.SHARE_WEIGHTS:
        orig_vs = net.get_value(all_states).squeeze(-1).view([params.NUM_ACTORS, -1])
        old_vs = orig_vs.detach()

    # We treat all PPO epochs as one epotorch.
    eps_scheduler.set_epoch_length(params.PPO_EPOCHS * params.NUM_MINIBATCHES)
    beta_scheduler.set_epoch_length(params.PPO_EPOCHS * params.NUM_MINIBATCHES)
    # We count from 1.
    eps_scheduler.step_epoch()
    beta_scheduler.step_epoch()

    if params.WEIGHT_SCHEDULE == 'constant':
        q_weight = params.Q_WEIGHT
    elif params.WEIGHT_SCHEDULE == 'linear':
        q_weight = (params.Q_WEIGHT/params.TRAIN_STEPS)  * float(n_steps)
    elif params.WEIGHT_SCHEDULE == 'exp':
        q_weight = math.pow((params.Q_WEIGHT/params.TRAIN_STEPS) * float(n_steps), 3)
    elif params.WEIGHT_SCHEDULE == 'stage':
        if n_steps < params.TRAIN_STEPS/2:
            q_weight = 0
        else:
            q_weight = params.Q_WEIGHT
    elif params.WEIGHT_SCHEDULE == 'stage-linear':
        stage_steps = (params.Q_WEIGHT/0.8)  * float(params.TRAIN_STEPS)
        if n_steps <= stage_steps:
            q_weight = (params.Q_WEIGHT/stage_steps)  * float(n_steps)
        else:
            q_weight = params.Q_WEIGHT
    else:
        print("Unsupported weight schedule")
    if q_weight > 1:
        q_weight = 1

    if params.HISTORY_LENGTH > 0:
        # LSTM policy. Need to go over all episodes instead of states.
        # We normalize all advantages at once instead of batch by batch, since each batch may contain different number of samples.
        normalized_advs = adv_normalize(advs)
        batches, alive_masks, time_masks, lengths = pack_history([all_states, actions, old_log_ps, normalized_advs], not_dones, max_length=params.HISTORY_LENGTH)


    for _ in range(params.PPO_EPOCHS):
        if params.HISTORY_LENGTH > 0:
            # LSTM policy. Need to go over all episodes instead of states.
            params.POLICY_ADAM.zero_grad()
            hidden = None
            surrogate = 0.0
            for i, batch in enumerate(batches):
                # Now we get chunks of time sequences, each of them with a maximum length of params.HISTORY_LENGTH.
                # select log probabilities, advantages of this minibattorch.
                batch_states, batch_actions, batch_old_log_ps, batch_advs = batch
                mask = time_masks[i]
                batch_worst_q = target_q_net(torch.cat((batch_states, batch_actions), dim=1)).squeeze(-1)
                # keep only the alive hidden states.
                if hidden is not None:
                    # print('hidden[0]', hidden[0].size())
                    hidden = [h[:, alive_masks[i], :].detach() for h in hidden]
                    # print('hidden[0]', hidden[0].size())
                # dist contains mean and variance of Gaussian.
                mean, std, hidden = net.multi_forward(batch_states, hidden=hidden)
                dist = mean, std
                # Convert state distribution to log likelyhood.
                new_log_ps = net.get_loglikelihood(dist, batch_actions)

                shape_equal_cmp(new_log_ps, batch_old_log_ps)

                # Calculate rewards
                # the surrogate rewards is basically exp(new_log_ps - old_log_ps) * advantage
                # dimension is the same as minibatch size.
                # We already normalized advs before. No need to normalize here.
                unclp_rew = surrogate_adv_q(batch_advs, new=new_log_ps, old=batch_old_log_ps, worst_q=batch_worst_q, q_weight=q_weight, mask=mask, normalize=False)
                clp_rew = surrogate_adv_q(batch_advs, new=new_log_ps, old=batch_old_log_ps, worst_q=batch_worst_q, q_weight=q_weight, clip_eps=params.CLIP_EPS, mask=mask, normalize=False)

                # Total loss, is the min of clipped and unclipped reward for each state, averaged.
                surrogate_batch = (-torch.min(unclp_rew, clp_rew) * mask).sum()
                # We sum the batch loss here because each batch contains uneven number of trajactories.
                surrogate = surrogate + surrogate_batch

            # Divide surrogate loss by number of samples in this battorch.
            surrogate = surrogate / all_states.size(0)
            # Calculate entropy bonus
            # So far, the entropy only depends on std and does not depend on time. No need to mask.
            entropy_bonus = net.entropies(dist)
            # Calculate regularizer under state perturbation.
            eps_scheduler.step_batch()
            beta_scheduler.step_batch()
            batch_action_means = None
            current_eps = eps_scheduler.get_eps()
            stdev = torch.exp(net.log_stdev)
            if params.ROBUST_PPO_DETACH_STDEV:
                # Detach stdev so that it won't be too large.
                stdev = stdev.detach()
            if params.ROBUST_PPO_METHOD == "sgld":
                kl_upper_bound = get_state_kl_bound_sgld(net, all_states, None,
                        eps=current_eps, steps=params.ROBUST_PPO_PGD_STEPS,
                        stdev=stdev, not_dones=not_dones).mean()
            else:
                raise ValueError(f"Unsupported robust PPO method {params.ROBUST_PPO_METHOD}")
            entropy = -params.ENTROPY_COEFF * entropy_bonus
            loss = surrogate + entropy + params.ROBUST_PPO_REG * kl_upper_bound
            # optimizer (only ADAM)
            loss.backward()
            if params.CLIP_GRAD_NORM != -1:
                torch.nn.utils.clip_grad_norm(net.parameters(), params.CLIP_GRAD_NORM)
            params.POLICY_ADAM.step()
        else:
            # Memoryless policy.
            # State is in shape (experience_size, observation_size). Usually 2048.
            state_indices = np.arange(all_states.shape[0])
            np.random.shuffle(state_indices)
            # We use a minibatch of states to do optimization, and each epoch contains several iterations.
            splits = np.array_split(state_indices, params.NUM_MINIBATCHES)
            # A typical mini-batch size is 2048/32=64
            for selected in splits:
                def sel(*args):
                    return [v[selected] for v in args]

                # old_log_ps: log probabilities of actions sampled based in experience buffer.
                # advs: advantages of these states.
                # both old_log_ps and advs are in shape (experience_size,) = 2048.
                tup = sel(all_states, actions, old_log_ps, advs)
                # select log probabilities, advantages of this minibattorch.
                batch_states, batch_actions, batch_old_log_ps, batch_advs = tup
                batch_worst_q = target_q_net(torch.cat((batch_states, batch_actions), dim=1)).squeeze(-1)
                batch_value = value_net(batch_states).squeeze(-1)
                '''
                print('batch states', batch_states.size())
                print('batch actions', batch_actions.size())
                print('batch old_log_ps', batch_old_log_ps.size())
                print('batch advs', batch_advs.size())
                '''
                # Forward propagation on current parameters (being constantly updated), to get distribution of these states
                # dist contains mean and variance of Gaussian.
                dist = net(batch_states)
                # Convert state distribution to log likelyhood.
                new_log_ps = net.get_loglikelihood(dist, batch_actions)

                shape_equal_cmp(new_log_ps, batch_old_log_ps)

                # print('advs:', batch_advs)
                # print('worst_q', batch_worst_q)
                # Calculate rewards
                # the surrogate rewards is basically exp(new_log_ps - old_log_ps) * advantage
                # dimension is the same as minibatch size.
                unclp_rew = surrogate_adv_q(batch_advs, new=new_log_ps, old=batch_old_log_ps, worst_q=batch_worst_q, q_weight=q_weight)
                clp_rew = surrogate_adv_q(batch_advs, new=new_log_ps, old=batch_old_log_ps, worst_q=batch_worst_q, q_weight=q_weight, clip_eps=params.CLIP_EPS)

                # unclp_ori_rew = surrogate_reward(batch_advs, new=new_log_ps, old=batch_old_log_ps)
                # clp_ori_rew = surrogate_reward(batch_advs, new=new_log_ps, old=batch_old_log_ps, clip_eps=params.CLIP_EPS)
                # Calculate entropy bonus
                entropy_bonus = net.entropies(dist).mean()

                # Calculate regularizer under state perturbation.
                eps_scheduler.step_batch()
                beta_scheduler.step_batch()
                batch_action_means = dist[0]
                current_eps = eps_scheduler.get_eps()
                stdev = torch.exp(net.log_stdev)
                if params.ROBUST_PPO_DETACH_STDEV:
                    # Detach stdev so that it won't be too large.
                    stdev = stdev.detach()
                if params.ROBUST_PPO_METHOD == "convex-relax":
                    kl_upper_bound = get_state_kl_bound(relaxed_net, batch_states, batch_action_means,
                            eps=current_eps, beta=beta_scheduler.get_eps(),
                            stdev=stdev).mean()
                elif params.ROBUST_PPO_METHOD == "sgld":
                    kl_upper_bound = get_state_kl_bound_sgld(net, batch_states, batch_action_means,
                            eps=current_eps, steps=params.ROBUST_PPO_PGD_STEPS,
                            stdev=stdev).mean()
                else:
                    raise ValueError(f"Unsupported robust PPO method {params.ROBUST_PPO_METHOD}")

                # Total loss, is the min of clipped and unclipped reward for each state, averaged.
                surrogate = -torch.min(unclp_rew, clp_rew).mean()
                entropy = -params.ENTROPY_COEFF * entropy_bonus
                loss = surrogate + entropy + params.ROBUST_PPO_REG * kl_upper_bound
                
                # If we are sharing weights, take the value step simultaneously 
                # (since the policy and value networks depend on the same weights)
                if params.SHARE_WEIGHTS:
                    tup = sel(returns, not_dones, old_vs)
                    batch_returns, batch_not_dones, batch_old_vs = tup
                    val_loss = value_step(batch_states, batch_returns, batch_advs,
                                          batch_not_dones, net.get_value, None, params,
                                          store, old_vs=batch_old_vs, opt_step=opt_step)
                    loss += params.VALUE_MULTIPLIER * val_loss

                # Optimizer step (Adam or SGD)
                if params.POLICY_ADAM is None:
                    grad = torch.autograd.grad(loss, net.parameters())
                    flat_grad = flatten(grad)
                    if params.CLIP_GRAD_NORM != -1:
                        norm_grad = torch.norm(flat_grad)
                        flat_grad = flat_grad if norm_grad <= params.CLIP_GRAD_NORM else \
                                    flat_grad / norm_grad * params.CLIP_GRAD_NORM

                    assign(flatten(net.parameters()) - params.PPO_LR * flat_grad, net.parameters())
                else:
                    params.POLICY_ADAM.zero_grad()
                    loss.backward()
                    if params.CLIP_GRAD_NORM != -1:
                        torch.nn.utils.clip_grad_norm(net.parameters(), params.CLIP_GRAD_NORM)
                    params.POLICY_ADAM.step()
        # Logging.
        kl_upper_bound = kl_upper_bound.item()
        surrogate = surrogate.item()
        entropy_bonus = entropy_bonus.item()
        worst_q = batch_worst_q.mean().item()
        curr_vs = batch_value.mean().item()
        print(f'eps={eps_scheduler.get_eps():8.6f}, beta={beta_scheduler.get_eps():8.6f}, kl={kl_upper_bound:10.5g}, '
              f'surrogate={surrogate:8.5f}, entropy={entropy_bonus:8.5f}, loss={loss.item():8.5f}, worst_q={worst_q:8.5f}, curr_vs={curr_vs:8.5f}')
    std = torch.exp(net.log_stdev)
    print(f'std_min={std.min().item():8.5f}, std_max={std.max().item():8.5f}, std_mean={std.mean().item():8.5f}')
    print(f'worst_q_weight={q_weight:8.5f}')

    # TODO: ADV: add row name suffix
    row ={
        'eps': eps_scheduler.get_eps(),
        'beta': beta_scheduler.get_eps(),
        'kl': kl_upper_bound,
        'surrogate': surrogate,
        'entropy': entropy_bonus,
        'loss': loss.item(),
        'worst_q': worst_q,
        'worst_q_weight': q_weight,
        'current_value': curr_vs
    }


    return loss.item(), surrogate, entropy_bonus, row

import torch 
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import functools
import math

def pack_history(features, not_dones, max_length):
    # Features is a list, each element has dimension (N, state_dim) or (N, ) where N contains a few episodes
    # not_dones splits these episodes (0 in not_dones is end of an episode)
    nnz = torch.nonzero(1.0 - not_dones, as_tuple=False).view(-1).cpu().numpy()
    # nnz has the position where not_dones = 0 (end of episode)
    assert isinstance(features, list)
    # Check dimension. All tensors must have the same dimension.
    size = features[0].size(0)
    for t in features:
        assert size == t.size(0)
    all_pieces = [[] for i in range(len(features))]
    lengths = []
    start = 0
    for i in nnz:
        end = i + 1
        for (a, b) in zip(all_pieces, features):
            a.append(b[start:end])
        lengths.append(end - start)
        start = end
    # The last episode is missing, unless the previous episode end at the last element.
    if end != size:
        for (a, b) in zip(all_pieces, features):
            a.append(b[end:])
        lengths.append(size - end)
    # First pad to longest sequence
    padded_features = [pad_sequence(a, batch_first=True) for a in all_pieces]
    # Then pad to a multiple of max_length
    longest = padded_features[0].size(1)
    extra = int(math.ceil(longest / max_length) * max_length - longest)
    new_padded_features = []
    for t in padded_features:
        if t.ndim == 3:
            new_tensor = torch.zeros(t.size(0), extra, t.size(2))
        else:
            new_tensor = torch.zeros(t.size(0), extra)
        new_tensor = torch.cat([t, new_tensor], dim=1)
        new_padded_features.append(new_tensor)
    del padded_features
    # now divide padded features into chunks with max_length.
    nbatches = new_padded_features[0].size(1) // max_length
    alive_masks = []  # which batch still alives after a chunk
    # time step masks for each chunk, each battorch.
    time_masks = []
    batches = [[] for i in range(nbatches)]  # batch of batches
    alive = torch.tensor(lengths)
    alive_iter = torch.tensor(lengths)
    for i in range(nbatches):
        full_mask = alive > 0
        iter_mask = alive_iter > 0
        for t in new_padded_features:
            # only keep the tensors that are alive
            batches[i].append(t[full_mask, i * max_length : i * max_length + max_length])
        # Remove deleted batches
        alive_iter = alive_iter[iter_mask]
        time_mask = alive_iter.view(-1, 1) > torch.arange(max_length).view(1, -1)
        alive -= max_length
        alive_iter -= max_length
        alive_masks.append(iter_mask)
        time_masks.append(time_mask)
    return batches, alive_masks, time_masks, lengths


def adv_normalize(adv, mask=None):
    if mask is None:
        if adv.nelement() == 1:
            return adv
        std = adv.std()
        mean = adv.mean()
    else:
        masked_adv = adv[mask]
        if masked_adv.nelement() == 1:
            return adv
        std = masked_adv.std()
        mean = masked_adv.mean()

    
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - mean)/(std + 1e-8)
    return n_advs

def surrogate_reward(adv, *, new, old, clip_eps=None, mask=None, normalize=True):
    '''
    Computes the surrogate reward for TRPO and PPO:
    R(\theta) = E[r_t * A_t]
    with support for clamping the ratio (for PPO), s.t.
    R(\theta) = E[clamp(r_t, 1-e, 1+e) * A_t]
    Inputs:
    - adv, unnormalized advantages as calculated by the agents
    - log_ps_new, the log probabilities assigned to taken events by \theta_{new}
    - log_ps_old, the log probabilities assigned to taken events by \theta_{old}
    - clip_EPS, the clipping boundary for PPO loss
    Returns:
    - The surrogate loss as described above
    '''
    log_ps_new, log_ps_old = new, old

    if normalize:
        # Normalized Advantages
        n_advs = adv_normalize(adv, mask)
    else:
        n_advs = adv

    assert shape_equal_cmp(log_ps_new, log_ps_old, n_advs)

    # Ratio of new probabilities to old ones
    ratio_new_old = torch.exp(log_ps_new - log_ps_old)

    # Clamping (for use with PPO)
    if clip_eps is not None:
        ratio_new_old = torch.clamp(ratio_new_old, 1-clip_eps, 1+clip_eps)

    return ratio_new_old * n_advs

def surrogate_adv_q(adv, *, new, old, worst_q, q_weight=0, clip_eps=None, mask=None, normalize=True):
    '''
    Computes the surrogate reward for TRPO and PPO:
    R(\theta) = E[r_t * A_t]
    with support for clamping the ratio (for PPO), s.t.
    R(\theta) = E[clamp(r_t, 1-e, 1+e) * A_t]
    Inputs:
    - adv, unnormalized advantages as calculated by the agents
    - log_ps_new, the log probabilities assigned to taken events by \theta_{new}
    - log_ps_old, the log probabilities assigned to taken events by \theta_{old}
    - clip_EPS, the clipping boundary for PPO loss
    Returns:
    - The surrogate loss as described above
    '''
    log_ps_new, log_ps_old = new, old

    if normalize:
        # Normalized Advantages
        n_advs = adv_normalize(adv, mask)
        n_worst_q = adv_normalize(worst_q, mask)
    else:
        n_advs = adv
        n_worst_q = worst_q

    assert shape_equal_cmp(log_ps_new, log_ps_old, n_advs)

    # Ratio of new probabilities to old ones
    ratio_new_old = torch.exp(log_ps_new - log_ps_old)

    # Clamping (for use with PPO)
    if clip_eps is not None:
        ratio_new_old = torch.clamp(ratio_new_old, 1-clip_eps, 1+clip_eps)

    robust_adv = n_advs + q_weight * n_worst_q

    return ratio_new_old * robust_adv

def get_params_norm(net, p=2):
    layer_norms = []
    layer_norms_dict = {}
    for name, params in net.named_parameters():
        if name != 'log_stdev' and name != 'log_weight' and params.ndim != 1:
            norm = torch.norm(params.view(-1), p=p).item() / np.prod(params.size())
            layer_norms.append(norm)
            layer_norms_dict[name] = norm
    return np.array(layer_norms), layer_norms_dict


"""Computing an estimated upper bound of KL divergence using SGLD."""
def get_state_kl_bound_sgld(net, batch_states, batch_action_means, eps, steps, stdev, not_dones=None):
    if not_dones is not None:
        # If we have not_dones, the underlying network is a LSTM.
        wrapped_net = functools.partial(net, not_dones=not_dones)
    else:
        wrapped_net = net
    if batch_action_means is None:
        # Not provided. We need to compute them.
        with torch.no_grad():
            batch_action_means, _ = wrapped_net(batch_states)
    else:
        batch_action_means = batch_action_means.detach()
    # upper and lower bounds for clipping
    states_ub = batch_states + eps
    states_lb = batch_states - eps
    step_eps = eps / steps
    # SGLD noise factor. We set (inverse) beta=1e-5 as gradients are relatively small here.
    beta = 1e-5
    noise_factor = np.sqrt(2 * step_eps * beta)
    noise = torch.randn_like(batch_states) * noise_factor
    var_states = (batch_states.clone() + noise.sign() * step_eps).detach().requires_grad_()
    for i in range(steps):
        # Find a nearby state new_phi that maximize the difference
        diff = (wrapped_net(var_states)[0] - batch_action_means) / stdev.detach()
        kl = (diff * diff).sum(axis=-1, keepdim=True).mean()
        # Need to clear gradients before the backward() for policy_loss
        kl.backward()
        # Reduce noise at every step.
        noise_factor = np.sqrt(2 * step_eps * beta) / (i+2)
        # Project noisy gradient to step boundary.
        update = (var_states.grad + noise_factor * torch.randn_like(var_states)).sign() * step_eps
        var_states.data += update
        # clip into the upper and lower bounds
        var_states = torch.max(var_states, states_lb)
        var_states = torch.min(var_states, states_ub)
        var_states = var_states.detach().requires_grad_()
    net.zero_grad()
    diff = (wrapped_net(var_states.requires_grad_(False))[0] - batch_action_means) / stdev
    return (diff * diff).sum(axis=-1, keepdim=True)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def shape_equal_cmp(*args):
    '''
    Checks that the shapes of the passed arguments are equal
    Inputs:
    - All arguments should be tensors
    Returns:
    - True if all arguments have the same shape, else ValueError
    '''
    for i in range(len(args)-1):
        if args[i].shape != args[i+1].shape:
            s = "\n".join([str(x.shape) for x in args])
            raise ValueError("Expected equal shapes. Got:\n%s" % s)
    return True

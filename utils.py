import sys
import argparse
import numpy as np
import policy_gradients.models as models
import datetime


def generate_experiment_id():
    # Get the current date and time
    now = datetime.datetime.now()
    # Format date and time as YYYYMMDD_HHMMSS
    formatted_date_time = now.strftime('%Y%m%d_%H%M%S')
    # Create the experiment ID
    experiment_id = f"exp_{formatted_date_time}"
    return experiment_id


# Tee object allows for logging to both stdout and to file
class Tee(object):
    def __init__(self, file_path, stream_type, mode='a'):
        assert stream_type in ['stdout', 'stderr']

        self.file = open(file_path, mode)
        self.stream_type = stream_type
        self.errors = 'chill'

        if stream_type == 'stdout':
            self.stream = sys.stdout
            sys.stdout = self
        else:
            self.stream = sys.stderr
            sys.stderr = self

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def override_json_params(params, json_params, excluding_params):
    # Override the JSON config with the argparse config
    missing_keys = []
    for key in json_params:
        if key not in params:
            missing_keys.append(key)
    assert not missing_keys, "Following keys not in args: " + str(missing_keys)

    missing_keys = []
    for key in params:
        if key not in json_params and key not in excluding_params:
            missing_keys.append(key)
    # assert not missing_keys, "Following keys not in JSON: " + str(missing_keys)

    json_params.update({k: params[k] for k in params if params[k] is not None})
    return json_params


def add_common_parser_opts(parser):
    # Basic setup
    parser.add_argument('--game', type=str, help='gym game')
    parser.add_argument('--mode', type=str, choices=['ppo', 'trpo', 'robust_ppo', 'adv_ppo', 'adv_trpo', 'adv_sa_ppo', 'robust_q_ppo'],
                        help='pg alg')
    parser.add_argument('--out-dir', type=str,
                        help='out dir for store + logging')
    parser.add_argument('--advanced-logging', type=str2bool, const=True, nargs='?')
    parser.add_argument('--kl-approximation-iters', type=int,
                        help='how often to do kl approx exps')
    parser.add_argument('--log-every', type=int)
    parser.add_argument('--policy-net-type', type=str,
                        choices=models.POLICY_NETS.keys())
    parser.add_argument('--value-net-type', type=str,
                        choices=models.VALUE_NETS.keys())
    parser.add_argument('--train-steps', type=int,
                        help='num agent training steps')
    parser.add_argument('--cpu', type=str2bool, const=True, nargs='?')

    # Which value loss to use
    parser.add_argument('--value-calc', type=str,
                        help='which value calculation to use')
    parser.add_argument('--initialization', type=str)

    # General Policy Gradient parameters
    parser.add_argument('--num-actors', type=int, help='num actors (serial)',
                        choices=[1])
    parser.add_argument('--t', type=int,
                        help='num timesteps to run each actor for')
    parser.add_argument('--gamma', type=float, help='discount on reward')
    parser.add_argument('--lambda', type=float, help='GAE hyperparameter')
    parser.add_argument('--val-lr', type=float, help='value fn learning rate')
    parser.add_argument('--val-epochs', type=int, help='value fn epochs')
    parser.add_argument('--initial-std', type=float, help='initial value of std for Gaussian policy. Default is 1.')

    # PPO parameters
    parser.add_argument('--adam-eps', type=float, choices=[0, 1e-5], help='adam eps parameter')

    parser.add_argument('--num-minibatches', type=int,
                        help='num minibatches in ppo per epoch')
    parser.add_argument('--ppo-epochs', type=int)
    parser.add_argument('--ppo-lr', type=float,
                        help='if nonzero, use gradient descent w this lr')
    parser.add_argument('--ppo-lr-adam', type=float,
                        help='if nonzero, use adam with this lr')
    parser.add_argument('--anneal-lr', type=str2bool,
                        help='if we should anneal lr linearly from start to finish')
    parser.add_argument('--clip-eps', type=float, help='ppo clipping')
    parser.add_argument('--clip-val-eps', type=float, help='ppo clipping value')
    parser.add_argument('--entropy-coeff', type=float,
                        help='entropy weight hyperparam')
    parser.add_argument('--value-clipping', type=str2bool,
                        help='should clip values (w/ ppo eps)')
    parser.add_argument('--value-multiplier', type=float,
                        help='coeff for value loss in combined step ppo loss')
    parser.add_argument('--share-weights', type=str2bool,
                        help='share weights in valnet and polnet')
    parser.add_argument('--clip-grad-norm', type=float,
                        help='gradient norm clipping (-1 for no clipping)')
    parser.add_argument('--policy-activation', type=str,
                        help='activation function for countinous policy network')
    parser.add_argument('--value-activation', type=str,
                        help='activation function for value network')
    # TRPO parameters
    parser.add_argument('--max-kl', type=float, help='trpo max kl hparam')
    parser.add_argument('--max-kl-final', type=float, help='trpo max kl final')
    parser.add_argument('--fisher-frac-samples', type=float,
                        help='frac samples to use in fisher vp estimate')
    parser.add_argument('--cg-steps', type=int,
                        help='num cg steps in fisher vp estimate')
    parser.add_argument('--damping', type=float, help='damping to use in cg')
    parser.add_argument('--max-backtrack', type=int, help='max bt steps in fvp')
    parser.add_argument('--trpo-kl-reduce-func', type=str, help='reduce function for KL divergence used in line search. mean or max.')

    # Robust PPO parameters.
    parser.add_argument('--robust-ppo-eps', type=float, help='max eps for robust PPO training')
    parser.add_argument('--robust-ppo-method', type=str, choices=['convex-relax', 'sgld', 'pgd'], help='robustness regularization methods')
    parser.add_argument('--robust-ppo-pgd-steps', type=int, help='number of PGD optimization steps')
    parser.add_argument('--robust-ppo-detach-stdev', type=str2bool, help='detach gradient of standard deviation term')
    parser.add_argument('--robust-ppo-reg', type=float, help='robust PPO regularization')
    parser.add_argument('--robust-ppo-eps-scheduler-opts', type=str, help='options for epsilon scheduler for robust PPO training')
    parser.add_argument('--robust-ppo-beta', type=float, help='max beta (IBP mixing factor) for robust PPO training')
    parser.add_argument('--robust-ppo-beta-scheduler-opts', type=str, help='options for beta scheduler for robust PPO training')

    # Adversarial PPO parameters.
    parser.add_argument('--adv-ppo-lr-adam', type=float,
                        help='if nonzero, use adam for adversary policy with this lr')
    parser.add_argument('--adv-entropy-coeff', type=float,
                        help='entropy weight hyperparam for adversary policy')
    parser.add_argument('--adv-eps', type=float, help='adversary perturbation eps')
    parser.add_argument('--adv-clip-eps', type=float, help='ppo clipping for adversary policy')
    parser.add_argument('--adv-val-lr', type=float, help='value fn learning rate for adversary policy')
    parser.add_argument('--adv-policy-steps', type=float, help='number of policy steps before adversary steps')
    parser.add_argument('--adv-adversary-steps', type=float, help='number of adversary steps before adversary steps')
    parser.add_argument('--adv-adversary-ratio', type=float, help='percentage of frames to attack for the adversary')

    # Adversarial attack parameters.
    parser.add_argument('--attack-method', type=str, choices=["none", "critic", "random", "action", "sarsa", "sarsa+action", "advpolicy", "action+imit"], help='adversarial attack methods.')
    parser.add_argument('--attack-ratio', type=float, help='attack only a ratio of steps.')
    parser.add_argument('--attack-steps', type=int, help='number of PGD optimization steps.')
    parser.add_argument('--attack-eps', type=str, help='epsilon for attack. If set to "same", we will use value of robust-ppo-eps.')
    parser.add_argument('--attack-step-eps', type=str, help='step size for each iteration. If set to "auto", we will use attack-eps / attack-steps')
    parser.add_argument('--attack-sarsa-network', type=str, help='sarsa network to load for attack.')
    parser.add_argument('--attack-sarsa-action-ratio', type=float, help='When set to non-zero, enable sarsa-action attack.')
    parser.add_argument('--attack-advpolicy-network', type=str, help='adversarial policy network to load for attack.')
    parser.add_argument('--collect-perturbed-states', type=str2bool, help='collect perturbed states during training')

    # Normalization parameters
    parser.add_argument('--norm-rewards', type=str, help='type of rewards normalization', 
                        choices=['rewards', 'returns', 'none'])
    parser.add_argument('--norm-states', type=str2bool, help='should norm states')
    parser.add_argument('--clip-rewards', type=float, help='clip rews eps')
    parser.add_argument('--clip-observations', type=float, help='clips obs eps')

    # Sequence training parameters
    parser.add_argument('--history-length', type=int, help='length of history to use for LSTM. If <= 1, we do not use LSTM.')
    parser.add_argument('--use-lstm-val', type=str2bool, help='use a lstm for value function')

    # Saving
    parser.add_argument('--save-iters', type=int, help='how often to save model (0 = no saving)')
    parser.add_argument('--force-stop-step', type=int, help='forcibly terminate after a given number of steps. Useful for debugging and tuning.')

    # Visualization
    parser.add_argument('--show-env', type=str2bool, help='Show environment visualization')
    parser.add_argument('--save-frames', type=str2bool, help='Save environment frames')
    parser.add_argument('--save-frames-path', type=str, help='Path to save environment frames')

    # Worst Q Estimation parmeters
    parser.add_argument('--q-epochs', type=int)
    parser.add_argument('--q-lr', type=float)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--weight-schedule', type=str, help='Worst q weight schedule')
    parser.add_argument('--q-weight', type=float, help='Max worst q weight in policy loss')

    return parser


def add_adversary_to_table(params, p, table_dict):
        if params['mode'] == "adv_ppo" or params['mode'] == 'adv_trpo' or params['mode'] == 'adv_sa_ppo':
            table_dict["adversary_policy_model"] = p.adversary_policy_model.state_dict()
            table_dict["adversary_policy_opt"] = p.ADV_POLICY_ADAM.state_dict()
            table_dict["adversary_val_model"] = p.adversary_val_model.state_dict()
            table_dict["adversary_val_opt"] = p.adversary_val_opt.state_dict()
        return table_dict

def add_q_to_table(params, p, table_dict):
    if params['mode'] == 'robust_q_ppo':
        table_dict["q_model"] = p.q_model.state_dict()
        table_dict["q_opt"] = p.q_opt.state_dict()
    return table_dict

def finalize_table(p, final_table, iteration, terminated_early, rewards, params):
    final_5_rewards = np.array(rewards)[-5:].mean()
    final_dict = {
        'iteration': iteration,
        '5_rewards': final_5_rewards,
        'terminated_early': terminated_early,
        'val_model': p.val_model.state_dict(),
        'policy_model': p.policy_model.state_dict(),
        'policy_opt': p.POLICY_ADAM.state_dict(),
        'val_opt': p.val_opt.state_dict(),
        'envs': p.envs
    }
    final_dict = add_adversary_to_table(params, p, final_dict)
    final_dict = add_q_to_table(params, p, final_dict)
    final_table.append_row(final_dict)

from  trainer import Trainer
import random
import numpy as np
import os
import argparse
import traceback
import json
import torch
import h5py

from  utils import *
from cox.store import Store, schema_from_dict
BASE_EXPERIMENT_PATH = "./experiments"

def save_models(p, base_path, params):
    path = base_path
    os.makedirs(path, exist_ok=True)
    torch.save(p.policy_model.state_dict(), os.path.join(path, "policy_mod.pth"))
    # torch.save(p.policy_opt.state_dict(), os.path.join(path, "policy_opt.pth"))
    torch.save(p.val_model.state_dict(), os.path.join(path, "value_mod.pth"))
    torch.save(p.val_opt.state_dict(), os.path.join(path, "value_opt.pth"))
    if params['mode'] == 'robust_q_ppo':
         torch.save(p.q_model.state_dict(), os.path.join(path, "q_mod.pth"))
         torch.save(p.q_opt.state_dict(), os.path.join(path, "q_opt.pth"))
    print(f'Models saved in {base_path}')

def main(v, config):
    for k, v in zip(params.keys(), params.values()):
        assert v is not None, f"Value for {k} is None"
    
    base_directory = os.path.join(BASE_EXPERIMENT_PATH,f' {params["out_dir"]}_{params["attack_method"]}', generate_experiment_id())
    os.makedirs(base_directory, exist_ok=True)
    p, model = Trainer.agent_from_params(params)
    with open(os.path.join(base_directory,'experiment_config.json'), 'w') as json_file:
        json.dump(config, json_file,  indent=4, separators=(',', ': '))
    if params['initial_std'] != 1.0:
        p.policy_model.log_stdev.data[:] = np.log(params['initial_std'])
    
    rewards = []
    ret = 0
    final_dict = {}
    try:
        for i in range(params['train_steps']):
            print('Step %d' % (i,))
            d: dict = {}
            mean_reward, d = p.train_step()
            rewards.append(mean_reward)
            if i == 0:
                final_dict['mean_reward'] = [mean_reward]
                for key in d.keys():
                    final_dict[key] = [d[key]]
                    
            else:
                final_dict['mean_reward'].append(mean_reward)
                for key in d.keys():
                    final_dict[key].append(d[key]) 
                 
        with h5py.File(os.path.join(base_directory,'data.h5'), 'w') as hdf_file:
            for key, value in final_dict.items():
                if isinstance(value, (int, float, str)): 
                    hdf_file.attrs[key] = value
                else: 
                    hdf_file.create_dataset(key, data=value)

            print("Dictionary saved to data.h5")
        save_models(p, base_path=base_directory, params=params)
    except KeyboardInterrupt:
        ret = 1
    except:
        print("An error occurred during training:")
        traceback.print_exc()
        ret = -1
    print(f'Models saved to {base_directory}')
    print(model)
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate experiments to be run.')
    parser.add_argument('--config-path', type=str, required=True,
                        help='json for this config')
    parser.add_argument('--out-dir-prefix', type=str, default="", required=False,
                        help='prefix for output log path')
    parser.add_argument('--load-model', type=str, default=None, required=False, help='load pretrained model and optimizer states before training')
    parser.add_argument('--no-load-adv-policy', action='store_true', required=False, help='Do not load adversary policy and value network from pretrained model.')
    parser.add_argument('--adv-policy-only', action='store_true', required=False, help='Run adversary only, by setting main agent learning rate to 0')
    parser.add_argument('--deterministic', action='store_true', help='disable Gaussian noise in action for --adv-policy-only mode')
    parser.add_argument('--seed', type=int, help='random seed', default=-1)
    parser = add_common_parser_opts(parser)
    
    args = parser.parse_args()
    params = vars(args)
        
    seed = params['seed']
    json_params = json.load(open(args.config_path))

    extra_params = ['config_path', 'out_dir_prefix', 'load_model', 'no_load_adv_policy', 'adv_policy_only', 'deterministic', 'seed']
    params = override_json_params(params, json_params, extra_params)

    if params['adv_policy_only']:
        if params['adv_ppo_lr_adam'] == 'same':
            params['adv_ppo_lr_adam'] = params['ppo_lr_adam']
            print(f"automatically setting adv_ppo_lr_adam to {params['adv_ppo_lr_adam']}")
        print('disabling policy training (train adversary only)')
        params['ppo_lr_adam'] = 0.0 * params['ppo_lr_adam']
    else:
        # deterministic mode only valid when --adv-policy-only is set
        assert not params['deterministic']

    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
    torch.set_printoptions(threshold=5000, linewidth=120)

    # Append a prefix for output path.
    if args.out_dir_prefix:
        params['out_dir'] = os.path.join(args.out_dir_prefix, params['out_dir'])
        print(f"setting output dir to {params['out_dir']}")

    attacks = ["none", "random", "critic"]
    # attacks = ["none"]
    for attack in attacks:
        params['attack_method']=attack
        main(params, params)


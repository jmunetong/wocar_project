import numpy as np
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

def get_data(path, dataset, verbose=False):
    d = {}
    with h5py.File(path, 'r') as f:
        for dataset_name in f.keys():
            # if verbose:
            #     print(dataset_name)
            d[dataset_name] = np.array(f[dataset_name][:])
    return d[dataset]

def get_datasets(vanilla, robust_ppo, robust_q_ppo, data_name):
    v = get_data(vanilla, data_name, verbose=True)
    p =  get_data(robust_ppo, data_name)
    q =  get_data(robust_q_ppo, data_name)
    return v, p, q


def plot(v, p, q, config, title, x_axis, y_axis, log=False):
    epochs=np.arange(len(v))
    # Creating the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns
    title = F'{title.title()} {config.title()}'
    # Plotting data
    ax1.plot(epochs, v, label='Vanilla-PPO', color='tab:blue')
    ax1.set_title(title.title())
    ax1.set_xlabel(x_axis.title())
    ax1.set_ylabel(y_axis.title())
    if log:
        ax1.set_yscale('log')
    ax1.legend()

    ax2.plot(epochs, p, label='R-PPO', color='tab:orange')
    ax2.set_title(title.title())
    ax2.set_xlabel(x_axis.title())
    ax2.set_ylabel(y_axis.title())
    if log:
        ax2.set_yscale('log')
    ax2.legend()

    ax3.plot(epochs, q, label='WocaR-PPO', color='tab:green')
    ax3.set_title(title.title())
    ax3.set_xlabel(x_axis.title())
    ax3.set_ylabel(y_axis.title())
    if log:
        ax3.set_yscale('log')
    ax3.legend()

    # Showing the plot
    plt.tight_layout()
    # plt.show()


def plot_together(v, p, q, title, config, x, y):
    epochs=np.arange(len(v))
    # Creating the figure
    fig, ax = plt.subplots(figsize=(8, 6))  # Single plot
    title = f'{title} {config.title()}'
    
    ax.plot(epochs, v, label='Vanilla-PPO', color='tab:blue', linewidth=1.5)
    ax.plot(epochs, p, label='R-PPO', color='tab:orange', linewidth=1.5)
    ax.plot(epochs, q, label='WocaR-PPO', color='tab:green', linewidth=1.5)
    
    ax.set_title(title.title())
    ax.set_xlabel(x.title())
    ax.set_ylabel(y.title())
    
    ax.legend()  
    
    # Display the plot
    plt.tight_layout()
    # plt.show()


def generate_plots(vanilla_path, ppo_path, q_ppo_path, title, x_axis, y_axis, data_name, config, in_one=False):
    
    v, p, q = get_datasets(vanilla=vanilla_path, robust_ppo=ppo_path, robust_q_ppo=q_ppo_path,  data_name=data_name)
    if in_one:
        plot_together(v, p, q, title, config, x_axis, y_axis)
    else:
        plot(v, p, q, title=title, x_axis=x_axis, y_axis=y_axis, config=config, log=False)


def build_reward_plot(p_v, p_r, p_w, task, attack):
    dir = f'./results/{task}/{attack}'
    os.makedirs(dir, exist_ok=True)
    title = f'Mean Reward: {task}'
    config = f'Attack: {attack}'
    x = 'iteration'
    y = 'Reward'

    #Rewards
    data_name = 'mean_reward'
    generate_plots(vanilla_path=p_v,
                ppo_path=p_r,
                q_ppo_path=p_w,
                title=title,
                x_axis=x,
                y_axis=y,
                data_name=data_name,
                config=config, 
                in_one=False
                )
    plt.savefig(os.path.join(dir, f"mean_reward_{attack}.png"))

def build_std_plot(p_v, p_r, p_w, task, attack):
    dir = f'./results/{task}/{attack}'
    os.makedirs(dir, exist_ok=True)
    title = f'Mean Std: {task}'
    config = f'Attack: {attack}'
    x = 'iteration'
    y = 'std'

    #Rewards
    data_name = 'mean_std'
    generate_plots(vanilla_path=p_v,
                ppo_path=p_r,
                q_ppo_path=p_w,
                title=title,
                x_axis=x,
                y_axis=y,
                data_name=data_name,
                config=config, 
                in_one=True
                )
    plt.savefig(os.path.join(dir, f"mean_std_{attack}.png"))

def build_entropy_plot(p_v, p_r, p_w, task, attack):
    dir = f'./results/{task}/{attack}'
    os.makedirs(dir, exist_ok=True)
    title = f'Entropy Bonus: {task}'
    config = f'Attack: {attack}'
    x = 'iteration'
    y = 'H'

    #Rewards
    data_name = 'entropy_bonus'
    generate_plots(vanilla_path=p_v,
                ppo_path=p_r,
                q_ppo_path=p_w,
                title=title,
                x_axis=x,
                y_axis=y,
                data_name=data_name,
                config=config, 
                in_one=True
                )
    plt.savefig(os.path.join(dir, f"entropy_{attack}.png"))

def generate_results(p_v, p_r, p_w, task, attack):
    
    build_reward_plot(p_v, p_r, p_w, task, attack)
    build_std_plot(p_v, p_r, p_w, task, attack)
    build_entropy_plot(p_v, p_r, p_w, task, attack)

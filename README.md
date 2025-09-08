# Robust Reinforcement Learning: WocaR-RL Framework

A CS234 project implementing **Worst-case-aware Robust Reinforcement Learning (WocaR-RL)** for training robust policies resilient to adversarial attacks and distributional shifts.

## Overview

This project explores robust reinforcement learning methodologies to address the vulnerability of deep RL policies to minor perturbations and adversarial examples. The implementation focuses on the **WocaR-RL framework**, which enhances neural network awareness of potentially detrimental actions without requiring additional sample generation.

### Key Algorithms Implemented

- **Vanilla PPO (Proximal Policy Optimization)** - Baseline policy gradient method
- **Robust PPO (R-PPO)** - Enhanced exploration through noise injection  
- **WocaR-RL PPO** - Worst-case-aware robust reinforcement learning

## Project Structure

```
wocar_project/
├── run.py                      # Main training script
├── run_experiments.sh          # Batch experiment runner
├── trainer.py                  # Core training logic and Trainer class
├── utils.py                    # Utility functions and experiment helpers
├── evaluate.py                 # Model evaluation utilities
├── requirements.txt            # Python dependencies
├── config_experiments/         # JSON configuration files
│   ├── config_hopper_vanilla_ppo.json
│   ├── config_hopper_robust_ppo_sgld.json
│   ├── config_hopper_robust_q_ppo_sgld.json
│   └── ...
├── policy_gradients/           # Core RL algorithms implementation
├── cox/                        # Logging and experiment management
├── results/                    # Training results and outputs
├── release_models/             # Pre-trained model checkpoints
└── experiments/                # Generated experiment data
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jmunetong/wocar_project
   cd wocar_project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install MuJoCo environment** (if not already installed):
   - Follow [MuJoCo installation guide](https://github.com/openai/mujoco-py)

## Usage

### Single Experiment

Run a training experiment with a specific configuration:

```bash
python run.py --config-path config_experiments/config_hopper_vanilla_ppo.json
```

### Batch Experiments

Execute multiple experiments across different attack scenarios:

```bash
bash run_experiments.sh
```

This runs experiments for:
- Hopper environment with Vanilla PPO
- Hopper environment with Robust PPO + SGLD
- Hopper environment with WocaR-RL + SGLD

### Configuration Options

Key parameters in configuration files:

- `mode`: Algorithm type (`"ppo"`, `"robust_q_ppo"`, etc.)
- `game`: Environment (`"Hopper-v4"`, `"HalfCheetah-v4"`)
- `attack_method`: Attack type (`"none"`, `"random"`, `"critic"`)
- `train_steps`: Number of training iterations (default: 970)
- `ppo_lr_adam`: Learning rate for policy updates
- `robust_ppo_eps`: Robustness perturbation radius

## Experimental Results

### Performance Summary

Results on Hopper-v4 environment (last 100 training iterations):

| Algorithm | No Attack | Random Attack | Critic Attack |
|-----------|-----------|---------------|---------------|
| **PPO**   | 2031      | 2994          | 2280          |
| **R-PPO** | 1540      | 1407          | 1672          |
| **WocaR** | **2494**  | 2039          | **2330**      |

### Key Findings

- **WocaR-RL** demonstrates superior robustness under both no-attack and critic attack conditions
- **Vanilla PPO** shows instability with sharp reward fluctuations across training
- **R-PPO** exhibits over-exploration leading to suboptimal performance
- **WocaR-RL** maintains stable training curves while achieving high mean rewards

## Technical Details

### WocaR-RL Framework

The WocaR-RL algorithm addresses robustness through:

1. **Worst-case Value Estimation**: Estimates worst-case cumulative reward without explicit attacker training
2. **Adversarial Action Space**: Defines actions reachable under state perturbations within ε-ball
3. **Convex Relaxation**: Uses neural network bounds to approximate worst-case actions
4. **State Importance Weighting**: Prioritizes critical states based on action value gaps

### Attack Methods

- **None**: Standard training without adversarial perturbations
- **Random**: Gaussian noise injection with fixed variance
- **Critic**: Fast Gradient Sign Method (FSGM) attacks on value function

## Files Description

- **`run.py`**: Main training script handling experiment setup, model training, and result logging
- **`trainer.py`**: Core `Trainer` class implementing PPO variants and WocaR-RL logic
- **`utils.py`**: Utility functions for experiment ID generation and argument parsing
- **`evaluate.py`**: Model evaluation and testing utilities
- **`results.py`**: Results analysis and visualization tools
- **`test.py`**: Testing framework for trained models

## Dependencies

Core requirements:
- Python 3.7+
- PyTorch 2.1.0
- Gymnasium 0.29.1
- NumPy 1.26.4
- MuJoCo (via gymnasium[mujoco])

See `requirements.txt` for complete dependency list.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{muneton2024robust,
  title={Robust Reinforcement Learning: Formulating an Optimal Destabilization Policy},
  author={Juan Muneton},
  year={2024},
  institution={Stanford University},
  note={CS234 Project Report}
}
```

## References

- Liang, Y., et al. "Efficient Adversarial Training without Attacking: Worst-Case-Aware Robust Reinforcement Learning." arXiv:2210.05927, 2022.
- Rahman, M. M., & Xue, Y. "Robust Policy Optimization in Deep Reinforcement Learning." arXiv:2212.07536, 2022.
- Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.

## License

This project is developed for academic research purposes. Please refer to individual component licenses for commercial use.

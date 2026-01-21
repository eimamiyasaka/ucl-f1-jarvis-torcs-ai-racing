# TORCS Reinforcement Learning - Lap Time Optimization

Deep RL system for training AI agents to minimize lap time from standing start in TORCS racing simulator.

## Overview

This project uses **PPO (Proximal Policy Optimization)** to train a neural network that controls steering, acceleration, and braking to achieve the fastest possible lap time from a standing start.

### Key Features

- **Deep RL from scratch**: Learn optimal driving policy directly from sensors
- **Track sensor observation**: 19 track edge sensors + speed, position, angle (30D state space)
- **Continuous control**: 3D action space (steering, accel, brake)
- **Reward shaping**: Progress-based rewards with penalties for off-track/crashes
- **Automatic monitoring**: Tensorboard logging, lap time tracking, checkpointing
- **Easy evaluation**: Compare multiple models, generate statistics

## Architecture

```
Observation (30D) → Neural Network (256x256) → Action (3D)
                    ↓
            [steering, accel, brake]
```

**Observation Space (30 dimensions):**
- speedX, speedY, speedZ (3)
- angle (1)
- trackPos (1)
- 19 track edge sensors (19)
- rpm (1)
- gear (1)
- wheelSpinVel (4)

**Action Space (3 dimensions):**
- steering: [-1, 1]
- acceleration: [0, 1]
- brake: [0, 1]

## Installation

### Prerequisites

1. TORCS simulator installed and configured
2. Python 3.7+
3. TORCS server running with scr_server module

### Install Dependencies

```bash
pip install -r requirements_rl.txt
```

Required packages:
- stable-baselines3[extra] >= 2.0.0
- torch >= 1.13.0
- gym >= 0.21.0
- numpy
- tensorboard

## Quick Start

### 1. Test the Environment

```bash
python torcs_rl_env.py
```

This will run a quick test with random actions to verify TORCS connection.

### 2. Start Training

**Basic training (500k steps, ~2-4 hours):**
```bash
python train_ppo.py --timesteps 500000
```

**Advanced training with target lap time:**
```bash
python train_ppo.py --timesteps 1000000 --target-time 90.0 --lr 0.0003
```

**Training will stop automatically when:**
- Target lap time is achieved (if specified)
- Total timesteps completed
- User interrupts (Ctrl+C)

### 3. Monitor Training

Open tensorboard to watch training progress:

```bash
tensorboard --logdir logs/ppo_torcs
```

Then open browser to: http://localhost:6006

You'll see:
- Episode reward over time
- Lap times (best/last)
- Value/policy loss
- Learning rate curves

### 4. Evaluate Trained Model

**Evaluate best model:**
```bash
python evaluate_ppo.py models/ppo_torcs/ppo_TIMESTAMP/best_model/best_model --episodes 10
```

**Compare multiple models:**
```bash
python evaluate_ppo.py --compare models/ppo_torcs/*/best_model/best_model --episodes 5
```

## Training Options

### Command Line Arguments

```bash
python train_ppo.py [OPTIONS]
```

**Key options:**
- `--timesteps`: Total training steps (default: 500000)
- `--port`: TORCS server port (default: 3001)
- `--lr`: Learning rate (default: 0.0003)
- `--target-time`: Stop when this lap time is achieved
- `--reward-type`: 'progress' or 'time' (default: progress)
- `--no-normalize`: Disable observation normalization
- `--resume`: Resume from saved model
- `--save-dir`: Model save directory
- `--log-dir`: Tensorboard log directory

### Reward Types

**Progress reward (recommended):**
- Rewards forward movement continuously
- Faster learning, more stable
- Use: `--reward-type progress`

**Time reward (sparse):**
- Only rewards at lap completion
- Slower learning, but directly optimizes lap time
- Use: `--reward-type time`

## Resume Training

To continue training from a checkpoint:

```bash
python train_ppo.py --resume models/ppo_torcs/ppo_TIMESTAMP/final_model.zip --timesteps 500000
```

## File Structure

```
gym_torcs/
├── torcs_rl_env.py         # Gym environment wrapper
├── train_ppo.py            # Training script
├── evaluate_ppo.py         # Evaluation script
├── torcs_jm_par.py         # TORCS client (existing)
├── RL_README.md            # This file
├── requirements_rl.txt     # Python dependencies
├── models/                 # Saved models
│   └── ppo_torcs/
│       └── ppo_TIMESTAMP/
│           ├── best_model/           # Best performing model
│           ├── final_model.zip       # Final trained model
│           ├── checkpoints/          # Periodic checkpoints
│           └── vec_normalize.pkl     # Normalization stats
└── logs/                   # Tensorboard logs
    └── ppo_torcs/
        └── ppo_TIMESTAMP/
```

## Hyperparameter Tuning

Default hyperparameters are optimized for 4-12 hour training. For different scenarios:

**Fast prototyping (1-2 hours):**
```bash
python train_ppo.py --timesteps 200000 --lr 0.001 --n-steps 1024
```

**High quality (1-3 days):**
```bash
python train_ppo.py --timesteps 2000000 --lr 0.0001 --n-steps 4096 --batch-size 128
```

**Fine-tuning existing model:**
```bash
python train_ppo.py --resume path/to/model.zip --timesteps 100000 --lr 0.00005
```

## Troubleshooting

### TORCS Connection Issues

**Error: "Could not connect to TORCS server"**

1. Ensure TORCS is running
2. Start a race with scr_server driver
3. Check port number matches (default: 3001)
4. On Windows, you may need to manually start TORCS

**Multiple environments:**
If training with multiple parallel environments, ensure each has a unique port.

### Training Not Improving

**Agent drives off track immediately:**
- Check observation normalization is enabled
- Reduce learning rate: `--lr 0.0001`
- Increase entropy coefficient for more exploration

**DNFs too frequent:**
- Review reward function penalties
- Add more progress reward weight
- Ensure off_track_threshold isn't too strict (default: 1.3)

**Lap times plateau:**
- Try time-based reward: `--reward-type time`
- Reduce learning rate for fine-tuning
- Resume training from best checkpoint

## Performance Expectations

**After 100k steps (~30 min):**
- Agent should stay on track
- Complete laps occasionally
- Lap times: 120-150s

**After 500k steps (~2-4 hours):**
- Consistent lap completion
- Lap times: 80-100s
- Few DNFs

**After 1M+ steps (~8+ hours):**
- Optimized racing lines
- Lap times: 60-80s (track dependent)
- High completion rate (>90%)

## Advanced Usage

### Custom Reward Function

Edit `torcs_rl_env.py`, method `_calculate_reward()`:

```python
def _calculate_reward(self, S, done, dnf_reason):
    # Your custom reward logic here
    reward = 0.0

    # Example: Add cornering bonus
    track_pos = abs(S.get('trackPos', 0))
    if track_pos < 0.3:  # On racing line
        reward += 0.1

    return reward
```

### Modify Network Architecture

Edit `train_ppo.py`, in the PPO model creation:

```python
policy_kwargs=dict(
    net_arch=[dict(pi=[512, 512, 256], vf=[512, 512, 256])]  # Larger network
)
```

### Change Observation Space

Edit `torcs_rl_env.py`, method `__init__()` to add/remove sensors.

## Tips for Best Results

1. **Start with default settings** - They're tuned for this task
2. **Monitor tensorboard** - Watch for instability or plateaus
3. **Use progress rewards** - More stable than sparse time-based rewards
4. **Enable normalization** - Critical for stable learning
5. **Set target lap time** - Stop training automatically when goal reached
6. **Save frequently** - Checkpoints every 20k steps by default
7. **Evaluate often** - Check best_model performance during training
8. **Compare models** - Use evaluation script to find best checkpoint

## Citation

Based on:
- PPO: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
- Stable-Baselines3: Raffin et al. (2021)
- TORCS: The Open Racing Car Simulator

## Support

For issues or questions:
1. Check tensorboard logs for training curves
2. Review evaluation statistics
3. Adjust hyperparameters based on performance
4. Ensure TORCS server is running correctly

---

**Happy Racing! 🏎️**

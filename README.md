`IsaacLiteEnv` is a **roguelike reinforcement learning environment** inspired by *The Binding of Isaac*.  
It’s built with **Gymnasium**, **Stable-Baselines3**, and **Pygame**, providing a compact yet extensible setup for training agents to **move, explore, and survive** in procedurally generated rooms.

---

##  Project Structure

isaac_lite_project/

│
├── configs/

    │ ├── env.yaml

    │ ├── imitate.yml

    │ ├── ppo.yaml

│
├── eval_logs/

│
├── isaac_lite/

    │ ├── init.py

    │ ├── env.py

    │ ├── game.py

│
├── logs/

    │ ├── a2c_explorer/

    │ ├── a2c_survivor/

    │ ├── ppo_explorer/

    │ ├── ppo_survivor/

    │ ├── human_sessions/

    │ └── ppo_bc_pretrained_survivor.zip

│
├── models/

├── notebooks/

├── results/

├── runs/

    │ └── a2c_explorer_seed7.zip

│
├── src/

    │ ├── eval.py

    │ ├── imitate.py

    │ ├── quick_train.py

    │ ├── solo.py

    │ ├── train.py

    │ └── watch.py

│
├── venv/

    ├── requirements.txt

    └── README.md

---


##  Features

- **Gymnasium-compatible**  
- **Two Personas:** `explorer` and `survivor` (different reward logic)  
- **Temporary powerups** (*speed*, *damage*)  
- **Boost-based rewards**  
- **Victory confetti / death fade animations**  
- **PPO and A2C **

---


### Setup Instructions

## 1. Clone and enter the project
```bash
git clone https://github.com/teninator/isaac-lite.git
cd isaac_lite_project
```

## 2. Create and activate a virtual environment
On PowerShell (Windows)
```
python -m venv venv
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
venv\Scripts\Activate.ps1
```

If you see:

    "running scripts is disabled on this system"

Run:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then:
```
venv\\Scripts\\Activate.ps1
```

On macOS/Linux
```
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies
```
pip install -r requirements.txt
```

If you have trouble building pygame, use:

```
pip install pygame==2.6.1
```

#### Personas

| Persona    | Focus     | Encouraged Behavior                         |
| ---------- | --------- | ------------------------------------------- |
| `survivor` | Defensive | Stay alive, avoid taking hits               |
| `explorer` | Curious   | Move often, explore rooms, collect powerups |


Example:

```
env = IsaacLiteEnv(persona='explorer')
```
Or set persona dynamically during training/watch runs:

```
python src/train.py --algo ppo --persona survivor
python src/watch.py --persona explorer
```

## Training

You can quickly start PPO training with:

```
python src/quick_train.py
```
or a full config run using:
```
python src/train.py --algo ppo --timesteps 200000 --seed 7 --persona explorer --logdir logs/ppo_explorer
```

## Command flags
| Flag          | Description               | Default    |
| ------------- | ------------------------- | ---------- |
| `--algo`      | Algorithm (`ppo` / `a2c`) | `ppo`      |
| `--persona`   | Reward persona            | `explorer` |
| `--timesteps` | Training steps            | `100000`   |
| `--logdir`    | Log output directory      | `logs/`    |


Trained models are stored in:

```
logs/{algo}_{persona}/
```

These works with:

    stable-baselines3 (PPO, A2C)

    tensorboard --logdir logs

Watching the Agent

To visualize the trained models:

```
python src/watch.py
```

You’ll be prompted to select from detected runs, or you can specify via:

```
python src/watch.py --persona explorer
```

## Keyboard Shortcuts
```
Key	Action
ESC	Quit playback
Action & Observation Spaces
Action Space
```

## Observation Space
```
Box(shape=(20,), dtype=float32)
Range	Description
[0:3]	Player (x, y, health)
[3:12]	Enemies (3× [x, y, alive])
[12:18]	Powerups (2× [x, y, exists])
[18:20]	Active Boosts ([damage, speed])
```
# Experiments & Results
## Commands

# Experiment 1 — PPO vs A2C (Explorer)
```
python src/train.py --algo ppo --persona explorer --logdir logs/ppo_explorer
```
```
python src/train.py --algo a2c --persona explorer --logdir logs/a2c_explorer
```

#  Experiment 2 — PPO (Survivor) vs PPO (Explorer)
```
python src/train.py --algo ppo --persona survivor --logdir logs/ppo_survivor
```

```
python src/train.py --algo ppo --persona explorer  --logdir logs/ppo_explorer
```


# Experiment 3 — PPO Seed Sweep
```
python src/train.py --algo ppo --persona explorer --seed %S --logdir logs/ppo_explorer
```


| Experiment           | Metric         | Finding                           |
| -------------------- | -------------- | --------------------------------- |
| PPO vs A2C           | Avg reward     | PPO converged faster              |
| Explorer vs Survivor | Steps survived | Survivor lived longer             |
| PPO seed sweep       | Variance       | Low variance, stable across seeds |

## Powerups
```
Type	Effect	Duration	Reward
Speed	×1.5 player speed	200 ticks	+3
Damage	×1.5 player damage	200 ticks	+5
```

## Evaluation & Imitation
You can also evaluate or clone behaviour from trained models:
```
python src/eval.py
python src/imitate.py
```
Example imitation training config (see configs/imitate.yaml):
```
data_path: logs/human_sessions/session.json
persona: survivor
seed: 7
save_path: logs/ppo_bc_pretrained_survivor.zip
```

## Troubleshooting
**PowerShell “scripts disabled”**

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Missing modules**
```
pip install stable-baselines3==2.2.0 gymnasium pygame numpy
```

**Numpy core error**
```
pip install --upgrade numpy==1.26.4
```

**To confirm you’re in the correct venv**
```
where python
```

It **should** point to:
```
.../isaac_lite_project/venv/Scripts/python.exe
```

## Logging with Tensorboards & Results

Training and evaluation logs are automatically saved under /logs.
Then launch TensorBoard to visualize performance:

```
tensorboard --logdir logs
```

**Example Manual Loop**

```
import gymnasium as gym
from isaac_lite.env import IsaacLiteEnv

env = IsaacLiteEnv(persona='survivor')
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    env.render()
env.close()
```
If events.out.tfevents files are missing:

- Ensure training used a valid --logdir.

- Run training from inside the venv (where python should show your venv path).

- Each algorithm/persona combo logs to its own folder

# Credits

Developed by group ***Minecraft*** Teni Adegbite(100861337), Jerico Robles(100876635), Jugal Patel(100744586), Nadman Khan(100785940)





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

```
python -m venv venv
```

On PowerShell (Windows)

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

The persona system modifies the reward logic to produce different agent behaviours.
Persona	Focus	Encouraged Behavior
survivor	Defensive	Stay alive, avoid damage
explorer	Curious	Move, explore rooms, collect powerups

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
or a full configuration run:

```
python src/train.py --algo ppo --timesteps 200000 --seed 7 --persona explorer --logdir logs/ppo_explorer
```

## Arguments
Flag	Description	Default
```--algo	Algorithm (ppo / a2c)	ppo
--persona	Persona mode	explorer
--timesteps	Training steps	100000
--logdir	Output directory	logs/
```

Trained models are stored in:

```
logs/{algo}_{persona}/
```

Compatible with:

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

## Logging & Results

Training and evaluation logs are automatically saved under /logs.
Launch TensorBoard to visualize performance:

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

# Credits

Developed by group ***Minecraft*** (Teni Adegbite, Jerico Robles, 




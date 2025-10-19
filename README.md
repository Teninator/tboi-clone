# IsaacLiteEnv 🧩

`IsaacLiteEnv` is a lightweight, roguelike-style Gymnasium environment inspired by *The Binding of Isaac*.  
It features simple discrete controls, procedural rooms, enemies, collectible boosts, and reward shaping designed for reinforcement learning experiments.

---

## 🚀 Features

- **Discrete action space** (9 actions) — move, shoot, idle, etc.  
- **Dynamic enemies** with individual kill tracking  
- **Temporary boosts (powerups)** — *speed* and *damage*  
- **Reward shaping** for exploration, aggression, and survival  
- **HUD with live stats** — score, kills, boosts, and steps  
- **Death fade animation** when the player dies  

---

## 🧩 Setup Instructions

### 1. Clone and enter the project

```bash
git clone https://github.com/your-repo/isaac_lite_project.git
cd isaac_lite_project
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

#### On **PowerShell (Windows)**:
If you get the error  
> running scripts is disabled on this system  

Run this once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Then activate:
```powershell
venv\Scripts\Activate.ps1
```

#### On **Command Prompt**:
```bash
venv\Scripts\activate.bat
```

#### On **macOS/Linux**:
```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you face issues building `pygame`, try:
```bash
pip install pygame==2.6.1
```

---

## 🧠 Training the Agent

### Quick Train Script
You can quickly test PPO training using:

```bash
python quick_train.py
```

or a custom run:

```bash
python train.py --algo ppo --timesteps 200000 --seed 7 --persona survivor --logdir logs/ppo_survivor
```

✅ Models will be saved automatically in the `/logs` directory.  
✅ Compatible with `stable-baselines3` PPO and A2C agents.  

---

## 🎮 Watching the Agent Play

To visualize the trained model:

```bash
python watch.py
```

The agent will load the most recent model from `/logs` and play automatically using `pygame`.

---

## 🧩 Using the Environment Manually

```python
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

---

##  Observation & Action Spaces

| Type | Description | Shape |
|------|--------------|--------|
| **Observation** | Player x, y, HP, and placeholder features | `(obs_dim,)` |
| **Action** | Discrete 9 (move, shoot, idle, etc.) | `(1,)` |

---

##  Powerups

| Type | Effect | Duration (ticks) | Reward |
|------|---------|------------------|--------|
| **Speed** | ×1.5 player speed | 200 | +3 |
| **Damage** | ×1.5 player damage | 200 | +5 |

---

## 🧰 Common Troubleshooting

### ❌ PowerShell cannot run `activate`
Run this once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### ❌ `No module named 'stable_baselines3'`
Ensure dependencies are installed:
```bash
pip install stable-baselines3==2.0.0
```

### ❌ `ModuleNotFoundError: No module named 'numpy._core.numeric'`
This occurs when switching Python environments.  
Run:
```bash
pip install --upgrade numpy==1.26.4
```
and make sure you’re in your **activated venv** before running scripts.

### ✅ To confirm you’re inside the venv:
```bash
where python
```
You should see a path like:
```
C:\Users\<user>\Desktop\isaac_lite_project\venv\Scripts\python.exe
```

---

## 📊 Logging

All training logs and model checkpoints are saved under `/logs`.  
Compatible with **TensorBoard**:
```bash
tensorboard --logdir logs
```

---

##  Credits

Created for testing **Deep Reinforcement Learning in a roguelike setting**, inspired by *The Binding of Isaac*.  
Environment and training scripts by **[Your Name]**, using `gymnasium`, `pygame`, and `stable-baselines3`.

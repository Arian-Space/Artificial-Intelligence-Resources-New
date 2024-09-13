import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Modelo XML mejorado para el coche autónomo
xml = """
<mujoco>
  <compiler autolimits="true"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <mesh name="chasis" scale=".01 .006 .0015"
      vertex=" 9   2   0
              -10  10  10
               9  -2   0
               10  3  -10
               10 -3  -10
              -8   10 -10
              -10 -10  10
              -8  -10 -10
              -5   0   20"/>
  </asset>

  <default>
    <joint damping=".03" actuatorfrcrange="-0.5 0.5"/>
    <default class="wheel">
      <geom type="cylinder" size=".03 .01" rgba=".5 .5 1 1"/>
    </default>
    <default class="decor">
      <site type="box" rgba=".5 1 .5 1"/>
    </default>
  </default>

  <worldbody>
    <geom type="plane" size="3 3 .01" material="grid"/>
    <body name="car" pos="0 0 .03">
      <freejoint/>
      <light name="top light" pos="0 0 2" mode="trackcom" diffuse=".4 .4 .4"/>
      <geom name="chasis" type="mesh" mesh="chasis"/>
      <geom name="front wheel" pos=".08 0 -.015" type="sphere" size=".015" condim="1" priority="1"/>
      <light name="front light" pos=".1 0 .02" dir="2 0 -1" diffuse="1 1 1"/>
      <body name="left wheel" pos="-.07 .06 0" zaxis="0 1 0">
        <joint name="left"/>
        <geom class="wheel"/>
        <site class="decor" size=".006 .025 .012"/>
        <site class="decor" size=".025 .006 .012"/>
      </body>
      <body name="right wheel" pos="-.07 -.06 0" zaxis="0 1 0">
        <joint name="right"/>
        <geom class="wheel"/>
        <site class="decor" size=".006 .025 .012"/>
        <site class="decor" size=".025 .006 .012"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <fixed name="forward">
      <joint joint="left" coef=".5"/>
      <joint joint="right" coef=".5"/>
    </fixed>
    <fixed name="turn">
      <joint joint="left" coef="-.5"/>
      <joint joint="right" coef=".5"/>
    </fixed>
  </tendon>

  <actuator>
    <motor name="forward" tendon="forward" ctrlrange="-1 1"/>
    <motor name="turn" tendon="turn" ctrlrange="-1 1"/>
  </actuator>

  <sensor>
    <jointactuatorfrc name="right" joint="right"/>
    <jointactuatorfrc name="left" joint="left"/>
  </sensor>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

action_dim = 2  # forward and steering

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp()
        return mu, std, value

def compute_reward(data):
    car_pos = data.qpos[:2]
    target_pos = np.array([10, 0])
    distance_to_target = np.linalg.norm(car_pos - target_pos)
    
    car_velocity = data.sensordata[:3]  # Usando el velocímetro
    speed = np.linalg.norm(car_velocity)
    
    distance_reward = -distance_to_target
    speed_reward = speed if speed < 2 else 2 - (speed - 2)  # Recompensa máxima a 2 m/s
    
    # Penalización por orientación incorrecta
    car_orientation = data.qpos[3:7]  # quaternion
    target_direction = target_pos - car_pos
    target_direction = target_direction / np.linalg.norm(target_direction)
    car_direction = mujoco.mju_rotVecQuat(np.array([1., 0., 0.]), car_orientation)
    orientation_reward = np.dot(car_direction[:2], target_direction)
    
    # Penalización por salirse del camino
    off_track_penalty = -10 if abs(car_pos[1]) > 1.5 else 0
    
    return distance_reward + 0.1 * speed_reward + 2 * orientation_reward + off_track_penalty

def run_episode(model, data, ac_model, max_steps=1000, sim_time=10):
    mujoco.mj_resetData(model, data)
    
    state = np.concatenate([data.qpos, data.qvel, data.sensordata])
    total_reward = 0
    states, actions, rewards, log_probs, values = [], [], [], [], []
  
    for _ in range(max_steps):
        if data.time >= sim_time:
            break
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mu, std, value = ac_model(state_tensor)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Aplicar la acción al coche
        forward, steering = action.squeeze().numpy()
        data.ctrl[0] = forward
        data.ctrl[1] = steering
        
        mujoco.mj_step(model, data)
        
        next_state = np.concatenate([data.qpos, data.qvel, data.sensordata])
        reward = compute_reward(data)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        
        total_reward += reward
        state = next_state
        
        # Terminar el episodio si el coche alcanza el objetivo o se sale demasiado del camino
        if np.linalg.norm(data.qpos[:2] - np.array([10, 0])) < 0.5 or abs(data.qpos[1]) > 2:
            break

    return (
        states, 
        actions, 
        rewards, 
        torch.cat(log_probs),
        torch.cat(values),
        total_reward
    )

def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
    advantages = []
    returns = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1].item()
        delta = rewards[t] + gamma * next_value - values[t].item()
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t].item())
    return advantages, returns

def train_ppo(model, data, episodes=2000, batch_size=64, clip_epsilon=0.2, max_grad_norm=0.5, sim_time=10):
    state_dim = model.nq + model.nv + model.nsensordata
    action_dim = model.nu
    ac_model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(ac_model.parameters(), lr=3e-4)
    
    for episode in range(episodes):
        states, actions, rewards, old_log_probs, values, total_reward = run_episode(
            model, data, ac_model, sim_time=sim_time
        )
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = old_log_probs.detach()
        values = values.detach()
        
        advantages, returns = compute_gae(rewards, values)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalizar ventajas
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimización PPO
        for _ in range(10):  # Número de épocas de optimización
            for i in range(0, len(states), batch_size):
                batch_states = states[i:i+batch_size]
                batch_actions = actions[i:i+batch_size]
                batch_old_log_probs = old_log_probs[i:i+batch_size]
                batch_advantages = advantages[i:i+batch_size]
                batch_returns = returns[i:i+batch_size]
                
                mu, std, new_values = ac_model(batch_states)
                dist = Normal(mu, std)
                new_log_probs = dist.log_prob(batch_actions)
                
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = nn.MSELoss()(new_values.squeeze(), batch_returns)
                
                loss = actor_loss + 0.5 * critic_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac_model.parameters(), max_grad_norm)
                optimizer.step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    
    return ac_model

# Entrenar el modelo
state_dim = model.nq + model.nv + model.nsensordata
action_dim = model.nu
ac_model = ActorCritic(state_dim, action_dim)
sim_time = 30  # Tiempo de simulación en segundos
trained_model = train_ppo(model, data, episodes=2000, sim_time=sim_time)

# Guardar el modelo entrenado
torch.save(trained_model.state_dict(), 'trained_autonomous_car_model.pth')

# Simulación y visualización
duration = sim_time  # Usar el mismo tiempo de simulación que en el entrenamiento
framerate = 60  # Hz

frames = []
mujoco.mj_resetData(model, data)

with mujoco.Renderer(model) as renderer:
    while data.time < duration:
        state = np.concatenate([data.qpos, data.qvel, data.sensordata])
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            mu, _, _ = trained_model(state_tensor)
        action = mu.numpy()[0]
        
        data.ctrl[:] = action
        
        mujoco.mj_step(model, data)
        
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)

fig, ax = plt.subplots()
ax.set_axis_off()
im = ax.imshow(frames[0])

def update(frame):
    im.set_array(frames[frame])
    return [im]

anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/framerate, blit=True)

plt.show()
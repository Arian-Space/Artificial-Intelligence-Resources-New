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
  <option timestep="0.01" integrator="RK4"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="road" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512" mark="edge" markrgb="0.8 0.8 0.8"/>
    <material name="road" texture="road" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
    <material name="car" rgba="1 0 0 1"/>
    <material name="wheel" rgba="0.3 0.3 0.3 1"/>
  </asset>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="50 50 0.1" material="road" friction="1 0.005 0.0001"/>
    <site name="target" pos="10 0 0.1" size="0.5 0.5 0.1" rgba="0 1 0 1" type="cylinder"/>
    <body name="car" pos="0 0 0.1">
      <joint name="free" type="free"/>
      <geom name="chassis" type="box" size="0.4 0.2 0.1" material="car"/>
      <body name="front_left_wheel" pos="0.3 0.25 0">
        <joint name="front_left_axle" type="hinge" axis="0 1 0"/>
        <joint name="front_left_steer" type="hinge" axis="0 0 1" limited="true" range="-0.5 0.5"/>
        <geom type="cylinder" size="0.1 0.05" material="wheel" euler="1.57 0 0"/>
      </body>
      <body name="front_right_wheel" pos="0.3 -0.25 0">
        <joint name="front_right_axle" type="hinge" axis="0 1 0"/>
        <joint name="front_right_steer" type="hinge" axis="0 0 1" limited="true" range="-0.5 0.5"/>
        <geom type="cylinder" size="0.1 0.05" material="wheel" euler="1.57 0 0"/>
      </body>
      <body name="rear_left_wheel" pos="-0.3 0.25 0">
        <joint name="rear_left_axle" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.1 0.05" material="wheel" euler="1.57 0 0"/>
      </body>
      <body name="rear_right_wheel" pos="-0.3 -0.25 0">
        <joint name="rear_right_axle" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.1 0.05" material="wheel" euler="1.57 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="front_left_steer" name="steer_left" gear="1" ctrlrange="-1 1"/>
    <motor joint="front_right_steer" name="steer_right" gear="1" ctrlrange="-1 1"/>
    <velocity joint="rear_left_axle" name="velocity_left" gear="1" ctrlrange="-1 1"/>
    <velocity joint="rear_right_axle" name="velocity_right" gear="1" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

action_dim = 3  # steering, left velocity, right velocity

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp()
        return mu, std, value

def compute_reward(data):
    # Posición del coche
    car_pos = data.qpos[:2]
    
    # Posición del objetivo
    target_pos = np.array([10, 0])
    
    # Distancia al objetivo
    distance_to_target = np.linalg.norm(car_pos - target_pos)
    
    # Velocidad del coche
    car_velocity = data.qvel[:2]
    
    # Recompensas
    distance_reward = -distance_to_target
    speed_reward = np.linalg.norm(car_velocity)
    
    # Penalización por salirse de la carretera
    road_width = 4
    if abs(car_pos[1]) > road_width / 2:
        off_road_penalty = -10
    else:
        off_road_penalty = 0
    
    reward = distance_reward + 0.1 * speed_reward + off_road_penalty
    
    return reward

def run_episode(model, data, ac_model, max_steps=1000, sim_time=10):
    mujoco.mj_resetData(model, data)
    
    state = np.concatenate([data.qpos, data.qvel])
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
        data.ctrl[:] = action.squeeze().numpy()
        
        mujoco.mj_step(model, data)
        
        next_state = np.concatenate([data.qpos, data.qvel])
        reward = compute_reward(data)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        
        total_reward += reward
        state = next_state
        
        # Terminar el episodio si el coche alcanza el objetivo
        if np.linalg.norm(data.qpos[:2] - np.array([10, 0])) < 0.5:
            break

    return (
        states, 
        actions, 
        rewards, 
        torch.cat(log_probs),
        torch.cat(values),
        total_reward
    )

# Las funciones compute_gae y train_ppo permanecen iguales que en el ejemplo original

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

def train_ppo(model, data, episodes=1000, batch_size=32, clip_epsilon=0.2, max_grad_norm=0.5, sim_time=10):
    state_dim = model.nq + model.nv
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
            mu, std, new_values = ac_model(states)
            dist = Normal(mu, std)
            new_log_probs = dist.log_prob(actions)
            
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac_model.parameters(), max_grad_norm)
            optimizer.step()
        
        print(f"Episode {episode}, Total Reward: {total_reward}")
    
    return ac_model

# Entrenar el modelo
state_dim = model.nq + model.nv
action_dim = model.nu
ac_model = ActorCritic(state_dim, action_dim)
sim_time = 30  # Tiempo de simulación en segundos
trained_model = train_ppo(model, data, episodes=1000, sim_time=sim_time)

# Guardar el modelo entrenado
torch.save(trained_model.state_dict(), 'trained_autonomous_car_model.pth')

# Simulación y visualización
duration = sim_time  # Usar el mismo tiempo de simulación que en el entrenamiento
framerate = 60  # Hz

frames = []
mujoco.mj_resetData(model, data)

with mujoco.Renderer(model) as renderer:
    while data.time < duration:
        state = np.concatenate([data.qpos, data.qvel])
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
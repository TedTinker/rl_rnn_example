#%% 
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import gym as gymnasium
import matplotlib.pyplot as plt

from utils import RecurrentReplayBuffer, plot_durations
from model import DQN

# Constants and Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU = 0.005
GAMMA = 0.99
RANDOM_ACTION_THRESHOLD_INIT = 1
RANDOM_ACTION_DECAY = 0.9999
RANDOM_ACTION_THRESHOLD = RANDOM_ACTION_THRESHOLD_INIT
EPISODES = 10000

# Model, buffer, and optimizer setup
buffer = RecurrentReplayBuffer()
policy_dqn = DQN().to(device)
target_dqn = DQN().to(device)
target_dqn.load_state_dict(policy_dqn.state_dict())
optimizer = optim.Adam(policy_dqn.parameters(), lr=0.001)
criterion = nn.SmoothL1Loss()

# Environment setup
env = gymnasium.make("CartPole-v1")
episode_durations = []



def train():
    """
    Train the model using experience from the recurrent replay buffer.
    """
    batch = buffer.sample()
    if(batch == False): return
    all_observation, action, reward, done, mask = batch

    # Create the 'previous action' tensor with a 'no-op' or 'zero' action at the start
    # Assuming action is a zero tensor of shape (batch_size, steps, value).
    prev_action = torch.cat([torch.zeros((action.shape[0], 1, action.shape[2])), action[:, :-1, :]], dim=1)

    # Split all_observation tensor to get the 'state' and 'next_state' tensors
    state = all_observation[:, :-1, :]
    next_state = all_observation[:, 1:, :]

    # Predicted values: Q^\pi (s_t, a_t)
    state_action_values, _ = policy_dqn(state, prev_action)
    state_action_values = state_action_values.gather(-1, action.type(torch.int64))
    state_action_values *= mask

    # Target values: r + \gamma Q^\pi(s_t+1, \pi(s_t+1))
    with torch.no_grad():
        # Create next_prev_action tensor from action tensor
        next_prev_action = action

        next_state_values, _ = target_dqn(next_state, next_prev_action)
        next_state_values = next_state_values.max(-1)[0].unsqueeze(-1)
        expected_state_action_values = (next_state_values * GAMMA) + reward
        expected_state_action_values *= ~done.type(torch.bool)
        expected_state_action_values *= mask

    # Compute loss and optimize
    loss = criterion(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_dqn.parameters(), 100)
    optimizer.step()

    # Soft update target network: \bar{\theta} <- \tau\theta + (1 - \tau) \bar{\theta}
    target_dqn_state_dict = target_dqn.state_dict()
    policy_dqn_state_dict = policy_dqn.state_dict()
    for key in policy_dqn_state_dict:
        target_dqn_state_dict[key] = policy_dqn_state_dict[key] * TAU + target_dqn_state_dict[key] * (1 - TAU)
    target_dqn.load_state_dict(target_dqn_state_dict)
    
    

def select_action(state, prev_action, h):
    """
    Select an action, either randomly or based on the current state.
    """
    global RANDOM_ACTION_THRESHOLD
    
    with torch.no_grad():
        action, h = policy_dqn(state, prev_action, h)
        action = action.squeeze(0)
        action = action.max(1)[1].view(1, 1)

    if random.random() > RANDOM_ACTION_THRESHOLD: pass
    else: action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    RANDOM_ACTION_THRESHOLD *= RANDOM_ACTION_DECAY

    return action, h



def run_episode(render = False, episode_num = 0):
    """
    Execute one episode of the environment.
    """
    state = env.reset()
    state = [state[0], state[2]]
    state = torch.tensor(state, device=device).unsqueeze(0).unsqueeze(0)
    t = 0
    h = torch.zeros([1,1,8])
    prev_action = torch.zeros([1,1])
    done = False

    while not done:
        if render: 
            frame = env.render(mode="rgb_array")
            plt.axis('off')
            plt.imshow(frame)
            plt.savefig(f'plots/{episode_num}/frame_{t}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        
        t += 1
        action, h = select_action(state, prev_action.unsqueeze(0), h)
        next_state, reward, done, _ = env.step(action.item())
        next_state = [next_state[0], next_state[2]]
        next_state = torch.tensor(next_state, device=device).unsqueeze(0).unsqueeze(0)
        reward = torch.tensor([-1 if done else 1], device=device)
        buffer.push(state, action, reward, next_state, done, done)
        state = next_state
        prev_action = action
        train()
        
        if done:
            episode_durations.append(t + 1)
            
            

if __name__ == '__main__':
    for episode in range(EPISODES):
        print(episode, end = '... ')
        if episode % 25 == 0:
            try: os.mkdir('plots/{}'.format(episode))
            except: pass
            run_episode(render = True, episode_num = episode)
            plot_durations(episode_durations, episode)
        else:
            run_episode()
## %%

# %%

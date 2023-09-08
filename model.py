#%%
import torch
import torch.nn as nn
from torchinfo import summary as torch_summary

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) Model. Q^\pi approximates Q^* by estimating r + \gamma Q^\pi (s_t+1, \pi(s_t+1))
    """
    def __init__(self, n_states=2, n_actions=2, hidden_size = 8):
        """
        Initialize the DQN model.
        
        Args:
        - n_states (int): Number of state inputs.
        - n_actions (int): Number of action outputs.
        - hidden size (int): Size of hidden states.
        """
        super().__init__()
        
        self.gru = nn.GRU(
            input_size =  n_states + 1,
            hidden_size = hidden_size,
            batch_first = True)
        
        self.network = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, state, prev_action, h = None):
        """
        Forward pass through the network.
        """
        x = torch.cat([state, prev_action], dim = -1)
        h = h.permute(1, 0, 2) if h != None else h
        h, _ = self.gru(x, h)  
        
        return self.network(h), h

if __name__ == "__main__":
    # Initialize and print the DQN model
    dqn = DQN()
    print(dqn)
    
    # Print the model summary
    print("\nModel Summary:")
    print(torch_summary(dqn, ((3, 10, 2),(3, 10, 1))))
# %%

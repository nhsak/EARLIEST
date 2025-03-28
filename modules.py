import torch
import torch.nn as nn
from torch.distributions import Bernoulli

class BaselineNetwork(nn.Module):
    """
    A network which predicts the average reward observed
    during a Markov decision-making process.
    Weights are updated w.r.t. the mean squared error between
    its prediction and the observed reward.
    """
    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Detach the input for the baseline network to avoid gradient propagation
        b = self.fc(x.detach())
        return b


class Controller(nn.Module):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, ninp, nout):
        super(Controller, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(ninp, nout)  # Optimized w.r.t. reward

        torch.nn.init.normal_(self.fc.bias, mean=-10, std=1e-1)

        

    def forward(self, x):
        # Pass input through the controller network
        
        probs = torch.sigmoid(self.fc(x))
        print(self.fc.bias)
        
        # Exploration vs exploitation adjustment
        probs = (1 - self._epsilon) * probs + self._epsilon * torch.FloatTensor([0.05]).to("cuda")  
        
        # Bernoulli distribution for the action
        m = Bernoulli(probs=probs)
        action = m.sample()  # Sample an action
        log_pi = m.log_prob(action)  # Compute log probability of sampled action
        
        # Return the action, log probability, and the negative log probability (for loss calculation)
        return action.squeeze(0), log_pi.squeeze(0), -torch.log(probs).squeeze(0)
